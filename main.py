import re
import random
import time
import json
import os
from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from collections import Counter
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_process_text(text):
    text = process_text(text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def create_custom_mapping(train_data_path, min_freq=10, max_classes=3000):
    try:
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"The file {train_data_path} does not exist.")
        
        with open(train_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Type of loaded data: {type(data)}")
        print("Keys in the loaded data:")
        print(json.dumps(list(data.keys()), indent=2))
        
        all_answers = []
        for img_id, answers in data['answers'].items():
            if isinstance(answers, list):
                all_answers.extend([ans['answer'].lower() for ans in answers if isinstance(ans, dict) and 'answer' in ans])
        
        if not all_answers:
            print("Warning: No answers were found in the data.")
            return None
        
        print(f"Total number of answers found: {len(all_answers)}")
        print("Sample answers:")
        print(all_answers[:10])
        
        answer_counts = Counter(all_answers)
        
        class_mapping = {
            'unanswerable': 0,
            'unusable image': 1,
            'yes': 2,
            'no': 3
        }
        
        idx = len(class_mapping)
        for answer, count in answer_counts.most_common():
            if count >= min_freq and idx < max_classes:
                if answer not in class_mapping:
                    class_mapping[answer] = idx
                    idx += 1
        
        class_mapping['[NUMBER]'] = idx
        
        print(f"Total number of classes: {len(class_mapping)}")
        print("Sample of class mapping:")
        print(json.dumps(dict(list(class_mapping.items())[:10]), indent=2))
        
        return class_mapping

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        import traceback
        traceback.print_exc()
        return None

class AdvancedVQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, max_answers=10):
        self.transform = transform
        self.image_dir = image_dir
        with open(df_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.answer = answer and 'answers' in self.data
        self.max_answers = max_answers
        self.image_ids = list(self.data['image'].keys())

        print(f"Type of loaded data: {type(self.data)}")
        print("Keys in the loaded data:")
        print(json.dumps(list(self.data.keys()), indent=2))

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for img_id in self.data['question']:
            self.data['question'][img_id] = advanced_process_text(self.data['question'][img_id])
            question = self.data['question'][img_id]
            words = question.split()
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            class_mapping_df = pd.read_csv("./data/custom_class_mapping.csv")
            self.answer2idx = {row["answer"]: row["class_id"] for _, row in class_mapping_df.iterrows()}
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

            print(f"Number of unique answers: {len(self.answer2idx)}")
            print(f"Sample answers: {list(self.answer2idx.items())[:10]}")

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image_path = f"{self.image_dir}/{self.data['image'][img_id]}"
        question = self.data['question'][img_id]
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image) if self.transform else image

        if self.answer:
            answers = self.data['answers'].get(img_id, [])
            answer_indices = []
            for ans in answers:
                processed_ans = process_text(ans['answer'])
                if processed_ans.isdigit():
                    answer_indices.append(self.answer2idx['[NUMBER]'])
                else:
                    answer_indices.append(self.answer2idx.get(processed_ans, self.answer2idx['unanswerable']))
            
            answer_indices = answer_indices[:self.max_answers]
            answer_indices += [-1] * (self.max_answers - len(answer_indices))
            
            mode_answer_idx = max(set(answer_indices), key=answer_indices.count) if answer_indices else -1
            return image, question, torch.tensor(answer_indices), torch.tensor(mode_answer_idx)
        else:
            return image, question, torch.tensor([]), torch.tensor(-1)

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    images, questions, answer_indices, mode_answers = zip(*batch)
    images = torch.stack(images)
    questions = list(questions)
    answer_indices = torch.stack(answer_indices)
    mode_answers = torch.stack(mode_answers)
    return images, questions, answer_indices, mode_answers

class AdvancedVQAModel(nn.Module):
    def __init__(self, device, num_answers):
        super().__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.bert_model = BertModel.from_pretrained('bert-large-uncased').to(device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        
        self.num_answers = num_answers
        
        # Projection layers
        self.clip_projection = nn.Linear(self.clip_model.config.projection_dim, 1024).to(device)
        self.bert_projection = nn.Linear(self.bert_model.config.hidden_size, 1024).to(device)
        
        # Multi-modal Transformer
        self.multimodal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, dropout=0.1),
            num_layers=4
        ).to(device)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_answers)
        ).to(device)

    def forward(self, image, question):
        # 画像の値を[0, 1]の範囲に収める
        image = torch.clamp(image, 0, 1)
        
        # CLIPの前処理を手動で行う
        image = self.processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(self.device)
        
        # テキストの処理
        text_inputs = self.processor.tokenizer(question, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        
        # CLIPモデルに入力
        with torch.no_grad():
            clip_outputs = self.clip_model(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask, pixel_values=image)
        
        # BERTの処理
        bert_inputs = self.bert_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            bert_outputs = self.bert_model(**bert_inputs)
        
        image_features = self.clip_projection(clip_outputs.image_embeds)
        text_features = self.bert_projection(bert_outputs.last_hidden_state)
        
        # Combine features
        combined_features = torch.cat([image_features.unsqueeze(1), text_features], dim=1)
        
        # Pass through multi-modal transformer
        transformed_features = self.multimodal_transformer(combined_features)
        
        # Get output
        output = self.output_layer(transformed_features.mean(dim=1))
        
        return output

def VQA_score(pred, answers):
    if pred in answers:
        return min(answers.count(pred) / 3, 1)
    else:
        return 0

def train(model, dataloader, optimizer, criterion, device, idx2answer, scheduler, scaler):
    model.train()
    total_loss = 0
    total_vqa_score = 0
    total_acc = 0

    for batch_idx, (image, question, answers, mode_answer) in enumerate(tqdm(dataloader)):
        image = image.to(device).float()
        answers = answers.to(device)
        mode_answer = mode_answer.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(image, question)
            loss = criterion(pred, mode_answer)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        
        pred_answers = pred.argmax(1).cpu().numpy()
        batch_vqa_score = 0
        for pred_ans, ans_list in zip(pred_answers, answers.cpu().numpy()):
            pred_ans_str = idx2answer[pred_ans]
            ans_list_str = [idx2answer[a] for a in ans_list if a != -1]
            batch_vqa_score += VQA_score(pred_ans_str, ans_list_str)
        total_vqa_score += batch_vqa_score / len(pred_answers)
        
        total_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    num_batches = len(dataloader)
    return (total_loss / num_batches, 
            total_vqa_score / num_batches, 
            total_acc / num_batches)

def eval(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for image, question, _, _ in tqdm(dataloader):
            image = image.to(device).float()

            pred = model(image, question)
            predictions.extend(pred.argmax(1).cpu().numpy())

    return predictions

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Advanced data augmentation
    advanced_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ])
    
    train_data_path = './data/train.json'
    custom_mapping = create_custom_mapping(train_data_path)
    if custom_mapping is None:
        print("Failed to create custom mapping. Exiting.")
        return

    df = pd.DataFrame(list(custom_mapping.items()), columns=['answer', 'class_id'])
    df.to_csv('./data/custom_class_mapping.csv', index=False)
    
    train_dataset = AdvancedVQADataset(df_path="./data/train.json", image_dir="./data/train", transform=advanced_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)

    val_dataset = AdvancedVQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=advanced_transform, answer=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = AdvancedVQAModel(device, len(train_dataset.answer2idx)).to(device)

    num_epochs = 30
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    best_val_score = float('-inf')
    patience = 5
    patience_counter = 0
    best_model_path = "best_vizwiz_vqa_model.pth"

    for epoch in range(num_epochs):
        train_loss, train_vqa_score, train_acc = train(model, train_loader, optimizer, criterion, device, train_dataset.idx2answer, scheduler, scaler)
        val_predictions = eval(model, val_loader, device)
        
        # バリデーションスコアの計算（実際のVQAスコア計算に置き換えてください）
        val_score = sum(val_predictions) / len(val_predictions)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, VQA Score: {train_vqa_score:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Score: {val_score:.4f}")
        print("-" * 50)
        
        # 毎エポックでモデルを保存
        torch.save(model.state_dict(), f"vizwiz_vqa_model_epoch_{epoch+1}.pth")
        print(f"Model saved for epoch {epoch+1}")

        # Early Stopping
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # 最良のモデルをロード（ファイルが存在する場合）
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print(f"Best model file {best_model_path} not found. Using the last trained model.")

    model.eval()
    
    # Final prediction generation
    val_predictions = eval(model, val_loader, device)
    submission = [train_dataset.idx2answer[pred] for pred in val_predictions]

    np.save("vizwiz_vqa_submission.npy", submission)
    print("Predictions saved as 'vizwiz_vqa_submission.npy'")

if __name__ == "__main__":
    main()