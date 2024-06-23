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
from torch.optim.lr_scheduler import CosineAnnealingLR

from collections import Counter
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

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

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, max_answers=10):
        self.transform = transform
        self.image_dir = image_dir
        with open(df_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.answer = answer
        self.max_answers = max_answers
        self.image_ids = list(self.data['image'].keys())

        print(f"Type of loaded data: {type(self.data)}")
        print("Keys in the loaded data:")
        print(json.dumps(list(self.data.keys()), indent=2))

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.data['question'].values():
            question = process_text(question)
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
        
        image = Image.open(image_path)
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
            
            mode_answer_idx = max(set(answer_indices), key=answer_indices.count)
            return image, question, torch.tensor(answer_indices), torch.tensor(mode_answer_idx)
        else:
            return image, question, torch.tensor([]), torch.tensor(-1)  # ダミーの answer_indices と mode_answer_idx を返す

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    images, questions, answer_indices, mode_answers = zip(*batch)
    images = torch.stack(images)
    questions = list(questions)  # questions are kept as a list of strings
    answer_indices = torch.stack(answer_indices)
    mode_answers = torch.stack(mode_answers)
    return images, questions, answer_indices, mode_answers

class ImprovedVQAModel(nn.Module):
    def __init__(self, device, num_answers):
        super().__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.num_answers = num_answers
        print(f"Number of answer classes: {self.num_answers}")

        clip_dim = self.clip_model.config.projection_dim
        self.attention = nn.MultiheadAttention(clip_dim, 8, batch_first=True)
        self.fc1 = nn.Linear(clip_dim * 2, 512)
        self.fc2 = nn.Linear(512, self.num_answers)
        self.dropout = nn.Dropout(0.3)

    def forward(self, image, question):
        image = torch.clamp(image, 0, 1)
        
        inputs = self.processor(text=question, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77, do_rescale=False)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            clip_outputs = self.clip_model(**inputs)
        
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds

        # Apply attention mechanism
        attn_output, _ = self.attention(image_features, text_features, text_features)
        
        # Concatenate attention output with image features
        combined = torch.cat((attn_output, image_features), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt)**gamma * ce_loss
    return focal_loss.mean()

def VQA_score(pred, answers):
    if pred in answers:
        return min(answers.count(pred) / 3, 1)
    else:
        return 0

def train(model, dataloader, optimizer, scheduler, device, idx2answer):
    model.train()
    total_loss = 0
    total_vqa_score = 0
    total_acc = 0

    start = time.time()
    for batch_idx, (image, question, answers, mode_answer) in enumerate(tqdm(dataloader)):
        image = image.to(device).float()
        answers = answers.to(device)
        mode_answer = mode_answer.to(device)

        pred = model(image, question)
        loss = focal_loss(pred, mode_answer)

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        pred_answers = pred.argmax(1).cpu().numpy()
        batch_vqa_score = 0
        for pred_ans, ans_list in zip(pred_answers, answers.cpu().numpy()):
            pred_ans_str = idx2answer[pred_ans]
            ans_list_str = [idx2answer[a] for a in ans_list if a != -1]
            batch_vqa_score += VQA_score(pred_ans_str, ans_list_str)
        total_vqa_score += batch_vqa_score / len(pred_answers)
        
        total_acc += (pred.argmax(1) == mode_answer).float().mean().item()

        if batch_idx == 0:
            print(f"Batch 0 - pred shape: {pred.shape}, mode_answer shape: {mode_answer.shape}")
            print(f"Batch 0 - pred sample: {pred[0][:10]}")
            print(f"Batch 0 - mode_answer sample: {mode_answer[0]}")
            print(f"Batch 0 - Loss: {loss.item()}")

    num_batches = len(dataloader)
    scheduler.step()  # エポックごとにスケジューラを更新
    return (total_loss / num_batches, 
            total_vqa_score / num_batches, 
            total_acc / num_batches, 
            time.time() - start)

def eval(model, dataloader, device):
    model.eval()
    total_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, question, _, _ in tqdm(dataloader):
            image = image.to(device).float()

            pred = model(image, question)
            
            total_acc += pred.argmax(1).float().mean().item()

    num_batches = len(dataloader)
    return (total_acc / num_batches, 
            time.time() - start)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_data_path = './data/train.json'
    custom_mapping = create_custom_mapping(train_data_path)
    if custom_mapping is None:
        print("Failed to create custom mapping. Exiting.")
        return

    df = pd.DataFrame(list(custom_mapping.items()), columns=['answer', 'class_id'])
    df.to_csv('./data/custom_class_mapping.csv', index=False)
    
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]), answer=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = ImprovedVQAModel(device, len(train_dataset.answer2idx)).to(device)

    print(f"Number of answers in dataset: {len(train_dataset.answer2idx)}")
    print(f"Sample answers: {list(train_dataset.answer2idx.items())[:10]}")

    sample_image, sample_question, sample_answers, sample_mode_answer = next(iter(train_loader))
    sample_image = sample_image.to(device).float()
    sample_output = model(sample_image, sample_question)
    print(f"Sample output shape: {sample_output.shape}")
    print(f"Sample output: {sample_output[0][:10]}")

    num_epoch = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)

    for epoch in range(num_epoch):
        train_loss, train_vqa_score, train_acc, train_time = train(model, train_loader, optimizer, scheduler, device, train_dataset.idx2answer)
        eval_acc, eval_time = eval(model, test_loader, device)
        
        print(f"Epoch {epoch + 1}/{num_epoch}")
        print(f"Train - Loss: {train_loss:.4f}, VQA Score: {train_vqa_score:.4f}, Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
        print(f"Eval  - Acc: {eval_acc:.4f}, Time: {eval_time:.2f}s")
        print("-" * 50)

    model.eval()
    submission = []
    with torch.no_grad():
        for image, question, _, _ in tqdm(test_loader, desc="Generating predictions"):
            image = image.to(device).float()
            pred = model(image, question)
            pred = pred.argmax(1).cpu().numpy()
            submission.extend([
                train_dataset.idx2answer[id] for id in pred
            ])

    submission = np.array(submission)
    
    # モデルの保存
    torch.save(model.state_dict(), "improved_model.pth")
    print("Model saved as 'improved_model.pth'")
    
    # 予測結果の保存
    np.save("improved_submission.npy", submission)
    print("Predictions saved as 'improved_submission.npy'")

if __name__ == "__main__":
    main()