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
from torch.optim.lr_scheduler import OneCycleLR

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
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
        'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    contractions = {
        "n't": " not", "'s": " is", "'re": " are", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_custom_mapping(train_data_path, min_freq=5, max_classes=5000):
    try:
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"The file {train_data_path} does not exist.")
        
        with open(train_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_answers = []
        for img_id, answers in data['answers'].items():
            if isinstance(answers, list):
                all_answers.extend([process_text(ans['answer']) for ans in answers if isinstance(ans, dict) and 'answer' in ans])
        
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
        class_mapping['[OTHER]'] = idx + 1
        
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

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image_path = f"{self.image_dir}/{self.data['image'][img_id]}"
        question = self.data['question'][img_id]
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image) if self.transform else image

        if self.answer and 'answers' in self.data:
            answers = self.data['answers'].get(img_id, [])
            answer_indices = []
            for ans in answers:
                processed_ans = process_text(ans['answer'])
                if processed_ans.isdigit():
                    answer_indices.append(self.answer2idx.get('[NUMBER]', 0))
                elif processed_ans in self.answer2idx:
                    answer_indices.append(self.answer2idx[processed_ans])
                else:
                    answer_indices.append(self.answer2idx.get('[OTHER]', 0))
            
            answer_indices = answer_indices[:self.max_answers]
            answer_indices += [0] * (self.max_answers - len(answer_indices))
            
            mode_answer_idx = max(set(answer_indices), key=answer_indices.count) if answer_indices else 0
            return image, question, torch.tensor(answer_indices), torch.tensor(mode_answer_idx)
        else:
            return image, question, torch.tensor([0] * self.max_answers), torch.tensor(0)

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    images, questions, answer_indices, mode_answers = zip(*batch)
    images = torch.stack(images)
    questions = list(questions)
    answer_indices = torch.stack(answer_indices)
    mode_answers = torch.stack(mode_answers)
    return images, questions, answer_indices, mode_answers

class EnhancedVQAModel(nn.Module):
    def __init__(self, device, num_answers):
        super().__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.num_answers = num_answers
        clip_dim = self.clip_model.config.projection_dim

        self.cross_attention1 = nn.MultiheadAttention(clip_dim, 16, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(clip_dim, 16, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_answers)
        )

    def forward(self, image, question):
        image = torch.clamp(image, 0, 1)
        
        inputs = self.processor(text=question, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77, do_rescale=False)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            clip_outputs = self.clip_model(**inputs)
        
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds

        attn_output1, _ = self.cross_attention1(image_features, text_features, text_features)
        attn_output2, _ = self.cross_attention2(attn_output1, text_features, text_features)
        
        fused_features = self.fusion(torch.cat([attn_output2, image_features], dim=-1))
        
        logits = self.classifier(fused_features)
        
        return logits

def improved_loss(pred, target, gamma=2.0, alpha=0.25, smooth=0.1, kl_weight=0.1):
    ce_loss = F.cross_entropy(pred, target, reduction='none', label_smoothing=smooth)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt)**gamma * ce_loss
    
    soft_target = F.one_hot(target, num_classes=pred.size(1)).float()
    soft_target = soft_target * (1 - smooth) + smooth / pred.size(1)
    log_pred = F.log_softmax(pred, dim=1)
    kl_div = F.kl_div(log_pred, soft_target, reduction='batchmean')
    
    return focal_loss.mean() + kl_weight * kl_div

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
        loss = improved_loss(pred, mode_answer)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
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
            total_acc / num_batches, 
            time.time() - start)

def eval(model, dataloader, device, idx2answer):
    model.eval()
    total_loss = 0
    total_vqa_score = 0
    total_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, question, answers, mode_answer in tqdm(dataloader):
            image = image.to(device).float()
            answers = answers.to(device)
            mode_answer = mode_answer.to(device)

            pred = model(image, question)
            loss = improved_loss(pred, mode_answer)

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
            total_acc / num_batches,
            time.time() - start)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(),
        transforms.ToTensor,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_data_path = './data/train.json'
    custom_mapping = create_custom_mapping(train_data_path, min_freq=5, max_classes=5000)
    if custom_mapping is None:
        print("Failed to create custom mapping. Exiting.")
        return

    df = pd.DataFrame(list(custom_mapping.items()), columns=['answer', 'class_id'])
    df.to_csv('./data/custom_class_mapping.csv', index=False)
    
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    val_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=test_transform, answer=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = EnhancedVQAModel(device, len(train_dataset.answer2idx)).to(device)

    num_epochs = 100  # Increased number of epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Using OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    best_vqa_score = 0
    for epoch in range(num_epochs):
        train_loss, train_vqa_score, train_acc, train_time = train(model, train_loader, optimizer, scheduler, device, train_dataset.idx2answer)
        val_loss, val_vqa_score, val_acc, val_time = eval(model, val_loader, device, train_dataset.idx2answer)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, VQA Score: {train_vqa_score:.4f}, Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
        print(f"Val   - Loss: {val_loss:.4f}, VQA Score: {val_vqa_score:.4f}, Acc: {val_acc:.4f}, Time: {val_time:.2f}s")
        print("-" * 50)

        if val_vqa_score > best_vqa_score:
            best_vqa_score = val_vqa_score
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with VQA Score: {best_vqa_score:.4f}")

        early_stopping(val_vqa_score)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load("best_model.pth"))
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
    
    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved as 'final_model.pth'")
    
    np.save("submission.npy", submission)
    print("Predictions saved as 'submission.npy'")

if __name__ == "__main__":
    main()