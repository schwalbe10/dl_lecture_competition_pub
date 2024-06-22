import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, max_answers=10):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer
        self.max_answers = max_answers

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.df["question"]:
            question = process_text(question)
            words = question.split()
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer and 'answers' in self.df.columns:
            all_answers = [process_text(ans["answer"]) for answers in self.df["answers"] for ans in answers]
            answer_counts = Counter(all_answers)
            self.answer2idx = {ans: idx for idx, (ans, count) in enumerate(answer_counts.most_common()) if count >= 9}
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

            print(f"Number of unique answers: {len(self.answer2idx)}")
            print(f"Sample answers: {list(self.answer2idx.items())[:10]}")

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image) if self.transform else image
        question = self.df["question"][idx]

        if self.answer and 'answers' in self.df.columns:
            answers = self.df["answers"][idx]
            answer_indices = [self.answer2idx.get(process_text(ans["answer"]), -1) for ans in answers]
            answer_indices = [idx for idx in answer_indices if idx != -1]
            if not answer_indices:
                answer_indices = [0]  # デフォルトの答えのインデックス
            
            # パディング
            answer_indices = answer_indices[:self.max_answers]  # 最大数に切り詰め
            answer_indices += [-1] * (self.max_answers - len(answer_indices))  # -1でパディング
            
            mode_answer_idx = max(set(answer_indices), key=answer_indices.count)
            return image, question, torch.tensor(answer_indices), torch.tensor(mode_answer_idx)
        else:
            return image, question

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    images, questions, answers, mode_answers = zip(*batch)
    images = torch.stack(images)
    answers = torch.stack(answers)
    mode_answers = torch.stack(mode_answers)
    return images, questions, answers, mode_answers

def VQA_score(pred, answers):
    if pred in answers:
        return min(answers.count(pred) / 3, 1)
    else:
        return 0

class VQAModel(nn.Module):
    def __init__(self, device, num_answers):
        super().__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.num_answers = num_answers
        print(f"Number of answer classes: {self.num_answers}")

        self.fc = nn.Linear(self.clip_model.config.projection_dim, self.num_answers).to(device)

    def forward(self, image, question):
        image = torch.clamp(image, 0, 1)
        
        inputs = self.processor(text=question, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            clip_outputs = self.clip_model(**inputs)
        
        image_features = clip_outputs.image_embeds
        logits = self.fc(image_features)
        
        return logits

def train(model, dataloader, optimizer, criterion, device, idx2answer):
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
        loss = criterion(pred, mode_answer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        pred_answers = pred.argmax(1).cpu().numpy()
        batch_vqa_score = 0
        for pred_ans, ans_list in zip(pred_answers, answers.cpu().numpy()):
            pred_ans_str = idx2answer[pred_ans]
            ans_list_str = [idx2answer[a] for a in ans_list if a != -1]  # -1 (パディング) を除外
            batch_vqa_score += VQA_score(pred_ans_str, ans_list_str)
        total_vqa_score += batch_vqa_score / len(pred_answers)
        
        total_acc += (pred.argmax(1) == mode_answer).float().mean().item()

        if batch_idx == 0:
            print(f"Batch 0 - pred shape: {pred.shape}, mode_answer shape: {mode_answer.shape}")
            print(f"Batch 0 - pred sample: {pred[0][:10]}")
            print(f"Batch 0 - mode_answer sample: {mode_answer[0]}")
            print(f"Batch 0 - Loss: {loss.item()}")

    num_batches = len(dataloader)
    return (total_loss / num_batches, 
            total_vqa_score / num_batches, 
            total_acc / num_batches, 
            time.time() - start)

def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, question in tqdm(dataloader):
            image = image.to(device).float()

            pred = model(image, question)
            
            total_acc += pred.argmax(1).float().mean().item()

    num_batches = len(dataloader)
    return (total_acc / num_batches, 
            time.time() - start)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VQAModel(device, len(train_dataset.answer2idx)).to(device)

    print(f"Number of answers in dataset: {len(train_dataset.answer2idx)}")
    print(f"Sample answers: {list(train_dataset.answer2idx.items())[:10]}")

    sample_image, sample_question, sample_answers, sample_mode_answer = next(iter(train_loader))
    sample_image = sample_image.to(device).float()
    sample_output = model(sample_image, sample_question)
    print(f"Sample output shape: {sample_output.shape}")
    print(f"Sample output: {sample_output[0][:10]}")

    num_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epoch):
        train_loss, train_vqa_score, train_acc, train_time = train(model, train_loader, optimizer, criterion, device, train_dataset.idx2answer)
        eval_acc, eval_time = eval(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{num_epoch}")
        print(f"Train - Loss: {train_loss:.4f}, VQA Score: {train_vqa_score:.4f}, Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
        print(f"Eval  - Acc: {eval_acc:.4f}, Time: {eval_time:.2f}s")
        print("-" * 50)

    model.eval()
    submission = []
    with torch.no_grad():
        for image, question in test_loader:
            image = image.to(device).float()
            pred = model(image, question)
            pred = pred.argmax(1).cpu().numpy()
            submission.extend([train_dataset.idx2answer[id] for id in pred])

    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()