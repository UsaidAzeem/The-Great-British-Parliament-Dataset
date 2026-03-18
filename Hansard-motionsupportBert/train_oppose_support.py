import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("Loading HanDeSeT data...")
df = pd.read_csv("HanDeSeT.csv")
print(f"Total rows: {len(df)}")

def combine_speech(row):
    parts = []
    for col in ['utt1', 'utt2', 'utt3', 'utt4', 'utt5']:
        if pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]))
    return " ".join(parts)

df["speech"] = df.apply(combine_speech, axis=1)
df["text"] = df["motion"].astype(str) + " [SEP] " + df["speech"].astype(str)
df = df[df["speech"].str.len() > 20].copy()
df = df[df["manual speech"].isin([0, 1])].copy()

print(f"After filtering: {len(df)}")
print(f"Label distribution: {df['manual speech'].value_counts().to_dict()}")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["manual speech"])
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

class ParliamentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("\nLoading model...")
MODEL_NAME = "sisyphus199/ukparliamentBERT"
TOKENIZER_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

MAX_LEN = 512
BATCH_SIZE = 4

train_dataset = ParliamentDataset(
    train_df["text"].tolist(),
    train_df["manual speech"].tolist(),
    tokenizer,
    max_len=MAX_LEN
)

val_dataset = ParliamentDataset(
    val_df["text"].tolist(),
    val_df["manual speech"].tolist(),
    tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
use_amp = device.type == "cuda"

model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

print(f"\nTraining config:")
print(f"  - Max len: {MAX_LEN}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Training samples: {len(train_df)}")
print(f"  - Model: sisyphus199/ukparliamentBERT (tokenizer: bert-base-cased)")
print(f"  - Task: Motion context + speech -> oppose (0) / support (1)")

print("\nTraining...")
best_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss_val = loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss_val
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.save_pretrained("oppose_support")
        tokenizer.save_pretrained("oppose_support")
        print(f"  -> Best model saved!")

print(f"\nBest validation accuracy: {best_accuracy:.4f}")
print("Model saved to oppose_support/")
