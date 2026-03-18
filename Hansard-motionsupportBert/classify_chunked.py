import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import csv

print("Loading model...")
MODEL_PATH = "oppose_support"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

INPUT_FILE = "dataset/hansard_2020_2025_speakers.csv"
OUTPUT_FILE = "dataset/hansard_classified.csv"
PROGRESS_FILE = "classification_progress.json"
BATCH_SIZE = 32
CHUNK_SIZE = 500

def get_topic_text(row):
    parts = []
    if pd.notna(row.get('major_topic')) and str(row.get('major_topic')).strip():
        parts.append(str(row['major_topic']))
    if pd.notna(row.get('minor_topic')) and str(row.get('minor_topic')).strip():
        parts.append(str(row['minor_topic']))
    if pd.notna(row.get('topic')) and str(row.get('topic')).strip():
        parts.append(str(row['topic']))
    return " ".join(parts) if parts else "unknown topic"

def classify_batch(texts):
    encoding = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    predictions = preds.cpu().tolist()
    support_probs = probs[:, 1].cpu().tolist()
    
    del outputs, probs, input_ids, attention_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return predictions, support_probs

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
    start_idx = progress['processed']
    print(f"Resuming from row {start_idx}")
else:
    progress = {'processed': 0}
    start_idx = 0

total_rows = sum(1 for _ in open(INPUT_FILE)) - 1
print(f"Total rows: {total_rows}")

if start_idx >= total_rows:
    print("Already done!")
    exit(0)

if start_idx == 0:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers.extend(['prediction', 'support_prob', 'oppose_prob'])
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

batch_texts = []
batch_rows = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    
    for i, row in enumerate(reader):
        if i < start_idx:
            continue
        
        row_dict = dict(zip(headers, row))
        topic = get_topic_text(row_dict)
        speech = row_dict.get('speech_text', '')
        text = topic + " [SEP] " + str(speech) if pd.notna(speech) else topic
        
        batch_texts.append(text)
        batch_rows.append(row)
        
        if len(batch_texts) >= BATCH_SIZE:
            preds, probs = classify_batch(batch_texts)
            
            with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row, pred, prob in zip(batch_rows, preds, probs):
                    row.extend([pred, prob, 1 - prob])
                    writer.writerow(row)
            
            progress['processed'] = i + 1
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f)
            
            print(f"Processed {i + 1}/{total_rows}")
            
            batch_texts = []
            batch_rows = []

if batch_texts:
    preds, probs = classify_batch(batch_texts)
    
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row, pred, prob in zip(batch_rows, preds, probs):
            row.extend([pred, prob, 1 - prob])
            writer.writerow(row)
    
    progress['processed'] = total_rows
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

os.remove(PROGRESS_FILE)
print(f"\nDone! Saved to {OUTPUT_FILE}")
