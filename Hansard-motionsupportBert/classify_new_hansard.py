import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

print("Loading model...")
MODEL_PATH = "oppose-support/oppose_support"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

print("Loading Hansard data...")
df = pd.read_csv("dataset/hansard_2020_2025_speakers.csv")

# Combine topic (motion) with speech
df["text"] = df["topic"].fillna("") + " [SEP] " + df["speech_text"].fillna("")

# Filter out empty texts
df = df[df["text"].str.len() > 10].reset_index(drop=True)
print(f"Classifying {len(df)} speeches...")

predictions = []
probabilities = []

batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    
    encoding = tokenizer(
        batch_texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        predictions.extend(preds.cpu().numpy().tolist())
        probabilities.extend(probs[:, 1].cpu().numpy().tolist())

df["prediction"] = predictions
df["support_prob"] = probabilities
df["oppose_prob"] = [1 - p for p in probabilities]

# 1 = support, 0 = oppose (based on the training code)
df["label"] = df["prediction"].map({0: "oppose", 1: "support"})

df.to_csv("hansard_2020_2025_classified.csv", index=False)
print(f"Saved to hansard_2020_2025_classified.csv")

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample predictions:")
print(df[["speaker_name", "topic", "label", "support_prob"]].head(20).to_string())
