import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

print("Loading model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "oppose_support")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

CHECKPOINT_FILE = "classify_checkpoint.json"
CHECKPOINT_INTERVAL = 500
OUTPUT_FILE = "speeches_classified_oppose_support.csv"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {data.get('processed', 0)} speeches already classified")
        return data
    return None

def save_checkpoint(processed_count, predictions, probabilities):
    checkpoint = {
        'processed': processed_count,
        'predictions': predictions,
        'probabilities': probabilities
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def combine_speech_handeset(row):
    parts = []
    for col in ['utt1', 'utt2', 'utt3', 'utt4', 'utt5']:
        if pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]))
    return " ".join(parts)

DATASET = os.environ.get('DATASET', 'hansard_2020_2025')

if DATASET == 'hansard_2025':
    print("Loading hansard_2020_2025_speakers.csv...")
    df = pd.read_csv("dataset/hansard_2020_2025_speakers.csv")
    df["text"] = df["major_topic"].fillna("") + " [SEP] " + df["speech_text"].fillna("")
    print(f"Dataset has {len(df)} speeches")
else:
    print("Loading HanDeSeT data...")
    df = pd.read_csv("HanDeSeT.csv")
    df["speech"] = df.apply(combine_speech_handeset, axis=1)
    df["text"] = df["motion"].astype(str) + " [SEP] " + df["speech"].astype(str)
    print(f"Dataset has {len(df)} speeches")

checkpoint = load_checkpoint()
start_idx = 0
predictions = []
probabilities = []

if checkpoint:
    start_idx = checkpoint['processed']
    predictions = checkpoint['predictions']
    probabilities = checkpoint['probabilities']
    print(f"Starting from index {start_idx}")

print(f"Classifying {len(df)} speeches (starting from {start_idx})...")

df_to_process = df.iloc[start_idx:]

with torch.no_grad():
    for idx, row in df_to_process.iterrows():
        text = row["text"]
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
        predictions.append(pred)
        probabilities.append(probs[0][1].item())
        
        current_count = start_idx + len(predictions)
        if current_count % 200 == 0:
            print(f"  Processed {current_count}/{len(df)}")
        if current_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(current_count, predictions, probabilities)
            print(f"  [CHECKPOINT SAVED at {current_count}]")

df["prediction"] = predictions[:len(df)]
df["support_prob"] = probabilities[:len(df)]
df["oppose_prob"] = [1 - p for p in probabilities[:len(df)]]

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved speeches to {OUTPUT_FILE}")

if DATASET == 'hansard_2025':
    mp_profiles = df.groupby("speaker_name").agg({
        "prediction": ["mean", "count"],
        "support_prob": "mean",
        "oppose_prob": "mean",
        "party": "first"
    }).reset_index()
    mp_profiles.columns = ["name", "support_rate", "total_speeches", "avg_support_prob", "avg_oppose_prob", "party"]
else:
    mp_profiles = df.groupby("name").agg({
        "prediction": ["mean", "count"],
        "support_prob": "mean",
        "oppose_prob": "mean",
        "party affiliation": "first",
        "title": lambda x: list(x.unique())[:5]
    }).reset_index()
    mp_profiles.columns = ["name", "support_rate", "total_speeches", "avg_support_prob", "avg_oppose_prob", "party", "motions"]

mp_profiles = mp_profiles.sort_values("support_rate", ascending=False)
mp_profiles.to_csv("mp_oppose_support_profiles.csv", index=False)
print(f"Saved MP profiles to mp_oppose_support_profiles.csv")

print("\nTop 10 MPs who support motions most often:")
print(mp_profiles.head(10)[["name", "party", "support_rate", "total_speeches"]].to_string())

print("\nTop 10 MPs who oppose motions most often:")
print(mp_profiles.tail(10)[["name", "party", "support_rate", "total_speeches"]].to_string())
