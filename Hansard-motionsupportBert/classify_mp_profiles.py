import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

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
OUTPUT_FILE = "mp_profiles_classified_oppose_support.csv"

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

print("Loading mp_profiles.csv...")
df = pd.read_csv("dataset/mp_profiles.csv")
print(f"Dataset has {len(df)} speeches")

df["text"] = df["topic"].fillna("") + " [SEP] " + df["speech_excerpt"].fillna("")

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
        if current_count % 500 == 0:
            print(f"  Processed {current_count}/{len(df)}")
        if current_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(current_count, predictions, probabilities)
            print(f"  [CHECKPOINT SAVED at {current_count}]")

df["predicted_position"] = predictions[:len(df)]
df["support_prob"] = probabilities[:len(df)]
df["oppose_prob"] = [1 - p for p in probabilities[:len(df)]]

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")

df["position_num"] = df["position"].map({"support": 1, "oppose": 0})
accuracy = (df["predicted_position"] == df["position_num"]).mean()
print(f"\nAccuracy: {accuracy:.2%}")

print("\nSaved: mp_profiles_classified_oppose_support.csv")