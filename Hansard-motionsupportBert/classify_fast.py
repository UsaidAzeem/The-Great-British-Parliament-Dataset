import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model...")
MODEL_PATH = "oppose-support/oppose_support"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()

# Enable FP16 for faster inference
# if device.type == "cuda":
#     model = model.half()

print("Loading data...")
df = pd.read_csv("dataset/hansard_2020_2025_speakers.csv")
print(f"Total speeches: {len(df)}")

# Filter empty texts
df["text"] = df["speech_text"].fillna("").astype(str)
df = df[df["text"].str.len() > 20].reset_index(drop=True)
print(f"After filtering empty: {len(df)}")

# Use topic as prefix if available
df["text"] = df["topic"].fillna("").astype(str) + " [SEP] " + df["text"]

batch_size = 64

print("\nClassifying speeches...")
predictions = []
probabilities = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    
    encoding = tokenizer(
        batch_texts,
        truncation=True,
        max_length=256,
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
df["label"] = df["prediction"].map({0: "oppose", 1: "support"})

output_path = "dataset/hansard_2020_2025_speakers_classified.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample predictions:")
print(df[["speaker_name", "topic", "label", "support_prob"]].head(20).to_string())