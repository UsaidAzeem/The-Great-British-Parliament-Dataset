import os
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_PATH = "oppose_support"
SPEECHES_FILE = "../dataset/hansard_2020_2025_speakers.csv"
OUTPUT_FILE = "model_evaluation.csv"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

print("Loading speeches...")
speeches_df = pd.read_csv(SPEECHES_FILE, usecols=['date', 'topic', 'major_topic', 'speech_text'])

print("Loading votes...")
votes_df = pd.read_csv("../dataset/division_votes.csv")
votes_df = votes_df.drop_duplicates(subset=['gid', 'person_id', 'vote'])
print(f"Unique (gid, person, vote): {len(votes_df)}")

date_topic = speeches_df[speeches_df['major_topic'].notna()].groupby('date')['major_topic'].apply(lambda x: x.value_counts().index[0]).to_dict()
date_speech = speeches_df[speeches_df['speech_text'].notna()].groupby('date')['speech_text'].first().to_dict()

def get_text(row):
    date = row['date']
    topic = date_topic.get(date, '')
    speech = date_speech.get(date, '')
    
    if pd.isna(speech) or not speech:
        return None
    
    text = str(topic) + " [SEP] " + str(speech)
    return text

votes_df['text'] = votes_df.apply(get_text, axis=1)
votes_df = votes_df[votes_df['text'].notna()].reset_index(drop=True)
print(f"Speeches with text: {len(votes_df)}")

votes_df['true_label'] = votes_df['vote'].map({'aye': 1, 'no': 0})
votes_df = votes_df[votes_df['true_label'].notna()].reset_index(drop=True)
print(f"Valid votes: {len(votes_df)}")

batch_size = 64
predictions = []
probabilities = []

print("Classifying...")
for i in tqdm(range(0, len(votes_df), batch_size)):
    batch_texts = votes_df['text'].iloc[i:i+batch_size].tolist()
    
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

votes_df['prediction'] = predictions
votes_df['support_prob'] = probabilities
votes_df['correct'] = (votes_df['prediction'] == votes_df['true_label']).astype(int)
votes_df['predicted_label'] = votes_df['prediction'].map({0: 'oppose', 1: 'support'})
votes_df['actual_label'] = votes_df['true_label'].map({0: 'oppose', 1: 'support'})

accuracy = votes_df['correct'].mean()
print(f"\nAccuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
confusion = pd.crosstab(votes_df['actual_label'], votes_df['predicted_label'], rownames=['Actual'], colnames=['Predicted'])
print(confusion)

votes_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")

print("\nSample predictions:")
print(votes_df[['mp_name', 'vote', 'predicted_label', 'actual_label', 'correct', 'support_prob']].head(20).to_string())
