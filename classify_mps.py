import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("uk_left_right_binary")
tokenizer = AutoTokenizer.from_pretrained("uk_left_right_binary")
model = model.to(device)
model.eval()

print("Loading data...")
df = pd.read_csv("HanDeSeT.csv")
print(f"Total speeches: {len(df)}")

text_cols = ["utt1", "utt2", "utt3", "utt4", "utt5"]
df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()

def classify_text(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        left_prob = probs[0].item()
        right_prob = probs[1].item()
        ideology_score = left_prob - right_prob
    
    return left_prob, right_prob, ideology_score

print("\nClassifying speeches...")
results = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    left_pct, right_pct, ideology_score = classify_text(row["text"])
    results.append({
        "mp_name": row["name"],
        "party": row["party affiliation"],
        "text": row["text"][:200],
        "left_pct": left_pct,
        "right_pct": right_pct,
        "ideology_score": ideology_score
    })

results_df = pd.DataFrame(results)
results_df.to_csv("speeches_classified.csv", index=False)

print("\nAggregating by MP...")
mp_profiles = results_df.groupby(["mp_name", "party"]).agg({
    "left_pct": "mean",
    "right_pct": "mean",
    "ideology_score": ["mean", "std"],
    "text": "count"
}).reset_index()

mp_profiles.columns = ["mp_name", "party", "avg_left_pct", "avg_right_pct", "mean_ideology", "std_ideology", "speech_count"]
mp_profiles["std_ideology"] = mp_profiles["std_ideology"].fillna(0)
mp_profiles = mp_profiles.sort_values("mean_ideology", ascending=False)
mp_profiles.to_csv("mp_ideology_profiles.csv", index=False)

print(f"\nTotal MPs: {len(mp_profiles)}")
print("\n=== Top 10 Most Left-Wing ===")
print(mp_profiles.head(10)[["mp_name", "party", "mean_ideology", "std_ideology", "speech_count"]].to_string(index=False))

print("\n=== Top 10 Most Right-Wing ===")
print(mp_profiles.tail(10)[["mp_name", "party", "mean_ideology", "std_ideology", "speech_count"]].to_string(index=False))

print("\n=== Average by Party ===")
party_avg = mp_profiles.groupby("party").agg({
    "mean_ideology": "mean",
    "std_ideology": "mean",
    "speech_count": "sum"
}).reset_index()
party_avg = party_avg.sort_values("mean_ideology")
print(party_avg.to_string(index=False))

print("\nSaved: speeches_classified.csv, mp_ideology_profiles.csv")
