# Hansard Motion Support Classifier

I built this model to predict whether an MP supports or opposes the motion they're speaking about in UK Parliament debates.

## What it does

Given a parliamentary motion and an MP's speech, it predicts whether that MP is **for** the motion (support) or **against** it (oppose).

**Example:**
- Motion: "That the Bill be now read a second time"
- MP's speech: "This legislation is crucial for our nation's future..."
- Output: **support** (with ~95% confidence)

## How I trained it

**Dataset:** HanDeSeT (German-English Dependency Treebank with UK Parliament speeches)
- ~4,000 speeches with manual labels (support = 1, oppose = 0)

**Model:** sisyphus199/ukparliamentBERT
- Pre-trained on UK Parliament Hansard 2000-2020
- Fine-tuned for 3 epochs

**Config:**
- Max length: 512 tokens
- Batch size: 4
- Learning rate: 2e-5

**Results:** ~85% validation accuracy

## Files

| File | What it does |
|------|--------------|
| `train_oppose_support.py` | Train the model from scratch |
| `classify_oppose_support.py` | Run predictions on HanDeSeT speeches |
| `classify_fast.py` | Fast batch classification on Hansard 2020-2025 data |
| `classify_chunked.py` | Classify speeches with checkpointing (for large datasets) |
| `classify_mp_profiles.py` | Classify MP profile speeches |
| `classify_new_hansard.py` | Classify newer Hansard debates |
| `evaluate_model.py` | Evaluate accuracy against actual division votes |

**Data:**
- `HanDeSeT.csv` - Training/evaluation dataset
- `speeches_classified_oppose_support.csv` - HanDeSeT speeches with predictions
- `mp_oppose_support_profiles.csv` - MP-level support/oppose rates

## Usage

### Train the model
```bash
python train_oppose_support.py
```
This saves the model to `oppose_support/`.

### Classify speeches
```bash
# On HanDeSeT
python classify_oppose_support.py

# On Hansard 2020-2025
DATASET=hansard_2025 python classify_oppose_support.py

# Fast batch mode
python classify_fast.py
```

### Evaluate
```bash
python evaluate_model.py
```

## Output format

The model outputs probabilities:
- `support_prob`: probability of supporting (closer to 1 = likely support)
- `oppose_prob`: probability of opposing (closer to 1 = likely oppose)
- `prediction`: 1 = support, 0 = oppose

MP profiles show each MP's overall tendency to support or oppose motions.

## Requirements

```
transformers
torch
pandas
scikit-learn
tqdm
```

## Why I built this

I wanted to analyze how MPs actually vote on issues, not just their party lines. This lets me see when MPs break from their party or have specific policy preferences.
