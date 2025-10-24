"""
finetune_anonymized_miniLM.py

Fine-tune sentence-transformers/all-MiniLM-L6-v2 using contrastive learning
(MultipleNegativesRankingLoss) on anonymized columns (usr, jnam, jobenv_req).

Usage:
    - Place your CSV with columns: usr,jnam,jobenv_req (no header order assumptions but needs these names).
    - python finetune_anonymized_miniLM.py --input jobs.csv --output ./finetuned_model
"""

import argparse
import random
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ---------------------------
# Templates (your best two)
# ---------------------------
template2 = lambda r: f"The user profile {r['usr']} submitted a computation {r['jnam']} to the environment {r['jobenv_req']}."
template4 = lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}."

# ---------------------------
# Utility functions
# ---------------------------
def make_positive_pairs(df, templates=(template2, template4)):
    examples = []
    for _, row in df.iterrows():
        s1 = templates[0](row)
        s2 = templates[1](row)
        examples.append(InputExample(texts=[s1, s2]))
    return examples

def make_negative_augment(df, n_aug_per_row=0, templates=(template2, template4), seed=42):
    """
    Optional: create 'negative-like' InputExample pairs by swapping env or user.
    Used to increase variability. Not strictly necessary because MultipleNegativesRankingLoss uses
    in-batch negatives, but can help.
    """
    random.seed(seed)
    examples = []
    jobenvs = df['jobenv_req'].tolist()
    usrs = df['usr'].tolist()
    jnams = df['jnam'].tolist()
    for _, row in df.iterrows():
        base = templates[0](row)
        for _ in range(n_aug_per_row):
            fake_env = random.choice(usrs)
            if fake_env == row['usr']:
                continue
            fake = templates[1]({'usr': fake_env, 'jnam': row['jnam'], 'jobenv_req': row['jobenv_req']})
            examples.append(InputExample(texts=[base, fake]))
    return examples

def average_cosine_for_pairs(model, pairs):
    if len(pairs) == 0:
        return None
    textsA = [p[0] for p in pairs]
    textsB = [p[1] for p in pairs]
    embA = model.encode(textsA, convert_to_numpy=True, show_progress_bar=False)
    embB = model.encode(textsB, convert_to_numpy=True, show_progress_bar=False)
    sims = []
    for a, b in zip(embA, embB):
        sims.append(float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]))
    return float(np.mean(sims)), float(np.std(sims)), sims

def build_eval_pairs(df, templates=(template2, template4), n_samples=500, seed=123):
    random.seed(seed)
    df_shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = min(n_samples, len(df_shuf))
    pos_pairs = []
    neg_pairs = []
    # positive pairs: same row -> t2,t4
    for i in range(n):
        row = df_shuf.iloc[i]
        pos_pairs.append((templates[0](row), templates[1](row)))
    # negative pairs: pair t2(row_i) with t4(row_j) where j != i
    for i in range(n):
        j = (i + 1) % len(df_shuf)  # simple shift gives mismatch
        row_i = df_shuf.iloc[i]
        row_j = df_shuf.iloc[j]
        neg_pairs.append((templates[0](row_i), templates[1](row_j)))
    return pos_pairs, neg_pairs

# ---------------------------
# Main training pipeline
# ---------------------------
def main(args):
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data
    df = pd.read_parquet(args.input)
    # ensure columns exist
    for c in ['usr', 'jnam', 'jobenv_req']:
        if c not in df.columns:
            raise ValueError(f"Input CSV must contain column: {c}")

    # optional filtering: drop NaNs
    df = df.dropna(subset=['usr', 'jnam', 'jobenv_req']).reset_index(drop=True)
    print(f"[data] loaded {len(df)} rows")

    # split train/val
    train_df, val_df = train_test_split(df, test_size=args.val_ratio, random_state=args.seed)
    print(f"[data] train={len(train_df)}, val={len(val_df)}")

    # create InputExample list
    train_examples = make_positive_pairs(train_df)
    if args.negative_aug_per_row > 0:
        train_examples += make_negative_augment(train_df, n_aug_per_row=args.negative_aug_per_row)

    # convert to dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using {device}")
    model = SentenceTransformer(args.model_name_or_path, device=device)

    # Prepare evaluator pairs for before/after comparison
    pos_eval_pairs, neg_eval_pairs = build_eval_pairs(val_df, n_samples=args.eval_samples)
    print(f"[eval] positive eval pairs: {len(pos_eval_pairs)}, negative eval pairs: {len(neg_eval_pairs)}")

    # Evaluate before training
    print("[eval] computing baseline similarities (before fine-tune)...")
    pos_mean_before, pos_std_before, _ = average_cosine_for_pairs(model, pos_eval_pairs)
    neg_mean_before, neg_std_before, _ = average_cosine_for_pairs(model, neg_eval_pairs)
    print(f"  POS before: mean={pos_mean_before:.4f} std={pos_std_before:.4f}")
    print(f"  NEG before: mean={neg_mean_before:.4f} std={neg_std_before:.4f}")

    # Loss and training setup
    train_loss = losses.MultipleNegativesRankingLoss(model)
    epochs = args.epochs
    total_steps = math.ceil(len(train_dataloader) * epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[train] epochs={epochs}, steps~={total_steps}, warmup={warmup_steps}")

    # Fit
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.lr},
        output_path=args.output,
        show_progress_bar=True
    )

    # Load fine-tuned model for evaluation
    finetuned = SentenceTransformer(args.output, device=device)
    print("[eval] computing similarities (after fine-tune)...")
    pos_mean_after, pos_std_after, _ = average_cosine_for_pairs(finetuned, pos_eval_pairs)
    neg_mean_after, neg_std_after, _ = average_cosine_for_pairs(finetuned, neg_eval_pairs)
    print(f"  POS after: mean={pos_mean_after:.4f} std={pos_std_after:.4f}")
    print(f"  NEG after: mean={neg_mean_after:.4f} std={neg_std_after:.4f}")

    print("Done. Model saved to:", args.output)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CSV file with columns: usr,jnam,jobenv_req")
    parser.add_argument("--output", type=str, default="./models/finetuned_all-MiniLM-L6-v2", help="output folder for fine-tuned model")
    parser.add_argument("--model_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="pretrained model")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--negative_aug_per_row", type=int, default=0, help="optional negative-like augmentations per row")
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
