#!/usr/bin/env python3
"""
finetune_SBERT_kmeans_triplet.py

Pipeline:
1) Load dataframe (parquet)
2) Build feature matrix for clustering using all columns EXCEPT those bạn muốn exclude (ex: 'anon') and except text columns (text_cols)
   - Categorical columns will be one-hot encoded (pd.get_dummies)
   - Numeric columns will be scaled
3) Try KMeans for k in [min_k, max_k] and pick k có silhouette_score cao nhất
4) Gán cụm (cluster labels)
5) Tạo triplet: for each anchor row, sample a positive from same cluster and negative from different cluster.
   - Texts are generated **only** using template4 as bạn yêu cầu
6) Fine-tune SentenceTransformer with TripletLoss
7) Evaluate average cosine similarity on pos/neg pairs before & after
"""

import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import math

# ---------------------------
# Template 4 (only this one)
# ---------------------------
template4 = lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}."

# ---------------------------
# Utils
# ---------------------------
def build_feature_matrix(df, exclude_cols=None, text_cols=None):
    """
    Build numeric matrix for clustering:
    - drop exclude_cols and text_cols
    - one-hot encode remaining non-numeric columns
    - scale numeric features
    """
    exclude_cols = exclude_cols or []
    text_cols = text_cols or []
    cols_to_drop = set(exclude_cols) | set(text_cols)
    available_cols = [c for c in df.columns if c not in cols_to_drop]

    if len(available_cols) == 0:
        raise ValueError("No columns left for clustering. Check exclude_cols and text_cols arguments.")

    features_df = df[available_cols].copy()
    # Fill NaNs
    features_df = features_df.fillna("###_NA_###")

    # For any non-numeric columns, get dummies
    non_numeric = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        features_df = pd.get_dummies(features_df, columns=non_numeric, dummy_na=False)

    # Now ensure all numeric
    numeric_matrix = features_df.astype(float).values
    scaler = StandardScaler()
    numeric_matrix = scaler.fit_transform(numeric_matrix)
    return numeric_matrix

def choose_best_k(X, min_k, max_k, random_state=42):
    """
    Try KMeans for k in [min_k, max_k] and select best by silhouette score.
    Returns chosen_k, dict_of_scores
    """
    n_samples = X.shape[0]
    scores = {}
    best_k = None
    best_score = -1.0

    # enforce sensible bounds
    max_k = min(max_k, n_samples - 1) if n_samples > 1 else 1
    min_k = max(2, min_k) if n_samples > 1 else 1

    for k in range(min_k, max_k + 1):
        try:
            print(f"[kmeans] Running k={k}")
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init='auto', batch_size=2048, max_iter=100)
            labels = km.fit_predict(X)
            # silhouette requires at least 2 clusters and < n_samples clusters
            if len(set(labels)) < 2 or len(set(labels)) >= n_samples:
                continue
            s = silhouette_score(X, labels)
            scores[k] = s
            if s > best_score:
                best_score = s
                best_k = k
        except Exception as e:
            # skip problematic k
            print(f"[kmeans] skip k={k} due to error: {e}")
            continue

    if best_k is None:
        # fallback to k=2 if cannot compute silhouette
        best_k = 2 if n_samples > 2 else 1
    return best_k, scores

def build_triplets(df, cluster_labels, n_triplets_per_anchor=1, seed=42):
    """
    Build InputExample triplets (anchor, positive, negative)
    For each anchor i:
      - positive: random other row j in same cluster
      - negative: random row k in different cluster
    """
    random.seed(seed)
    np.random.seed(seed)

    df = df.reset_index(drop=True).copy()
    df['__cluster__'] = cluster_labels
    clusters = df['__cluster__'].unique().tolist()
    cluster_to_indices = {c: df.index[df['__cluster__']==c].tolist() for c in clusters}

    triplets = []
    n = len(df)
    for idx in range(n):
        anchor_row = df.iloc[idx]
        c = anchor_row['__cluster__']
        same_idxs = cluster_to_indices[c].copy()
        if len(same_idxs) <= 1:
            # cannot form positive in same cluster, skip anchor
            continue
        # remove anchor itself
        same_idxs = [i for i in same_idxs if i != idx]
        diff_clusters = [cc for cc in clusters if cc != c]
        if len(diff_clusters) == 0:
            continue
        for _ in range(n_triplets_per_anchor):
            pos_idx = random.choice(same_idxs)
            # pick negative from a random different cluster
            neg_cluster = random.choice(diff_clusters)
            neg_idx = random.choice(cluster_to_indices[neg_cluster])
            anchor_text = template4(anchor_row)
            pos_text = template4(df.iloc[pos_idx])
            neg_text = template4(df.iloc[neg_idx])
            triplets.append(InputExample(texts=[anchor_text, pos_text, neg_text]))
    return triplets

def average_cosine_for_pairs(model, pairs):
    if len(pairs) == 0:
        return None, None, []
    textsA = [p[0] for p in pairs]
    textsB = [p[1] for p in pairs]
    embA = model.encode(textsA, convert_to_numpy=True, show_progress_bar=False)
    embB = model.encode(textsB, convert_to_numpy=True, show_progress_bar=False)
    sims = []
    for a, b in zip(embA, embB):
        sims.append(float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]))
    return float(np.mean(sims)), float(np.std(sims)), sims

def build_eval_pairs_from_df(df, templates=(template4,), n_samples=500, seed=123):
    """
    Build pos/neg eval pairs:
    - positive: template4(row_i) with template4(row_i) OR template4(row_i) with template4(same-row-other-template) 
      but since we use only template4, pos = two different rows from same original row isn't possible.
      We'll do pos = (template4(row_i), template4(another row in same cluster if exists))
    - negative: pair with row from different cluster
    """
    # This function will be used after cluster labels assigned; to be called when df has '__cluster__'
    random.seed(seed)
    df_shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = min(n_samples, len(df_shuf))
    pos_pairs = []
    neg_pairs = []
    # Build index by cluster
    if '__cluster__' not in df_shuf.columns:
        # fallback to simple shift mismatch
        for i in range(n):
            row_i = df_shuf.iloc[i]
            j = (i + 1) % len(df_shuf)
            row_j = df_shuf.iloc[j]
            pos_pairs.append((template4(row_i), template4(row_i)))  # identical
            neg_pairs.append((template4(row_i), template4(row_j)))
        return pos_pairs, neg_pairs

    cluster_to_indices = {}
    for i, r in df_shuf.iterrows():
        c = r['__cluster__']
        cluster_to_indices.setdefault(c, []).append(i)

    for i in range(n):
        row_i = df_shuf.iloc[i]
        c = row_i['__cluster__']
        same_indices = cluster_to_indices.get(c, [])
        # positive: try to pick another row from same cluster, else pair with itself
        pos = None
        if len(same_indices) > 1:
            # pick a different index from same_indices
            other = random.choice([x for x in same_indices if x != i])
            pos = (template4(row_i), template4(df_shuf.iloc[other]))
        else:
            pos = (template4(row_i), template4(row_i))
        pos_pairs.append(pos)
        # negative: choose row from different cluster
        other_clusters = [cc for cc in cluster_to_indices.keys() if cc != c]
        if len(other_clusters) == 0:
            # fallback: shift
            j = (i + 1) % len(df_shuf)
            neg_pairs.append((template4(row_i), template4(df_shuf.iloc[j])))
        else:
            neg_cluster = random.choice(other_clusters)
            neg_idx = random.choice(cluster_to_indices[neg_cluster])
            neg_pairs.append((template4(row_i), template4(df_shuf.iloc[neg_idx])))
    return pos_pairs, neg_pairs

# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    df = pd.read_parquet(args.input)
    # ensure text columns exist
    for c in args.text_cols.split(','):
        if c not in df.columns:
            raise ValueError(f"Text column '{c}' not found in input dataframe.")

    print(f"[data] loaded {len(df)} rows, columns: {list(df.columns)}")
    # Drop rows with missing text fields used for template generation
    df = df.dropna(subset=args.text_cols.split(',')).reset_index(drop=True)
    print(f"[data] after dropna on text_cols => {len(df)} rows")

    # Build feature matrix for clustering
    exclude_cols = [c.strip() for c in args.exclude_cols.split(',')] if args.exclude_cols else []
    text_cols = [c.strip() for c in args.text_cols.split(',')] if args.text_cols else []
    print(f"[data] exclude columns: {exclude_cols}")
    print(f"[data] text columns: {text_cols}")
    
    X = build_feature_matrix(df, exclude_cols=exclude_cols, text_cols=text_cols)
    print(f"[kmeans] feature matrix shape: {X.shape}")

    # 2. Giảm chiều
    pca = PCA(n_components=0.9)
    X = pca.fit_transform(X)
    print(f"[kmeans] feature matrix shape after PCA: {X.shape}")

    # choose best k
    best_k, scores = choose_best_k(X, args.min_k, args.max_k, random_state=args.seed)
    print(f"[kmeans] tried silhouette scores: {scores}")
    print(f"[kmeans] selected k = {best_k}")

    # final KMeans with best_k
    if best_k <= 1:
        # trivial: all same cluster
        cluster_labels = np.zeros(len(df), dtype=int)
    else:
        km = MiniBatchKMeans(n_clusters=best_k, random_state=args.seed, n_init='auto', batch_size=2048, max_iter=100)
        cluster_labels = km.fit_predict(X)

    df['__cluster__'] = cluster_labels
    # optional: show counts per cluster
    cluster_counts = df['__cluster__'].value_counts().to_dict()
    print(f"[kmeans] cluster sizes: {cluster_counts}")

    # --------------------------
    # Save clustering logs
    # --------------------------
    cluster_log_path = Path(args.output) / "cluster_log.csv"
    cluster_summary_path = Path(args.output) / "cluster_summary.txt"

    # Lưu toàn bộ dataframe có cột __cluster__
    df_out = df.copy()
    df_out.to_csv(cluster_log_path, index=False, encoding='utf-8-sig')

    # Ghi thông tin tổng quan
    with open(cluster_summary_path, "w", encoding="utf-8") as f:
        f.write(f"Best K = {best_k}\n")
        f.write(f"Silhouette scores: {scores}\n")
        f.write(f"Cluster sizes: {cluster_counts}\n")

    print(f"[log] Saved clustering results to:")
    print(f"       - {cluster_log_path}")
    print(f"       - {cluster_summary_path}")

    # split train/val on rows (not on clusters)
    train_df, val_df = train_test_split(df, test_size=args.val_ratio, random_state=args.seed)
    print(f"[data] train={len(train_df)}, val={len(val_df)}")

    # Build triplets from train_df
    triplets = build_triplets(train_df, train_df['__cluster__'].values, n_triplets_per_anchor=args.n_triplets_per_anchor, seed=args.seed)
    if len(triplets) == 0:
        raise RuntimeError("No triplets created. Check your clusters (maybe clusters are size 1).")
    print(f"[data] created {len(triplets)} triplets for training")

    # Prepare DataLoader
    train_dataloader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using {device}")
    model = SentenceTransformer(args.model_name_or_path, device=device)

    # Prepare evaluation pairs using val_df (val_df must have '__cluster__')
    pos_eval_pairs, neg_eval_pairs = build_eval_pairs_from_df(val_df, n_samples=args.eval_samples, seed=args.seed)
    print(f"[eval] pos_pairs={len(pos_eval_pairs)}, neg_pairs={len(neg_eval_pairs)}")

    # Evaluate before
    print("[eval] computing baseline similarities (before fine-tune)...")
    pos_mean_before, pos_std_before, _ = average_cosine_for_pairs(model, pos_eval_pairs)
    neg_mean_before, neg_std_before, _ = average_cosine_for_pairs(model, neg_eval_pairs)
    print(f"  POS before: mean={pos_mean_before:.4f} std={pos_std_before:.4f}")
    print(f"  NEG before: mean={neg_mean_before:.4f} std={neg_std_before:.4f}")

    # Loss: TripletLoss
    train_loss = losses.TripletLoss(model=model, triplet_margin=args.margin)
    epochs = args.epochs
    total_steps = math.ceil(len(train_dataloader) * epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[train] epochs={epochs}, steps~={total_steps}, warmup={warmup_steps}, margin={args.margin}")

    # Fit
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.lr},
        output_path=args.output,
        show_progress_bar=True
    )

    # Load finetuned model and evaluate
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
    parser = argparse.ArgumentParser(description="Fine-tune SBERT using KMeans-based triplets (template4 only).")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file (pandas-readable)")
    parser.add_argument("--output", type=str, default="./models/finetuned_all-MiniLM-L6-v2_kmeans_triplet", help="Output folder to save model")
    parser.add_argument("--model_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Pretrained SBERT model")
    parser.add_argument("--exclude_cols", type=str, default="anon", help="Comma-separated columns to exclude from clustering (default: 'anon')")
    parser.add_argument("--text_cols", type=str, default="usr,jnam,jobenv_req", help="Comma-separated text columns used by template4")
    parser.add_argument("--min_k", type=int, default=2, help="Minimum K for KMeans search")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum K for KMeans search")
    parser.add_argument("--n_triplets_per_anchor", type=int, default=1, help="How many triplets to sample per anchor")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
