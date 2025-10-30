#!/usr/bin/env python3
"""
finetune_SBERT_kmeans_triplet.py (with cluster + triplet caching)

Enhancements:
- If cluster results already exist -> load them
- If triplets already exist -> load them
- Save triplet logs
"""

import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import math
import pickle

# ---------------------------
# Template 4
# ---------------------------
template4 = lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}."

# ---------------------------
# Utils
# ---------------------------
def build_feature_matrix(df, exclude_cols=None, text_cols=None):
    exclude_cols = exclude_cols or []
    text_cols = text_cols or []
    cols_to_drop = set(exclude_cols) | set(text_cols)
    available_cols = [c for c in df.columns if c not in cols_to_drop]
    print(f"[kmeans] Using columns for clustering: {available_cols}")
    if len(available_cols) == 0:
        raise ValueError("No columns left for clustering. Check exclude_cols and text_cols arguments.")

    features_df = df[available_cols].copy().fillna("###_NA_###")
    non_numeric = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        features_df = pd.get_dummies(features_df, columns=non_numeric, dummy_na=False)

    numeric_matrix = features_df.astype(float).values
    scaler = StandardScaler()
    return scaler.fit_transform(numeric_matrix)

def choose_best_k(X, min_k, max_k, random_state=42):
    scores = {}
    best_k = None
    best_score = -1.0
    n_samples = X.shape[0]
    max_k = min(max_k, n_samples - 1) if n_samples > 1 else 1
    min_k = max(2, min_k) if n_samples > 1 else 1

    for k in range(min_k, max_k + 1):
        try:
            print(f"[kmeans] Running k={k}")
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init='auto', batch_size=2048, max_iter=100)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2 or len(set(labels)) >= n_samples:
                continue
            s = silhouette_score(X, labels)
            scores[k] = s
            if s > best_score:
                best_score = s
                best_k = k
        except Exception as e:
            print(f"[kmeans] skip k={k} due to error: {e}")
            continue
    return (2 if best_k is None else best_k), scores

def build_triplets(df, cluster_labels, n_triplets_per_anchor=1, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    df = df.reset_index(drop=True).copy()
    df['__cluster__'] = cluster_labels
    clusters = df['__cluster__'].unique().tolist()
    cluster_to_indices = {c: df.index[df['__cluster__']==c].tolist() for c in clusters}

    triplets = []
    for idx in range(len(df)):
        anchor_row = df.iloc[idx]
        c = anchor_row['__cluster__']
        same_idxs = [i for i in cluster_to_indices[c] if i != idx]
        if len(same_idxs) == 0:
            continue
        diff_clusters = [cc for cc in clusters if cc != c]
        if len(diff_clusters) == 0:
            continue
        for _ in range(n_triplets_per_anchor):
            pos_idx = random.choice(same_idxs)
            neg_cluster = random.choice(diff_clusters)
            neg_idx = random.choice(cluster_to_indices[neg_cluster])
            triplets.append(InputExample(
                texts=[
                    template4(anchor_row),
                    template4(df.iloc[pos_idx]),
                    template4(df.iloc[neg_idx])
                ]
            ))
    return triplets

def average_cosine_for_pairs(model, pairs):
    if len(pairs) == 0:
        return None, None, []
    textsA, textsB = zip(*pairs)
    embA = model.encode(list(textsA), convert_to_numpy=True, show_progress_bar=False)
    embB = model.encode(list(textsB), convert_to_numpy=True, show_progress_bar=False)
    sims = [float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0]) for a,b in zip(embA, embB)]
    return float(np.mean(sims)), float(np.std(sims)), sims

def build_eval_pairs_from_df(df, n_samples=500, seed=123):
    random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = min(n_samples, len(df))
    pos_pairs, neg_pairs = [], []
    cluster_to_indices = {c: df.index[df['__cluster__']==c].tolist() for c in df['__cluster__'].unique()}
    for i in range(n):
        row_i = df.iloc[i]
        c = row_i['__cluster__']
        same_indices = [x for x in cluster_to_indices[c] if x != i]
        if same_indices:
            pos = (template4(row_i), template4(df.iloc[random.choice(same_indices)]))
        else:
            pos = (template4(row_i), template4(row_i))
        pos_pairs.append(pos)
        other_clusters = [cc for cc in cluster_to_indices if cc != c]
        if other_clusters:
            neg_cluster = random.choice(other_clusters)
            neg_idx = random.choice(cluster_to_indices[neg_cluster])
            neg_pairs.append((template4(row_i), template4(df.iloc[neg_idx])))
    return pos_pairs, neg_pairs

# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_parquet(args.input)
    df = df.dropna(subset=args.text_cols.split(',')).reset_index(drop=True)
    print(f"[data] loaded {len(df)} rows")

    output_dir = Path(args.output)
    cluster_log_path = output_dir / "cluster_log.csv"
    triplets_path = output_dir / "triplets.pt"
    triplets_log_path = output_dir / "triplets_log.txt"

    # ==================================================
    # Load or compute clusters
    # ==================================================
    if cluster_log_path.exists():
        print(f"[cache] Found existing cluster file: {cluster_log_path}")
        df = pd.read_csv(cluster_log_path)
        print(f"[cache] Loaded cluster labels with {df['__cluster__'].nunique()} clusters")
    else:
        print("[kmeans] No cluster cache found, computing...")
        exclude_cols = [c.strip() for c in args.exclude_cols.split(',')] if args.exclude_cols else []
        text_cols = [c.strip() for c in args.text_cols.split(',')] if args.text_cols else []
        X = build_feature_matrix(df, exclude_cols=exclude_cols, text_cols=text_cols)
        print(f"[kmeans] feature matrix shape: {X.shape}")
        # pca = PCA(n_components=0.9)
        # X = pca.fit_transform(X)
        best_k, scores = choose_best_k(X, args.min_k, args.max_k, random_state=args.seed)
        print(f"[kmeans] selected k = {best_k} with silhouette scores: {scores}")
        km = MiniBatchKMeans(n_clusters=best_k, random_state=args.seed, n_init='auto', batch_size=2048, max_iter=100)
        cluster_labels = km.fit_predict(X)
        df['__cluster__'] = cluster_labels
        df.to_csv(cluster_log_path, index=False, encoding='utf-8-sig')
        print(f"[log] Saved cluster results to {cluster_log_path}")

    # ==================================================
    # Load or build triplets
    # ==================================================
    if triplets_path.exists():
        print(f"[cache] Found existing triplets: {triplets_path}")
        with open(triplets_path, "rb") as f:
            triplets = pickle.load(f)
        print(f"[cache] Loaded {len(triplets)} triplets")
    else:
        print("[triplet] Building new triplets...")
        train_df, _ = train_test_split(df, test_size=args.val_ratio, random_state=args.seed)
        triplets = build_triplets(train_df, train_df['__cluster__'].values,
                                  n_triplets_per_anchor=args.n_triplets_per_anchor,
                                  seed=args.seed)
        if len(triplets) == 0:
            raise RuntimeError("No triplets created.")
        with open(triplets_path, "wb") as f:
            pickle.dump(triplets, f)
        with open(triplets_log_path, "w", encoding="utf-8") as f:
            f.write(f"Triplets: {len(triplets)}\n")
            f.write(f"Anchors: {len(train_df)}\n")
            f.write(f"Clusters: {df['__cluster__'].nunique()}\n")
        print(f"[triplet] Saved {len(triplets)} triplets to {triplets_path}")

    # ==================================================
    # Fine-tune model
    # ==================================================
    train_dataloader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name_or_path, device=device)
    print(f"[train] using {device}")

    # Eval pairs
    pos_eval, neg_eval = build_eval_pairs_from_df(df, n_samples=args.eval_samples, seed=args.seed)
    pos_before, _, _ = average_cosine_for_pairs(model, pos_eval)
    neg_before, _, _ = average_cosine_for_pairs(model, neg_eval)
    print(f"POS before={pos_before:.4f}, NEG before={neg_before:.4f}")

    train_loss = losses.TripletLoss(model=model, triplet_margin=args.margin)
    total_steps = math.ceil(len(train_dataloader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.lr},
        output_path=args.output,
        show_progress_bar=True
    )

    finetuned = SentenceTransformer(args.output, device=device)
    pos_after, _, _ = average_cosine_for_pairs(finetuned, pos_eval)
    neg_after, _, _ = average_cosine_for_pairs(finetuned, neg_eval)
    print(f"POS after={pos_after:.4f}, NEG after={neg_after:.4f}")
    print("✅ Done. Model saved to:", args.output)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SBERT using KMeans-based triplets (template4 only) with caching.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./models/finetuned_all-MiniLM-L6-v2_kmeans_triplet")
    parser.add_argument("--model_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--exclude_cols", type=str, default="anon")
    parser.add_argument("--text_cols", type=str, default="usr,jnam,jobenv_req")
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--n_triplets_per_anchor", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
