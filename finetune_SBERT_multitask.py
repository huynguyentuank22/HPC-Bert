#!/usr/bin/env python3
"""
Finetune all-MiniLM-L6-v2 on multiple tasks using TripletLoss
Save triplets (.pt), full training logs (per step), cosine before/after, and model weights per task.
"""

import os
import random
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch
from datetime import datetime


# =========================================================
# UTILITIES
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text_from_template(df):
    """Áp dụng template để sinh câu mô tả job"""
    # template4 = lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}."
    template4 = lambda r: f"Job {r['jnam']} submitted by user {r['usr']} required {r['cnumr']} cores, allocated {r['cnumat']} cores, utilized {r['cnumut']} cores with {r['nnumr']} nodes requested, {r['nnuma']} NUMA nodes allocated, {r['nnumu']} nodes used, priority level {r['pri']}, requested frequency {r['freq_req']} MHz, allocated frequency {r['freq_alloc']} MHz, memory size limit {r['mszl']:.2f} GB, allocated memory {r['msza']} bytes, maximum memory used {r['mmszu']:.2f} GB, achieving {r['flops']:.2f} FLOPS, {r['mbwidth']:.2f} MB/s bandwidth, operational intensity {r['opint']:.2f}, CPU utilization {r['uctmut']:.2f}%, system CPU time {r['sctmut']:.2f}%, user+system CPU time {r['usctmut']:.2f}%, average power consumption {r['avgpcon']:.2f} watts, minimum power {r['minpcon']:.2f} watts, maximum power {r['maxpcon']:.2f} watts, energy consumption {r['econ']:.2f} units, average idle time {r['idle_time_ave']:.2f} seconds, performance metrics perf1={r['perf1']:.2f}, perf2={r['perf2']:.2f}, perf3={r['perf3']:.2f}, perf4={r['perf4']:.2f}, perf5={r['perf5']:.2f}, perf6={r['perf6']:.2f}, elapsed time {r['elpl']:.2f} seconds, duration {r['duration']:.2f} seconds, classified as {r['pclass']} workload, executed in {r['jobenv_req']} environment, scheduled on {r['schedsdt']}, started at {r['sdt']}, ended at {r['edt']}, with exit code {r['ec']} and final state {r['exit state']}."
    df["text"] = df.apply(template4, axis=1)
    return df


def build_triplets(df, task_name, target_func, task_type):
    triplets = []
    df = df.copy()
    df["target"] = df.apply(target_func, axis=1)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if task_type == "classification":
        class_0_texts = df.loc[df["target"] == 0, "text"].values
        class_1_texts = df.loc[df["target"] == 1, "text"].values

        targets = df["target"].values
        texts = df["text"].values
        n = len(texts)

        # Sinh index random trước
        pos_idx_0 = np.random.randint(0, len(class_0_texts), n)
        pos_idx_1 = np.random.randint(0, len(class_1_texts), n)
        neg_idx_0 = np.random.randint(0, len(class_0_texts), n)
        neg_idx_1 = np.random.randint(0, len(class_1_texts), n)

        triplets = [
            InputExample(texts=[
                texts[i],
                class_0_texts[pos_idx_0[i]] if targets[i] == 0 else class_1_texts[pos_idx_1[i]],
                class_1_texts[neg_idx_1[i]] if targets[i] == 0 else class_0_texts[neg_idx_0[i]]
            ])
            for i in range(n)
        ]

    elif task_type == "regression":
        df = df.sort_values(by="target").reset_index(drop=True)
        n = len(df)
        half = n // 2
        texts = df["text"].values
        triplets = [
            InputExample(texts=[texts[i], texts[i + 1], texts[(i + half) % n]])
            for i in range(n - 1)
        ]

    print(f"[{task_name}] Generated {len(triplets)} triplets.")
    return triplets


def evaluate_cosine(model, triplets, n_samples=300):
    """Đánh giá cosine giữa anchor-pos và anchor-neg"""
    if len(triplets) == 0:
        return None, None
    n = min(n_samples, len(triplets))
    sample_triplets = random.sample(triplets, n)
    texts_all = []
    for t in sample_triplets:
        texts_all.extend([t.texts[0], t.texts[1], t.texts[2]])

    embeddings = model.encode(texts_all, convert_to_tensor=True, batch_size=64)
    anchor_emb, pos_emb, neg_emb = embeddings[0::3], embeddings[1::3], embeddings[2::3]

    cos_pos = util.cos_sim(anchor_emb, pos_emb).diagonal().cpu().numpy()
    cos_neg = util.cos_sim(anchor_emb, neg_emb).diagonal().cpu().numpy()

    return float(np.mean(cos_pos)), float(np.mean(cos_neg))


# =========================================================
# TASK DEFINITIONS
# =========================================================

TASKS = {
    "ec": {
        "type": "classification",
        "target": lambda j: 1 if j["exit state"] == "completed" else 0
    },
    "pclass": {
        "type": "classification",
        "target": lambda j: 1 if j["pclass"] == "compute-bound" else 0
    },
    "avgpcon": {
        "type": "regression",
        "target": lambda j: int(j["avgpcon"] / j["nnuma"])
    },
    "duration": {
        "type": "regression",
        "target": lambda j: int(j["duration"] / 60)
    }
}


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def finetune_task(df, args, task_name, task_info):
    print(f"\n=== Fine-tuning for task: {task_name} ({task_info['type']}) ===")

    # Task folder
    task_dir = os.path.join(args.output, f"{task_name}_triplet")
    os.makedirs(task_dir, exist_ok=True)

    # Triplet generation
    triplets = build_triplets(df, task_name, task_info["target"], task_info["type"])

    # Save triplets (CSV + Torch .pt)
    triplet_texts = pd.DataFrame([
        {"anchor": t.texts[0], "positive": t.texts[1], "negative": t.texts[2]}
        for t in triplets
    ])
    triplet_texts.to_csv(os.path.join(task_dir, "triplet_samples.csv"), index=False)
    torch.save(triplets, os.path.join(task_dir, "triplets.pt"))
    print(f"💾 Saved triplets for {task_name} to {task_dir}")

    # Init model
    model = SentenceTransformer(args.model_name_or_path, device="cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate cosine before training
    cos_pos_before, cos_neg_before = evaluate_cosine(model, triplets, args.eval_samples)
    print(f"🔍 Before finetune | Cos(Pos): {cos_pos_before:.4f}, Cos(Neg): {cos_neg_before:.4f}")

    # DataLoader
    train_dataloader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin
    )

    # =========================================================
    # 🧠 Train with SentenceTransformer.fit()
    # =========================================================
    print(f"🚀 Training {task_name} for {args.epochs} epochs using SentenceTransformer.fit()...")

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.lr},
        output_path=os.path.join(task_dir, "model"),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True  # automatic mixed precision for faster training on GPU
    )

    # =========================================================
    # Evaluate after training
    # =========================================================
    cos_pos_after, cos_neg_after = evaluate_cosine(model, triplets, args.eval_samples)
    print(f"✅ After finetune | Cos(Pos): {cos_pos_after:.4f}, Cos(Neg): {cos_neg_after:.4f}")

    # Save training log
    training_log = {
        "task": task_name,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "params": vars(args),
        "cos_before": {"pos": cos_pos_before, "neg": cos_neg_before},
        "cos_after": {"pos": cos_pos_after, "neg": cos_neg_after},
        "num_triplets": len(triplets)
    }

    with open(os.path.join(task_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved model and log to {task_dir}")

# =========================================================
# ARGPARSE
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune all-MiniLM-L6-v2 with TripletLoss for multiple tasks")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, default="./models/finetuned_all-MiniLM-L6-v2_multitask", help="Output folder to save models and logs")
    parser.add_argument("--model_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    os.makedirs(args.output, exist_ok=True)

    print(f"📂 Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    df = build_text_from_template(df)

    for task_name, task_info in TASKS.items():
        finetune_task(df, args, task_name, task_info)

    print("\n🎉 All tasks completed successfully!")
