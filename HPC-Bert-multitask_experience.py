#!/usr/bin/env python3
"""
HPC-Bert_experience.py

Phiên bản đã chỉnh sửa:
- Với từng task, tự động load model SBERT đã finetune đúng cho task đó
  từ path: models/finetuned_all-MiniLM-L6-v2_multitask/{task}_triplet/model
- Sau đó train 3 model: KNN, RF, XGBoost
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

if __name__ == "__main__":
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # === Paths ===
    result_path = "baseline_results"
    os.makedirs(result_path, exist_ok=True)
    
    data_folder = "data_train"
    emb_folder = "data_embedding_semantic_hpcb-multitask"
    os.makedirs(emb_folder, exist_ok=True)
    
    test_yms = ["24_04"]
    regression_metric = lambda y_true, y_pred: f"MAE: {mean_absolute_error(y_true, y_pred):.4f}"

    # === Tasks ===
    tasks = {
        "ec": {"type": "classification", "target": lambda j: 1 if j["exit state"] == "completed" else 0},
        "pclass": {"type": "classification", "target": lambda j: 1 if j.pclass == "compute-bound" else 0},
        "avgpcon": {"type": "regression", "target": lambda j: int(j.avgpcon / j.nnuma)},
        "duration": {"type": "regression", "target": lambda j: int(j.duration / 60)}
    }

    # === Features ===
    features = {"hpcb-multitask_anon": lambda df: np.vstack(df["embedding_anon"].values)}

    # === Model candidates ===
    model_candidates = {
        "classification": {
            "KNN": KNeighborsClassifier,
            # "RF":  RandomForestClassifier,
            # "XGB": XGBClassifier
        },
        "regression": {
            "KNN": KNeighborsRegressor,
            # "RF":  RandomForestRegressor,
            # "XGB": XGBRegressor
        }
    }

    # === Semantic template (giữ nguyên 1 template tối ưu) ===
    semantic_templates = [
        lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}."
    ]

    # === Loop qua từng task ===
    for task, task_info in tasks.items():
        print(f"\n🔹 Loading SBERT for task: {task}")
        model_path = f"models/finetuned_all-MiniLM-L6-v2_multitask/{task}_triplet/model"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model path not found: {model_path}")

        sbert_model = SentenceTransformer(model_path)
        print(f"✅ Loaded SBERT from {model_path}")

        # === Loop qua từng template ===
        for template_idx, template_fn in enumerate(semantic_templates, start=1):
            print(f"\n🔹 Running template {template_idx} for task {task}...")

            # Storage
            x_train, y_train, x_test, y_test = (
                {f: [] for f in features},
                [],
                {f: [] for f in features},
                []
            )

            # === Load data và encode ===
            for data_path in tqdm([
                os.path.join(data_folder, f) for f in os.listdir(data_folder)
                if f.endswith(".parquet")
            ]):
                df = pd.read_parquet(data_path)
                ym = os.path.basename(data_path).split(".parquet")[0]
                emb_save_path = os.path.join(emb_folder, f"{ym}_template{template_idx}_{task}.parquet")

                if os.path.exists(emb_save_path):
                    emb_df = pd.read_parquet(emb_save_path)
                    df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
                    print(f"✅ Loaded cached embeddings for {task}, {ym}")
                else:
                    print(f"Generating embeddings for {ym} using SBERT ({task}) ...")
                    df["merged_text"] = df.apply(template_fn, axis=1)
                    embeddings = sbert_model.encode(
                        df["merged_text"].tolist(),
                        batch_size=256,
                        show_progress_bar=True,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    df["embedding_anon"] = list(embeddings)

                    emb_df = df[["embedding_anon"]].copy()
                    emb_df.to_parquet(emb_save_path)
                    print(f"✅ Saved embeddings: {emb_save_path}")

                for feat in features:
                    x_values = list(features[feat](df))
                    if ym in test_yms:
                        x_test[feat] += x_values
                    else:
                        x_train[feat] += x_values

                y_values = df.apply(task_info["target"], axis=1).tolist()
                if ym in test_yms:
                    y_test += y_values
                else:
                    y_train += y_values

            # === Train 3 model (KNN, RF, XGB) ===
            task_type = task_info["type"]
            for feat in features:
                for model_name, model_cls in model_candidates[task_type].items():
                    print(f"\n▶ Training {model_name} for {task} ({task_type})...")

                    model_instance = model_cls(n_jobs=-1, metric="cosine", weights='distance', algorithm='auto') 
                    model_instance.fit(x_train[feat], y_train)
                    y_pred = model_instance.predict(x_test[feat])

                    if task_type == "classification":
                        report = classification_report(y_test, y_pred)
                    else:
                        report = regression_metric(y_test, y_pred)

                    result_file = os.path.join(
                        result_path, f"{model_name}_{feat}_template{template_idx}_{task}.txt"
                    )
                    with open(result_file, "w", encoding="utf-8") as f:
                        f.write(report)

                    print(f"✅ Saved result to {result_file}")

    print("\n🎯 All tasks completed successfully!")
