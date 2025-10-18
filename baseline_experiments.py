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

    # Result path
    result_path = "baseline_results"
    os.makedirs(result_path, exist_ok=True)
    
    # Data folder  
    data_folder = "data_train"
    data_folder_embedd = "data_embedding"
    
    # Months of data to use for the testing phase
    test_yms = ["24_04"] # "23_06", "23_07", "23_08", "23_09", "23_10", "23_11", "23_12", "24_01", "24_02", "24_03",
    
    # Define regression metric
    regression_metric = lambda y_true, y_pred: f"MAE: {mean_absolute_error(y_true, y_pred):.4f}"
    
    # Definition of tasks
    tasks = {
        "ec": {
            "type": "classification",
            "target": lambda j: 1 if j["exit state"] == "completed" else 0
        },
        "pclass": {
            "type": "classification",
            "target": lambda j: 1 if j.pclass == "compute-bound" else 0
        },
        "avgpcon": {
            "type": "regression",
            "target": lambda j: int(j.avgpcon / j.nnuma)
        },
        "duration": {
            "type": "regression",
            "target": lambda j: int(j.duration / 60)
        }
    }
    
    # Definition of the input encoding
    features = {
        "int_anon": lambda df: df[["jnam", "usr", "jobenv_req"]].apply(
            lambda c: c.apply(lambda j: int(j.split("_")[-1]) if isinstance(j, str) and j.split("_")[-1].isdigit() else 0)
        ).values,
        "sb_anon": lambda df: np.vstack(df["embedding_anon"].values),
        "sb": lambda df: np.vstack(df["embedding"].values)
    }
    
    # Available ML models
    model_candidates = {
        "classification": {
            "KNN": KNeighborsClassifier,
            # "RF": RandomForestClassifier,
            # "XGB": XGBClassifier
        },
        "regression": {
            "KNN": KNeighborsRegressor,
            # "RF": RandomForestRegressor,
            # "XGB": XGBRegressor
        }
    }
    
    # Prepare storage
    x_train, y_train, x_test, y_test = (
        {f: [] for f in features},
        {t: [] for t in tasks},
        {f: [] for f in features},
        {t: [] for t in tasks}
    )

    # === Load and split data ===
    print("Loading data...")
    
    # Load SBERT model once (outside the loop)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    for data_path in tqdm([
        os.path.join(data_folder, f) for f in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, f)) and f.endswith(".parquet")
    ]):
        df = pd.read_parquet(data_path)
        ym = os.path.basename(data_path).split(".parquet")[0]
        
        # === 🔹 Tạo hoặc load embedding_anon nếu chưa có ===
        emb_save_path = os.path.join(data_folder_embedd, f"{ym}_embedding_only.parquet")

        if os.path.exists(emb_save_path):
            # 🔹 Nếu file embedding đã tồn tại → load lại
            emb_df = pd.read_parquet(emb_save_path)
            df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
            print(f"✅ Loaded existing embeddings for {ym}")

        else:
            # 🔹 Nếu chưa có → tạo mới
            print(f"Generating SBert embeddings for {ym} ...")
            df["merged_text"] = df.apply(
                lambda r: f"{r['usr']}, {r['jnam']}, {r['jobenv_req']}", axis=1
            )
            embeddings = sbert_model.encode(
                df["merged_text"].tolist(),
                batch_size=256,
                show_progress_bar=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            df["embedding_anon"] = list(embeddings)

            # Lưu lại riêng
            emb_df = df[["embedding_anon"]].copy()
            emb_df.to_parquet(emb_save_path)
            print(f"✅ Saved new embeddings: {emb_save_path}")

        for feat in features:
            x_values = list(features[feat](df))
            if ym in test_yms:
                x_test[feat] += x_values
            else:
                x_train[feat] += x_values
        
        for task in tasks:
            y_values = df.apply(tasks[task]["target"], axis=1).tolist()
            if ym in test_yms:
                y_test[task] += y_values
            else:
                y_train[task] += y_values
    
    # === Train and evaluate each model ===
    print("\nTraining and evaluating models...\n")
    for feat in features:
        for task in tasks:
            task_type = tasks[task]["type"]
            
            for model_name, model_cls in model_candidates[task_type].items():
                print(f"Running {model_name} on {task} ({task_type}) with features {feat}...")
                
                # Initialize model
                model_instance = model_cls(n_jobs=-1)
                model_instance.fit(x_train[feat], y_train[task])
                y_pred = model_instance.predict(x_test[feat])
                
                # Choose metric
                if task_type == "classification":
                    report = classification_report(y_test[task], y_pred)
                else:
                    report = regression_metric(y_test[task], y_pred)
                
                # Save results
                result_file = os.path.join(result_path, f"{model_name}_{feat}_{task}.txt")
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(report)
                
                print(f"✅ Saved {result_file}")

    print("\nAll experiments completed successfully!")
