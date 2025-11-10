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
from catboost import CatBoostClassifier, CatBoostRegressor

if __name__ == "__main__":
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # === Paths ===
    result_path = "baseline_results"
    pred_path = "predict_results"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    data_folder = "data_train"
    emb_folder = "data_embedding_semantic"
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

    # === Only using semantic embeddings ===
    features = {"sb_anon": lambda df: np.vstack(df["embedding_anon"].values)}

    # === Model candidates ===
    model_candidates = {
    "classification": {
        "KNN": KNeighborsClassifier,
        "RF":  RandomForestClassifier,
        "XGB": XGBClassifier,
        "CatBoost": CatBoostClassifier
    },
    "regression": {
        "KNN": KNeighborsRegressor,
        "RF":  RandomForestRegressor,
        "XGB": XGBRegressor,
        "CatBoost": CatBoostRegressor
    }
}

    # === Define multiple semantic templates ===
    semantic_templates = [
        # Template 1: Cấu trúc bị động rút gọn (Compact Passive Construction)
        # Diễn đạt hành động “submit” ở dạng bị động nhằm đa dạng hóa ngữ pháp và trật tự thông tin.
        # Câu mô tả ngắn gọn, đưa "Job" lên đầu để nhấn mạnh đối tượng công việc, đồng thời vẫn giữ đủ ba thực thể: user – job – environment.
        lambda r: f"Job submitted by user {r['usr']} with name {r['jnam']} requiring environment {r['jobenv_req']}.",

        # Template 2: Tập trung vào Chủ thể/Hành động cơ bản (User–Job–Environment Relation)
        # Mô tả trực tiếp mối quan hệ giữa người dùng (user), công việc (job) và môi trường tính toán (environment).
        # Cấu trúc đơn giản, dễ hiểu, giúp mô hình học được ngữ cảnh cơ bản của hành động “submit”.
        lambda r: f"The user profile {r['usr']} submitted a computation {r['jnam']} to the environment {r['jobenv_req']}.",
        
        # Template 3: Tập trung vào Ngữ cảnh/Môi trường (Environment Focus)
        # Nhấn mạnh rằng môi trường được yêu cầu cho một tác vụ cụ thể của người dùng.
        lambda r: f"The high-priority computational environment {r['jobenv_req']} was specifically requested by user {r['usr']} for running the job named {r['jnam']}.",
        
        # Template 4: Tập trung vào Đối tượng/Công việc (Job Focus)
        # Nhấn mạnh tính chất của Job và vai trò của User/Environment đối với Job đó.
        lambda r: f"Job {r['jnam']}, which will be executed by {r['usr']}, requires exclusive access to the infrastructure {r['jobenv_req']}.",
        
        # Template 5: Mối quan hệ Hành động & Liên kết (Action & Association)
        # Dùng các động từ mạnh hơn để mô tả hành động lập lịch/chạy.
        lambda r: f"The scheduling system recorded that {r['usr']} is deploying job {r['jnam']} onto the {r['jobenv_req']} partition.",
        
        # Template 6: Cú pháp bị động (Passive Voice)
        # Thử nghiệm các cấu trúc câu khác để buộc mô hình học các phụ thuộc khác.
        lambda r: f"The hardware configuration {r['jobenv_req']} is being utilized by job {r['jnam']} which was initialized by {r['usr']}.",

        lambda r: (
            f"Job {r['jnam']} (ID {r['jid']}), submitted by user {r['usr']}, requested {r['cnumr']} cores "
            f"and {r['nnumr']} nodes with a priority level of {r['pri']}. "
            f"The job was submitted at {r['adt']} and inserted into the queue at {r['qdt']}. "
            f"It requested a CPU frequency of {r['freq_req']} MHz and a memory limit of {r['mszl']:.2f} GB, "
            f"running in the {r['jobenv_req']} environment with an elapsed time limit of {r['elpl']:.2f} seconds. "
        )
    ]

    # === Load SBERT once ===
    print("Loading SBERT model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # === Iterate through each template ===
    for template_idx, template_fn in enumerate(semantic_templates, start=1):
        print(f"\n🔹 Running Semantic Template {template_idx}/{len(semantic_templates)} ...")

        # Prepare storage
        x_train, y_train, x_test, y_test = (
            {f: [] for f in features},
            {t: [] for t in tasks},
            {f: [] for f in features},
            {t: [] for t in tasks}
        )

        # === Load data and generate embeddings for current template ===
        for data_path in tqdm([
            os.path.join(data_folder, f) for f in os.listdir(data_folder)
            if os.path.isfile(os.path.join(data_folder, f)) and f.endswith(".parquet")
        ]):
            df = pd.read_parquet(data_path)
            ym = os.path.basename(data_path).split(".parquet")[0]

            emb_save_path = os.path.join(emb_folder, f"{ym}_template{template_idx}.parquet")
            # emb_save_path = os.path.join(emb_folder, f"{ym}_template_full.parquet")

            if os.path.exists(emb_save_path):
                emb_df = pd.read_parquet(emb_save_path)
                df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
                print(f"✅ Loaded cached embeddings for template {template_idx}, {ym}")
            else:
                print(f"Generating embeddings for {ym} using template {template_idx}...")
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

            for task in tasks:
                y_values = df.apply(tasks[task]["target"], axis=1).tolist()
                if ym in test_yms:
                    y_test[task] += y_values
                else:
                    y_train[task] += y_values

        # === Train and evaluate models for current template ===
        print(f"\n🏁 Training models for template {template_idx}...\n")
        for feat in features:
            for task in tasks:
                task_type = tasks[task]["type"]

                for model_name, model_cls in model_candidates[task_type].items():
                    print(f"▶ Template {template_idx} | {model_name} | {task} ({task_type})")

                    model_instance = model_cls(n_jobs=-1) if model_name != "CatBoost" else model_cls(task_type="GPU") if torch.cuda.is_available() else model_cls(thread_count=-1)
                    model_instance.fit(x_train[feat], y_train[task])
                    y_pred = model_instance.predict(x_test[feat])

                    if task_type == "classification":
                        report = classification_report(y_test[task], y_pred)
                    else:
                        report = regression_metric(y_test[task], y_pred)

                    result_file = os.path.join(
                        result_path, f"{model_name}_{feat}_template{template_idx}_{task}.txt"
                    )
                    # result_file = os.path.join(
                    #     result_path, f"{model_name}_{feat}_template_full_{task}.txt"
                    # )
                    with open(result_file, "w", encoding="utf-8") as f:
                        f.write(report)

                    print(f"✅ Saved result to {result_file}")

                    # === Save predictions for later analysis ===
                    pred_df = pd.DataFrame({
                        "y_true": y_test[task],
                        "y_pred": y_pred
                    })
                    pred_file = os.path.join(
                        pred_path, f"{model_name}_{feat}_template{template_idx}_{task}_pred.csv"
                    )
                    # pred_file = os.path.join(
                    #     pred_path, f"{model_name}_{feat}_template_full_{task}_pred.csv"
                    # )
                    pred_df.to_csv(pred_file, index=False)
                    print(f"💾 Saved predictions to {pred_file}")


    print("\n🎯 All templates completed successfully!")
