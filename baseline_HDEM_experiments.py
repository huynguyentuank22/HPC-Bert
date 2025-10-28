import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from HDEM import Dynamic_Weighted_Ensemble

# -----------------------------
# Config
# -----------------------------
data_folder = "data_train"
data_folder_embedd = "data_embedding"
os.makedirs(data_folder_embedd, exist_ok=True)

result_path = "baseline_results"
os.makedirs(result_path, exist_ok=True)

test_yms = ["24_04"]  # tháng dùng cho test (sau sẽ tách 50/50 thành val/test)

# -----------------------------
# Task definitions
# -----------------------------
tasks = {
    "duration": {
        "type": "regression",
        "target": lambda j: int(j.duration / 60)
    },
    "avgpcon": {
        "type": "regression",
        "target": lambda j: int(j.avgpcon / j.nnuma)
    }
}

# -----------------------------
# Feature extractors
# -----------------------------
features = {
    "int_anon": lambda df: df[["jnam", "usr", "jobenv_req"]].apply(
        lambda c: c.apply(
            lambda j: int(j.split("_")[-1])
            if isinstance(j, str) and j.split("_")[-1].isdigit()
            else 0
        )
    ).values,
    "sb_anon": lambda df: np.vstack(df["embedding_anon"].values),
    "sb": lambda df: np.vstack(df["embedding"].values),
}

# -----------------------------
# Storage init
# -----------------------------
x_train = {f: [] for f in features}
x_test_temp = {f: [] for f in features}
y_train = {t: [] for t in tasks}
y_test_temp = {t: [] for t in tasks}

# -----------------------------
# SBERT model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load and preprocess data
# -----------------------------
print("Loading data and generating/loading embeddings...")
for data_path in tqdm(sorted([
    os.path.join(data_folder, f)
    for f in os.listdir(data_folder)
    if os.path.isfile(os.path.join(data_folder, f)) and f.endswith(".parquet")
])):
    df = pd.read_parquet(data_path)
    ym = os.path.basename(data_path).split(".parquet")[0]

    # === 🔹 Tạo hoặc load embedding_anon ===
    emb_save_path = os.path.join(data_folder_embedd, f"{ym}_embedding_only.parquet")

    if os.path.exists(emb_save_path):
        emb_df = pd.read_parquet(emb_save_path)
        df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
        print(f"✅ Loaded existing embeddings for {ym}")
    else:
        print(f"Generating SBert embeddings for {ym} ...")
        df["merged_text"] = df.apply(
            lambda r: f"{r['usr']}, {r['jnam']}, {r['jobenv_req']}", axis=1
        )
        embeddings = sbert_model.encode(
            df["merged_text"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            device=device,
        )
        df["embedding_anon"] = list(embeddings)
        emb_df = df[["embedding_anon"]].copy()
        emb_df.to_parquet(emb_save_path)
        print(f"✅ Saved new embeddings: {emb_save_path}")

    # === Trích xuất features ===
    for feat in features:
        x_values = list(features[feat](df))
        if ym in test_yms:
            x_test_temp[feat] += x_values
        else:
            x_train[feat] += x_values

    # === Trích xuất targets ===
    for task in tasks:
        y_values = df.apply(tasks[task]["target"], axis=1).tolist()
        if ym in test_yms:
            y_test_temp[task] += y_values
        else:
            y_train[task] += y_values

# -----------------------------
# Split test_temp → val/test 50/50
# -----------------------------
def split_half(arr_list):
    """Chia đôi danh sách làm val/test 50/50."""
    n = len(arr_list)

    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)

    half = n // 2
    idx_val, idx_test = idx[:half], idx[half:]
    arr = np.array(arr_list)
    return list(arr[idx_val]), list(arr[idx_test])

x_val = {f: [] for f in features}
x_test = {f: [] for f in features}
y_val = {t: [] for t in tasks}
y_test = {t: [] for t in tasks}

for feat in features:
    if len(x_test_temp[feat]) > 0:
        x_val[feat], x_test[feat] = split_half(x_test_temp[feat])

for task in tasks:
    if len(y_test_temp[task]) > 0:
        y_val[task], y_test[task] = split_half(y_test_temp[task])

# -----------------------------
# Run HDEM per task & feature
# -----------------------------
for task in tasks:
    print(f"\n=== Running HDEM for task: {task} ===")

    for feat_name in features:
        print(f"\n--- Feature: {feat_name} ---")

        X_train = np.array(x_train[feat_name])
        X_val = np.array(x_val[feat_name]) if len(x_val[feat_name]) > 0 else np.empty((0,))
        X_test = np.array(x_test[feat_name]) if len(x_test[feat_name]) > 0 else np.empty((0,))
        y_tr = np.array(y_train[task])
        y_v = np.array(y_val[task]) if len(y_val[task]) > 0 else np.empty((0,))
        y_te = np.array(y_test[task]) if len(y_test[task]) > 0 else np.empty((0,))

        # 🧩 Debug: Kiểm tra số lượng mẫu
        print(f"📊 Data shapes for {feat_name} - {task}:")
        print(f"    X_train: {X_train.shape}, y_train: {y_tr.shape}")
        print(f"    X_val:   {X_val.shape}, y_val:   {y_v.shape}")
        print(f"    X_test:  {X_test.shape}, y_test:  {y_te.shape}")

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_val.ndim == 1:
            X_val = X_val.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # Nếu thiếu val/test, chia lại từ train
        if X_val.shape[0] == 0 or y_v.shape[0] == 0:
            print("⚠️ Not enough val/test data, splitting from train...")
            X_train, X_val, y_tr, y_v = train_test_split(X_train, y_tr, test_size=0.2, random_state=42)

        class DummyScaler:
            """
            Scaler giả lập để dùng trong pipeline khi không cần chuẩn hóa dữ liệu.
            Giữ nguyên dữ liệu đầu vào, nhưng vẫn có đủ các hàm/thuộc tính như StandardScaler.
            """
            def __init__(self):
                self.n_features_in_ = None

            def fit(self, X, y=None):
                # Lưu số lượng feature để tương thích sklearn
                if hasattr(X, "shape"):
                    self.n_features_in_ = X.shape[1]
                else:
                    self.n_features_in_ = len(X[0]) if len(X) > 0 else 0
                return self

            def transform(self, X):
                # Trả về nguyên xi dữ liệu đầu vào
                return X

            def fit_transform(self, X, y=None):
                self.fit(X, y)
                return X

            def inverse_transform(self, X):
                # Vì DummyScaler không thay đổi dữ liệu, inverse = chính dữ liệu
                return X

        # Chuẩn hóa
        scaler = DummyScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test) if X_test.shape[0] > 0 else X_test

        # Khởi tạo HDEM
        hdem = Dynamic_Weighted_Ensemble(X_train_s, X_val_s, X_test_s, y_tr, y_v, y_te, scaler=scaler)
        model_list = [
            ["extratrees", "randomforest", "xgboost"],
            ["randomforest", "mlp", "gradientboosting"],
            ["lasso", "xgboost", "extratrees"],
        ]
        hdem.num_sub = len(model_list)
        hdem.init_base_sub(model_list)
        hdem.meta_model_name = "gradientboosting"

        # Chạy mô hình
        try:
            metrics = hdem.run_model()
        except Exception as e:
            print(f"❌ Error running HDEM for {feat_name} ({task}): {e}")
            continue

        mae_value = metrics.get("MAE", None)

        # Ghi file riêng
        result_file = os.path.join(result_path, f"HDEM_{feat_name}_{task}.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"MAE: {mae_value:.4f}\n")
        print(f"✅ Saved MAE result to {result_file}")

print("\n🎯 All HDEM experiments finished successfully!")
