from tqdm import tqdm
import pandas as pd
import json
import requests
import os

def get_data(idx_start=0, idx_end=None):
    # Get feature list
    feature_list_df =  pd.read_csv("docs/feature_list.csv")
    feature_list = feature_list_df["Column"].tolist()

    # Get list of file paths
    with open("docs/croissant.json", "r") as f:
        croissant_files = json.load(f)

        data_folder = croissant_files["distribution"][1:]

        if idx_end is None:
            idx_end = len(data_folder)
            
        data_folder = data_folder[idx_start:idx_end]

        for data_file in tqdm(data_folder):
            file_name = data_file["@id"]
            url = data_file["contentUrl"]

            save_path = os.path.join("data", file_name)
            # Gửi yêu cầu tải
            response = requests.get(url, stream=True)
            response.raise_for_status()  # kiểm tra lỗi

            # Lấy kích thước file (nếu có)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            # Hiển thị progress bar
            with open(save_path, "wb") as f, tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {file_name}"
            ) as bar:
                for chunk in response.iter_content(block_size):
                    bar.update(len(chunk))
                    f.write(chunk)

            print(f"\nSaved {file_name} to {save_path}\n")

def verify_data():
    data_folder = os.listdir('./data')
    jobs_list = pd.read_csv('docs/parquet_files_summary.csv')['# of jobs'].tolist()

    print("Check if all parquet files have the correct number of rows:")
    print(all(pd.read_parquet(f'./data/{file}').shape[0] == jobs for file, jobs in tqdm(zip(data_folder, jobs_list))))


if __name__ == "__main__":
    # get_data(idx_start=11)
    verify_data()