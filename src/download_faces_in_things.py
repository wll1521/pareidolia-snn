import os
import pandas as pd
import requests
from tqdm import tqdm

def download_faces_in_things(csv_path="data/metadata.csv", save_dir="data/faces_in_things"):
    df = pd.read_csv(csv_path)
    url_col = 'url' if 'url' in df.columns else df.columns[0]

    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading {len(df)} images to {save_dir}...")

    for i, url in tqdm(enumerate(df[url_col]), total=len(df)):
        try:
            img_data = requests.get(url, timeout=10).content
            with open(f"{save_dir}/img_{i}.jpg", "wb") as f:
                f.write(img_data)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    download_faces_in_things()
