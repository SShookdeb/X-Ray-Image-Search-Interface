import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import load_image, get_image_transform, load_resnet_model, DEVICE


DATASET_DIR = "Dataset"
METADATA = "metadata/X-ray_Metadata - Sheet1.csv"
OUT_FILE = "embeddings/image_embeddings.npy"

def main():
    df = pd.read_csv(METADATA)
    model = load_resnet_model()
    transform = get_image_transform()

    embeddings = []
    image_names = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = None
        for root, _, files in os.walk(DATASET_DIR):
            if row["image_name"] in files:
                img_path = os.path.join(root, row["image_name"])
                break

        if img_path is None:
            continue

        img = transform(load_image(img_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(img).cpu().numpy().flatten()

        embeddings.append(emb)
        image_names.append(row["image_name"])

    np.save(OUT_FILE, {"embeddings": embeddings, "names": image_names})
    print("✅ Image embeddings saved")

if __name__ == "__main__":
    main()
