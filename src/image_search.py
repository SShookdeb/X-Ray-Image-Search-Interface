import numpy as np
import torch

from src.utils import (
    load_image,
    get_image_transform,
    load_resnet_model,
    DEVICE,
)

EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"


class ImageSearch:
    def __init__(self):
        # Load saved embeddings
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()

        self.embeddings = np.array(data["embeddings"], dtype=np.float32)
        self.names = data["names"]

        # Normalize stored embeddings (IMPORTANT)
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        # Load model & transforms
        self.model = load_resnet_model()
        self.model.eval()

        self.transform = get_image_transform()

    def search(self, image_path, top_k=5):
        # Load & preprocess query image
        img = load_image(image_path)
        img = self.transform(img).unsqueeze(0).to(DEVICE)

        # Extract query embedding
        with torch.no_grad():
            query_emb = self.model(img)

        query_emb = query_emb.squeeze(0).cpu().numpy().astype(np.float32)

        # Normalize query embedding
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Cosine similarity (dot product of normalized vectors)
        similarities = self.embeddings @ query_emb

        # Top-K most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.names[i], float(similarities[i]))
            for i in top_indices
        ]

        return results
