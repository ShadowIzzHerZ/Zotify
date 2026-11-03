"""Embed catalog item text with sentence-transformers."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"


def main():
    df = pd.read_parquet("data/catalog.parquet")
    model = SentenceTransformer(MODEL)
    emb = model.encode(df.text.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    np.save("data/embeddings.npy", emb.astype(np.float32))
    print(emb.shape)


if __name__ == "__main__":
    main()
