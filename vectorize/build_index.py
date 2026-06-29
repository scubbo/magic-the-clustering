"""
Builds a FAISS IndexFlatIP from embeddings.npy and saves it to artifacts/index.faiss.
Run after embed.py has produced the embeddings.

IndexFlatIP on L2-normalized vectors gives exact cosine similarity search.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"


def build(embeddings_path: Path = ARTIFACTS_DIR / "embeddings.npy",
          index_path: Path = ARTIFACTS_DIR / "index.faiss") -> None:
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss not installed. Run: pip install faiss-cpu  (or faiss-gpu)", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embeddings from {embeddings_path} ...", flush=True)
    embeddings = np.load(embeddings_path)
    n, d = embeddings.shape
    print(f"  shape: {embeddings.shape}", flush=True)

    print("Building IndexFlatIP ...", flush=True)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"  index contains {index.ntotal} vectors", flush=True)

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}", flush=True)


if __name__ == "__main__":
    build()
