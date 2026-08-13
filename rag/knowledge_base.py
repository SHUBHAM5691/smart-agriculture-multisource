import json
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from utils.config import settings

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_JSONL_PATH = ROOT / "data" / "knowledge" / "chunks.jsonl"
FALLBACK_PATH = ROOT / "data" / "agriculture_knowledge.txt"
INDEX_DIR = ROOT / "data" / "vector_index"
FAISS_INDEX_PATH = INDEX_DIR / "knowledge.faiss"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"


def _corpus_fingerprint() -> str:
    """Identify the exact corpus/model combination represented by the saved index."""
    source = KNOWLEDGE_JSONL_PATH if KNOWLEDGE_JSONL_PATH.exists() else FALLBACK_PATH
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(settings.embedding_model.encode("utf-8"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    if KNOWLEDGE_JSONL_PATH.exists():
        chunks = [json.loads(line) for line in KNOWLEDGE_JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if chunks:
            return chunks

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = FALLBACK_PATH.read_text(encoding="utf-8")
    parts = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""]
    ).split_text(text)
    return [{
        "content": chunk,
        "source": "Curated agriculture fallback",
        "source_url": None,
        "source_id": "curated_fallback",
        "page": None,
        "state": None,
        "publication_year": None,
        "crops": [],
        "topics": [],
    } for chunk in parts]


@lru_cache(maxsize=1)
def get_embedding_model():
    # Import lazily so the PyTorch/transformers native libraries are not loaded
    # while users are running the independent LightGBM prediction workflow.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_vector_index():
    import faiss

    fingerprint = _corpus_fingerprint()
    if FAISS_INDEX_PATH.exists() and INDEX_METADATA_PATH.exists():
        try:
            metadata = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
            if (
                metadata.get("fingerprint") == fingerprint
                and metadata.get("chunk_count") == len(load_chunks())
            ):
                return faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception:
            # A partial/corrupt index is safe to discard because it is derived data.
            pass

    embeddings = get_embedding_model().encode(
        [chunk["content"] for chunk in load_chunks()], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    temporary_index = FAISS_INDEX_PATH.with_suffix(".tmp.faiss")
    temporary_metadata = INDEX_METADATA_PATH.with_suffix(".tmp.json")
    faiss.write_index(index, str(temporary_index))
    temporary_metadata.write_text(
        json.dumps({
            "fingerprint": fingerprint,
            "chunk_count": len(load_chunks()),
            "embedding_model": settings.embedding_model,
            "dimension": int(embeddings.shape[1]),
        }, indent=2),
        encoding="utf-8",
    )
    temporary_index.replace(FAISS_INDEX_PATH)
    temporary_metadata.replace(INDEX_METADATA_PATH)
    return index


def prepare_vector_index() -> dict:
    """Build or load the persistent index during application startup."""
    existed = FAISS_INDEX_PATH.exists() and INDEX_METADATA_PATH.exists()
    index = get_vector_index()
    return {
        "chunk_count": int(index.ntotal),
        "loaded_from_disk": existed,
        "index_path": str(FAISS_INDEX_PATH),
    }


def semantic_search(query: str, limit: int) -> list[tuple[int, float]]:
    query_embedding = get_embedding_model().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, indices = get_vector_index().search(query_embedding, limit)
    return [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
