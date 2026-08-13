import re

from rag.knowledge_base import load_chunks, semantic_search
from utils.config import settings

_KNOWN_CROPS = {"rice", "wheat", "maize", "cotton", "sugarcane", "potato", "tomato", "banana", "chilli", "onion"}


def _filter_crop_mismatches(docs: list[dict], query: str) -> list[dict]:
    requested = set(re.findall(r"[a-z]+", query.lower())) & _KNOWN_CROPS
    if len(requested) != 1:
        return docs
    crop = next(iter(requested))
    filtered = []
    for doc in docs:
        mentioned = set(re.findall(r"[a-z]+", doc["content"].lower())) & _KNOWN_CROPS
        if not mentioned or crop in mentioned:
            filtered.append(doc)
    return filtered


def retrieve_with_diagnostics(query: str) -> tuple[list[dict], str | None]:
    try:
        chunks = load_chunks()
        candidates = semantic_search(query, min(len(chunks), settings.top_k * 3))
        docs = [
            {**chunks[index], "relevance": round(score, 3)}
            for index, score in candidates
            if score >= settings.rag_min_relevance
        ]
        docs = _filter_crop_mismatches(docs, query)[: settings.top_k]
        return ([{**doc, "rank": rank} for rank, doc in enumerate(docs, 1)], None)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
