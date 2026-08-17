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


def _filter_location_scope(docs: list[dict], state: str | None) -> list[dict]:
    """Keep national chunks and only regional chunks matching the selected state."""
    normalized_state = (state or "").strip().casefold()
    filtered = []
    for doc in docs:
        document_state = str(doc.get("state") or "").strip()
        if not document_state:
            filtered.append(doc)
        elif normalized_state and document_state.casefold() == normalized_state:
            filtered.append(doc)
    return filtered


def _rerank_seasonal_sources(docs: list[dict], query: str) -> list[dict]:
    query_terms = set(re.findall(r"[a-z]+", query.lower()))
    def score(doc: dict) -> float:
        value = float(doc.get("relevance") or 0.0)
        season = str(doc.get("season") or "").lower()
        if season and season in query_terms:
            value += 0.12
        return value
    return sorted(docs, key=score, reverse=True)


def retrieve_with_diagnostics(query: str, state: str | None = None) -> tuple[list[dict], str | None]:
    try:
        chunks = load_chunks()
        # Over-fetch before metadata filtering because regional TNAU chunks dominate
        # the corpus and could otherwise crowd national chunks out of the candidate set.
        candidate_limit = min(len(chunks), max(settings.top_k * 125, 500))
        candidates = semantic_search(query, candidate_limit)
        docs = [
            {**chunks[index], "relevance": round(score, 3)}
            for index, score in candidates
            if score >= settings.rag_min_relevance
        ]
        docs = _filter_location_scope(docs, state)
        docs = _filter_crop_mismatches(docs, query)
        docs = _rerank_seasonal_sources(docs, query)[: settings.top_k]
        return ([{**doc, "rank": rank} for rank, doc in enumerate(docs, 1)], None)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
