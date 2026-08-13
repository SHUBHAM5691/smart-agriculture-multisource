"""Download configured official sources and create metadata-rich JSONL chunks."""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import certifi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources.json"
OUTPUT = ROOT / "data" / "knowledge" / "chunks.jsonl"
DOWNLOADS = ROOT / "data" / "downloads"
USER_AGENT = "SmartAgricultureKnowledgeIngestor/1.0"

CROPS = (
    "rice", "wheat", "maize", "cotton", "sugarcane", "potato", "tomato",
    "banana", "chilli", "onion", "groundnut", "sorghum", "millet", "pulses",
)
TOPICS = (
    "irrigation", "nutrient", "fertilizer", "soil", "pest", "disease", "weed",
    "harvest", "storage", "weather", "variety", "sowing", "seed", "water",
)


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=90, context=context) as response:
        return response.read()


def clean_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def infer_tags(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    return [word for word in vocabulary if re.search(rf"\b{re.escape(word)}\b", lower)]


def pdf_pages(raw: bytes, source_id: str) -> list[tuple[int | None, str]]:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    path = DOWNLOADS / f"{source_id}.pdf"
    path.write_bytes(raw)
    return [(number, clean_text(page.extract_text() or "")) for number, page in enumerate(PdfReader(path).pages, 1)]


def html_pages(raw: bytes) -> list[tuple[int | None, str]]:
    soup = BeautifulSoup(raw, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()
    return [(None, clean_text(soup.get_text("\n")))]


def vikaspedia_documents(raw: bytes, source: dict) -> list[tuple[dict, list[tuple[int | None, str]]]]:
    """Expand Vikaspedia's Next.js collection page into its linked practice documents."""
    soup = BeautifulSoup(raw, "html.parser")
    payload_node = soup.find("script", id="__NEXT_DATA__")
    if not payload_node or not payload_node.string:
        return []
    payload = json.loads(payload_node.string)["props"]["pageProps"]
    root_data = payload.get("ssrPageData") or payload.get("ssrPageContent") or {}
    documents = []
    for related in root_data.get("relatedPages", []):
        url = related.get("url")
        if not url:
            continue
        child_soup = BeautifulSoup(fetch(url), "html.parser")
        child_node = child_soup.find("script", id="__NEXT_DATA__")
        if not child_node or not child_node.string:
            continue
        child = json.loads(child_node.string)["props"]["pageProps"].get("ssrPageContent") or {}
        content_html = child.get("content") or ""
        text = clean_text(BeautifulSoup(content_html, "html.parser").get_text("\n"))
        if len(text) < 80:
            continue
        updated = str(child.get("updated_at") or child.get("created_at") or "")
        child_source = {
            **source,
            "name": f"Vikaspedia - {child.get('title') or related.get('title') or 'Package of practices'}",
            "url": url,
            "publication_year": int(updated[:4]) if updated[:4].isdigit() else None,
        }
        documents.append((child_source, [(None, text)]))
    return documents


def build_chunks(source: dict, pages: list[tuple[int | None, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    records = []
    for page, text in pages:
        if len(text) < 80:
            continue
        for content in splitter.split_text(text):
            records.append({
                "content": content,
                "source_id": source["id"],
                "source": source["name"],
                "source_url": source["url"],
                "page": page,
                "state": source.get("state"),
                "scope": source.get("scope"),
                "publication_year": source.get("publication_year"),
                "crops": infer_tags(content, CROPS),
                "topics": sorted(set(source.get("default_topics", []) + infer_tags(content, TOPICS))),
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", help="Only ingest the named source id; repeatable")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = set(args.source or [])
    records = []
    for source in sources:
        if selected and source["id"] not in selected:
            continue
        print(f"Downloading {source['name']}...", file=sys.stderr)
        raw = fetch(source["url"])
        if source["kind"] == "pdf":
            documents = [(source, pdf_pages(raw, source["id"]))]
        elif source["id"] == "vikaspedia_package_of_practices":
            documents = vikaspedia_documents(raw, source)
        else:
            documents = [(source, html_pages(raw))]
        before = len(records)
        for document_source, pages in documents:
            records.extend(build_chunks(document_source, pages))
        if len(records) == before:
            print(f"WARNING: no usable chunks extracted from {source['name']}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"Wrote {len(records)} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
