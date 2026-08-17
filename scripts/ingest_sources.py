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

# One-based physical PDF pages where each state section begins. The ICAR PDF
# contains duplicated physical pages, so only every second page is retained.
KHARIF_SECTION_STARTS = [
    (16, ["Andaman and Nicobar Islands"]),
    (22, ["Andhra Pradesh"]),
    (34, ["Arunachal Pradesh"]),
    (40, ["Assam"]),
    (48, ["Bihar"]),
    (56, ["Chhattisgarh"]),
    (62, ["Goa"]),
    (66, ["Gujarat"]),
    (78, ["Haryana", "Delhi"]),
    (84, ["Himachal Pradesh"]),
    (100, ["Jammu and Kashmir"]),
    (120, ["Jharkhand"]),
    (126, ["Karnataka"]),
    (136, ["Kerala"]),
    (142, ["Ladakh"]),
    (146, ["Lakshadweep"]),
    (150, ["Madhya Pradesh"]),
    (156, ["Maharashtra"]),
    (178, ["Manipur"]),
    (184, ["Meghalaya"]),
    (188, ["Mizoram"]),
    (194, ["Nagaland"]),
    (204, ["Odisha"]),
    (208, ["Punjab"]),
    (226, ["Rajasthan"]),
    (250, ["Sikkim"]),
    (254, ["Tamil Nadu", "Puducherry"]),
    (260, ["Telangana"]),
    (266, ["Tripura"]),
    (274, ["Uttar Pradesh"]),
    (280, ["Uttarakhand"]),
    (296, ["West Bengal"]),
    (308, []),
]


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


def kharif_state_documents(raw: bytes, source: dict) -> list[tuple[dict, list[tuple[int | None, str]]]]:
    pages = pdf_pages(raw, source["id"])
    documents = []
    for section_index, (start_page, states) in enumerate(KHARIF_SECTION_STARTS[:-1]):
        end_page = KHARIF_SECTION_STARTS[section_index + 1][0]
        # The official PDF repeats each extracted physical page twice.
        unique_pages = pages[start_page - 1:end_page - 1:2]
        for state in states:
            documents.append(({
                **source,
                "name": f"{source['name']} - {state}",
                "state": state,
            }, unique_pages))
    return documents


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
                "season": source.get("season"),
                "crops": infer_tags(content, CROPS),
                "topics": sorted(set(source.get("default_topics", []) + infer_tags(content, TOPICS))),
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", help="Only ingest the named source id; repeatable")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--append", action="store_true", help="Replace selected sources while preserving other existing chunks")
    args = parser.parse_args()
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = set(args.source or [])
    records = []
    if args.append and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("source_id") not in selected:
                records.append(record)
    for source in sources:
        if selected and source["id"] not in selected:
            continue
        print(f"Downloading {source['name']}...", file=sys.stderr)
        raw = fetch(source["url"])
        if source["kind"] == "state_sectioned_pdf":
            documents = kharif_state_documents(raw, source)
        elif source["kind"] == "pdf":
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
