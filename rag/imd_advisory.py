"""Retrieve current state agromet bulletins directly from the official IMD site."""

from __future__ import annotations

import io
import re
import ssl
from functools import lru_cache
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import certifi
from pypdf import PdfReader

IMD_ENDPOINT = "https://mausam.imd.gov.in/responsive/agrometinformation/getimageenglish_state.php?s="
STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jammu And Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Meghalaya", "Odisha", "Punjab", "Rajasthan",
    "Tamil Nadu", "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal",
)
LIVE_TERMS = re.compile(r"\b(current|today|tomorrow|latest|weather|forecast|rain|agromet|advisory)\b", re.I)


def wants_live_advisory(question: str) -> bool:
    return bool(LIVE_TERMS.search(question))


def identify_state(text: str) -> str | None:
    lower = text.lower().replace("&", "and")
    return next((state for state in STATES if state.lower() in lower), None)


def _fetch(url: str) -> bytes:
    parts = urlsplit(url)
    scheme = "https" if parts.netloc.endswith("imd.gov.in") else parts.scheme
    url = urlunsplit((scheme, parts.netloc, quote(parts.path), parts.query, parts.fragment))
    request = Request(url, headers={"User-Agent": "SmartAgricultureAssistant/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=20, context=context) as response:
        return response.read()


@lru_cache(maxsize=32)
def fetch_current_state_advisory(state: str) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    listing_url = IMD_ENDPOINT + quote(state)
    soup = BeautifulSoup(_fetch(listing_url), "html.parser")
    pdf_urls = []
    for link in soup.find_all("a", href=True):
        url = urljoin(listing_url, link["href"])
        if ".pdf" in url.lower() and url not in pdf_urls:
            pdf_urls.append(url)
    if not pdf_urls:
        raise RuntimeError(f"IMD returned no current PDF bulletin for {state}")

    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    for pdf_url in pdf_urls[:2]:
        reader = PdfReader(io.BytesIO(_fetch(pdf_url)))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for content in splitter.split_text(text):
            if len(content.strip()) >= 80:
                documents.append({
                    "content": content,
                    "source": f"IMD current agromet advisory - {state}",
                    "source_url": pdf_url,
                    "source_id": "imd_current_agromet",
                    "state": state,
                    "publication_year": None,
                    "page": None,
                    "crops": [],
                    "topics": ["current weather", "agromet advisory"],
                    "live": True,
                })
    return documents


def retrieve_live_advisory(question: str, prediction: dict | None, limit: int = 3) -> tuple[list[dict], str | None]:
    prediction_text = " ".join(str(value) for value in (prediction or {}).get("inputs", {}).values())
    state = identify_state(f"{question} {prediction_text}")
    if not state:
        return [], "A state name is required for the current IMD agromet advisory."
    try:
        query_terms = set(re.findall(r"[a-z]+", question.lower()))
        docs = fetch_current_state_advisory(state)
        docs.sort(key=lambda doc: len(query_terms & set(re.findall(r"[a-z]+", doc["content"].lower()))), reverse=True)
        return [{**doc, "rank": rank, "relevance": None} for rank, doc in enumerate(docs[:limit], 1)], None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
