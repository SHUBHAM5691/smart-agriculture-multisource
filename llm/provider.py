import json
import logging
import ssl
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from socket import timeout as SocketTimeout

import certifi

from utils.config import settings

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    pass


def _call_groq(prompt_text: str, timeout: int = 60) -> str:
    if not settings.groq_api_key:
        raise ProviderError("GROQ_API_KEY is not configured")
    payload = {
        "model": settings.groq_model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.2,
    }
    request = Request(
        f"{settings.groq_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "smart-agriculture-groq/1.0",
            "Authorization": f"Bearer {settings.groq_api_key}",
        },
    )
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=timeout, context=tls_context) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body)
    choices = data.get("choices") or []
    text = choices[0].get("message", {}).get("content", "") if choices else ""
    if not text or not str(text).strip():
        raise ProviderError("Groq returned an empty response")
    text = str(text).strip()
    logger.debug("Groq returned a response")
    return text


def generate_text(prompt_text: str) -> str:
    max_retries = settings.groq_max_retries
    timeout = settings.groq_timeout
    backoff_base = settings.groq_retry_backoff

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return _call_groq(prompt_text, timeout=timeout)
        except (HTTPError, URLError, ValueError, json.JSONDecodeError, OSError, ProviderError, SocketTimeout) as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_for = backoff_base * (2 ** (attempt - 1))
                logger.warning("Groq attempt %s failed; retrying in %ss", attempt, sleep_for)
                time.sleep(sleep_for)
                continue
            # exhausted retries
            raise ProviderError(f"Groq failed after {max_retries} attempts: {exc}") from exc
