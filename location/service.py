from __future__ import annotations

import json
import ssl
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
STATE_ALIASES = {
    "national capital territory of delhi": "Delhi",
    "nct of delhi": "Delhi",
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
    "jammu and kashmir": "Jammu and Kashmir",
    "andaman and nicobar": "Andaman and Nicobar Islands",
}


class LocationError(RuntimeError):
    pass


def _clean(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_state(value: object, known_states: list[str]) -> str | None:
    text = _clean(value)
    if not text:
        return None
    normalized = text.casefold()
    alias = STATE_ALIASES.get(normalized)
    if alias:
        return alias
    for state in known_states:
        candidate = state.casefold()
        if normalized == candidate or candidate in normalized or normalized in candidate:
            return state
    return text


@lru_cache(maxsize=64)
def reverse_geocode(latitude: float, longitude: float) -> dict:
    query = urlencode({
        "lat": f"{latitude:.6f}",
        "lon": f"{longitude:.6f}",
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 10,
    })
    request = Request(
        f"{NOMINATIM_REVERSE_URL}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "SmartAgricultureM6/1.0 (educational decision-support application)",
        },
    )
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=15, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise LocationError(f"Location lookup failed: {exc}") from exc
    address = payload.get("address") or {}
    district = next(
        (_clean(address.get(key)) for key in (
            "state_district", "district", "county", "city_district", "city", "town"
        ) if _clean(address.get(key))),
        None,
    )
    return {
        "state": _clean(address.get("state")),
        "district": district,
        "display_name": _clean(payload.get("display_name")),
    }
