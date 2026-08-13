"""Turn-based multilingual speech services."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit

from llm.provider import generate_text
from utils.config import settings
from utils.json_tools import extract_json_object


class VoiceError(RuntimeError):
    """A user-facing voice-service failure."""


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    tts_voice: str
    asr_prompt: str


_LANGUAGES = {
    "hi": LanguageProfile("hi", "Hindi", "hi-IN-SwaraNeural", "कृषि, उर्वरक, सिंचाई, नाइट्रोजन, फास्फोरस, पोटाश, मिट्टी, फसल"),
    "gu": LanguageProfile("gu", "Gujarati", "gu-IN-DhwaniNeural", "કૃષિ, ખાતર, સિંચાઈ, નાઇટ્રોજન, જમીન, પાક"),
    "mr": LanguageProfile("mr", "Marathi", "mr-IN-AarohiNeural", "शेती, खत, सिंचन, नायट्रोजन, माती, पीक"),
    "en": LanguageProfile("en", "English", "en-IN-NeerjaNeural", "agriculture, fertilizer, irrigation, nitrogen, phosphorus, soil, crop"),
}
_LANGUAGE_ALIASES = {
    "hindi": "hi", "gujarati": "gu", "marathi": "mr", "english": "en",
}


def available_languages() -> list[LanguageProfile]:
    return list(_LANGUAGES.values())


def resolve_language(code: str | None) -> LanguageProfile:
    normalized = (code or "hi").lower().split("-")[0]
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    return _LANGUAGES.get(normalized, _LANGUAGES["hi"])


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "farmer-question.wav",
    language_hint: str | None = None,
) -> dict:
    if not audio_bytes:
        raise VoiceError("No recorded audio was received.")
    if not settings.groq_api_key:
        raise VoiceError("GROQ_API_KEY is not configured for speech recognition.")
    try:
        from groq import Groq

        # The Groq SDK's audio resource uses an absolute /openai/v1/... path.
        # Supplying the chat base URL here would duplicate that prefix.
        configured_url = urlsplit(settings.groq_base_url)
        groq_origin = f"{configured_url.scheme}://{configured_url.netloc}"
        client = Groq(api_key=settings.groq_api_key, base_url=groq_origin)
        kwargs = {
            "file": (filename or "farmer-question.wav", audio_bytes),
            "model": settings.asr_model,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if language_hint:
            profile = resolve_language(language_hint)
            kwargs["language"] = profile.code
            kwargs["prompt"] = profile.asr_prompt
        transcription = client.audio.transcriptions.create(**kwargs)
        text = str(getattr(transcription, "text", "") or "").strip()
        detected = str(getattr(transcription, "language", "") or language_hint or "hi")
        if not text:
            raise VoiceError("Speech recognition returned an empty transcript. Please record again.")
        return {"text": text, "language": resolve_language(detected).code}
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(f"Speech recognition failed: {type(exc).__name__}: {exc}") from exc


def normalize_transcript(text: str, language_code: str) -> str:
    """Clean ASR text without changing the farmer's intent or inventing values."""
    text = " ".join((text or "").split())
    if not text:
        raise VoiceError("The transcript is empty.")
    profile = resolve_language(language_code)
    prompt = f'''Normalize this {profile.name} agricultural speech transcript.
Rules:
- Preserve the exact meaning and do not add missing facts or measurements.
- Correct only obvious ASR errors using agricultural context.
- Write spoken numbers as digits and standardize clear units such as kg, acre, hectare, mm, °C, %, and pH.
- Preserve the user's language and script; preserve commonly used English agricultural terms.
- Return JSON only: {{"normalized_text":"..."}}
Transcript: {text}
'''
    try:
        result = extract_json_object(generate_text(prompt))
        normalized = " ".join(str(result.get("normalized_text") or "").split())
        return normalized or text
    except Exception:
        return text


def _plain_speech_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#>`]", "", text)
    return " ".join(text.split())


def synthesize_speech(text: str, language_code: str) -> bytes:
    speech_text = _plain_speech_text(text)
    if not speech_text:
        raise VoiceError("There is no answer text to speak.")
    profile = resolve_language(language_code)
    try:
        import edge_tts

        audio_parts = [
            chunk["data"]
            for chunk in edge_tts.Communicate(speech_text, profile.tts_voice).stream_sync()
            if chunk["type"] == "audio"
        ]
        if not audio_parts:
            raise VoiceError("The TTS provider returned no audio.")
        return b"".join(audio_parts)
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(f"Voice generation failed: {type(exc).__name__}: {exc}") from exc
