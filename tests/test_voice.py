from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voice.service import VoiceError, _plain_speech_text, normalize_transcript, resolve_language, transcribe_audio


def test_language_registry_defaults_safely_to_hindi():
    assert resolve_language("mr-IN").name == "Marathi"
    assert resolve_language("unsupported").name == "Hindi"


def test_normalization_preserves_original_when_llm_is_unavailable():
    with patch("voice.service.generate_text", side_effect=TimeoutError("offline")):
        assert normalize_transcript("  मेरी फसल ठीक है?  ", "hi") == "मेरी फसल ठीक है?"


def test_normalization_uses_structured_llm_result():
    with patch(
        "voice.service.generate_text",
        return_value='{"normalized_text":"मेरे पास 2 acre भूमि है।"}',
    ):
        assert normalize_transcript("मेरे पास दो एकड़ भूमि है", "hi") == "मेरे पास 2 acre भूमि है।"


def test_markdown_is_removed_before_tts():
    assert _plain_speech_text("Use **Urea**. [Source](https://example.com)") == "Use Urea. Source"


def test_empty_audio_is_rejected_without_api_call():
    with pytest.raises(VoiceError, match="No recorded audio"):
        transcribe_audio(b"")


@patch("voice.service.settings")
@patch("groq.Groq")
def test_asr_passes_hindi_hint_and_returns_language(client_class, mocked_settings):
    mocked_settings.groq_api_key = "test"
    mocked_settings.groq_base_url = "https://api.groq.com/openai/v1"
    mocked_settings.asr_model = "whisper-large-v3"
    create = MagicMock(return_value=SimpleNamespace(text="धान में पानी कब दें?", language="hindi"))
    client_class.return_value.audio.transcriptions.create = create

    result = transcribe_audio(b"wav", "question.wav", "hi")

    assert result == {"text": "धान में पानी कब दें?", "language": "hi"}
    client_class.assert_called_once_with(api_key="test", base_url="https://api.groq.com")
    assert create.call_args.kwargs["language"] == "hi"
    assert "कृषि" in create.call_args.kwargs["prompt"]
