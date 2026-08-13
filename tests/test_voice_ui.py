from pathlib import Path


def test_voice_ui_is_single_step_and_automatic():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'st.chat_input(' in source
    assert 'accept_audio=settings.voice_enabled' in source
    assert 'submission.audio' in source
    assert 'st.audio_input(' not in source
    assert 'language_hint=None' in source
    assert 'voice_audio_autoplay = True' in source
    assert 'autoplay=autoplay' in source
    assert '"Transcribe recording"' not in source
    assert '"Send voice question"' not in source
    assert 'st.selectbox("Spoken language"' not in source
