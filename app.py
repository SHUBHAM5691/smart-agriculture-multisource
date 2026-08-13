import pandas as pd
import streamlit as st

from memory.session_manager import clear_session, get_prediction, initialize_session
from orchestrator.workflow import answer_question, generate_prediction
from rag.knowledge_base import prepare_vector_index
from utils.config import settings
from voice import (
    VoiceError,
    normalize_transcript,
    resolve_language,
    synthesize_speech,
    transcribe_audio,
)


DEFAULT_VALUES = {
    "crop_n": 90.0, "crop_p": 42.0, "crop_k": 43.0, "crop_temperature": 25.0,
    "crop_humidity": 80.0, "crop_ph": 6.5, "crop_rainfall": 200.0,
    "fert_soil_ph": 6.2, "fert_soil_moisture": 35.0, "fert_organic_carbon": 1.0,
    "fert_ec": 0.2, "fert_nitrogen": 80.0, "fert_phosphorus": 40.0,
    "fert_potassium": 50.0, "fert_temperature": 28.0, "fert_humidity": 70.0,
    "fert_rainfall": 180.0, "fert_used_last": 100.0, "fert_yield_last": 3.0,
    "yield_year": 2024, "yield_area": 1.0, "yield_production": 1.0,
    "yield_rainfall": 1200.0, "yield_fertilizer": 100.0, "yield_pesticide": 10.0,
}


def number(label: str, key: str, minimum: float, maximum: float, step: float = 1.0):
    default = DEFAULT_VALUES.get(key, minimum)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in (minimum, maximum, default)) and float(step).is_integer():
        return st.number_input(
            label, min_value=int(minimum), max_value=int(maximum),
            value=int(default), step=int(step), key=key,
        )
    return st.number_input(
        label, min_value=minimum, max_value=maximum,
        value=float(default), step=float(step), key=key,
    )


def select(label: str, options: list[str], key: str):
    return st.selectbox(label, options, index=0, key=key)


def crop_form() -> dict | None:
    with st.form("crop_inputs"):
        st.subheader("Crop prediction inputs")
        left, right = st.columns(2)
        with left:
            nitrogen = number("Nitrogen (N)", "crop_n", 0.0, 200.0)
            phosphorus = number("Phosphorus (P)", "crop_p", 0.0, 200.0)
            potassium = number("Potassium (K)", "crop_k", 0.0, 200.0)
            temperature = number("Temperature (°C)", "crop_temperature", -10.0, 60.0, 0.1)
        with right:
            humidity = number("Humidity (%)", "crop_humidity", 0.0, 100.0, 0.1)
            ph = number("Soil pH", "crop_ph", 0.0, 14.0, 0.1)
            rainfall = number("Rainfall (mm)", "crop_rainfall", 0.0, 3000.0, 0.1)
        submitted = st.form_submit_button("Generate Crop Prediction", type="primary")
    if not submitted:
        return None
    values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
    if any(value is None for value in values):
        st.error("Complete all crop inputs before generating the prediction.")
        return None
    return {
        "_prediction_type": "crop", "nitrogen": nitrogen, "phosphorus": phosphorus,
        "potassium": potassium, "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall,
    }


def fertilizer_form() -> dict | None:
    with st.form("fertilizer_inputs"):
        st.subheader("Fertilizer prediction inputs")
        left, right = st.columns(2)
        with left:
            soil_ph = number("Soil pH", "fert_soil_ph", 0.0, 14.0, 0.1)
            soil_moisture = number("Soil moisture", "fert_soil_moisture", 0.0, 100.0, 0.1)
            organic_carbon = number("Organic carbon", "fert_organic_carbon", 0.0, 100.0, 0.1)
            electrical_conductivity = number("Electrical conductivity", "fert_ec", 0.0, 100.0, 0.01)
            nitrogen = number("Nitrogen level", "fert_nitrogen", 0.0, 500.0)
            phosphorus = number("Phosphorus level", "fert_phosphorus", 0.0, 500.0)
            potassium = number("Potassium level", "fert_potassium", 0.0, 500.0)
            temperature = number("Temperature (°C)", "fert_temperature", -10.0, 60.0, 0.1)
            humidity = number("Humidity (%)", "fert_humidity", 0.0, 100.0, 0.1)
            rainfall = number("Rainfall (mm)", "fert_rainfall", 0.0, 3000.0, 0.1)
        with right:
            soil_type = select("Soil type", ["Clay", "Loamy", "Sandy", "Silt"], "fert_soil_type")
            crop_type = select("Crop type", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"], "fert_crop_type")
            season = select("Season", ["Kharif", "Rabi", "Zaid"], "fert_season")
            growth_stage = select("Crop growth stage", ["Flowering", "Harvest", "Sowing", "Vegetative"], "fert_growth_stage")
            irrigation = select("Irrigation type", ["Canal", "Drip", "Rainfed", "Sprinkler"], "fert_irrigation")
            previous_crop = select("Previous crop", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"], "fert_previous_crop")
            region = select("Region", ["Central", "East", "North", "South", "West"], "fert_region")
            used_last_season = number("Fertilizer used last season (numeric quantity)", "fert_used_last", 0.0, 100000.0)
            yield_last_season = number("Yield last season (numeric quantity)", "fert_yield_last", 0.0, 100000.0, 0.1)
        submitted = st.form_submit_button("Generate Fertilizer Prediction", type="primary")
    values = [soil_ph, soil_moisture, organic_carbon, electrical_conductivity, nitrogen, phosphorus,
              potassium, temperature, humidity, rainfall, soil_type, crop_type, season, growth_stage,
              irrigation, previous_crop, region, used_last_season, yield_last_season]
    if not submitted:
        return None
    if any(value is None for value in values):
        st.error("Complete all fertilizer inputs before generating the prediction.")
        return None
    return {
        "_prediction_type": "fertilizer", "ph": soil_ph, "soil_moisture": soil_moisture,
        "organic_carbon": organic_carbon, "electrical_conductivity": electrical_conductivity,
        "nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium,
        "temperature": temperature, "humidity": humidity, "rainfall": rainfall,
        "soil_type": soil_type, "crop_type": crop_type, "season": season,
        "crop_growth_stage": growth_stage, "irrigation_type": irrigation,
        "previous_crop": previous_crop, "region": region,
        "fertilizer_used_last_season": used_last_season, "yield_last_season": yield_last_season,
    }


def yield_form() -> dict | None:
    with st.form("yield_inputs"):
        st.subheader("Yield prediction inputs")
        crop = select("Crop", ["Arecanut", "Arhar/Tur", "Bajra", "Banana", "Barley", "Cotton(lint)", "Maize", "Rice", "Sugarcane", "Wheat", "Potato"], "yield_crop")
        season = select("Season", ["Autumn     ", "Kharif     ", "Rabi       ", "Summer     ", "Whole Year ", "Winter     "], "yield_season")
        state = select("State", ["Andhra Pradesh", "Assam", "Bihar", "Gujarat", "Haryana", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"], "yield_state")
        left, right = st.columns(2)
        with left:
            crop_year = number("Crop year", "yield_year", 1900, 2100)
            area = number("Area (hectares)", "yield_area", 0.01, 100000000.0, 0.01)
            production = number("Production", "yield_production", 0.0, 1000000000.0, 0.1)
        with right:
            rainfall = number("Annual rainfall (mm)", "yield_rainfall", 0.0, 5000.0, 0.1)
            fertilizer = number("Fertilizer used", "yield_fertilizer", 0.0, 1000000000.0, 0.1)
            pesticide = number("Pesticide used", "yield_pesticide", 0.0, 1000000000.0, 0.1)
        submitted = st.form_submit_button("Generate Yield Prediction", type="primary")
    values = [crop, season, state, crop_year, area, production, rainfall, fertilizer, pesticide]
    if not submitted:
        return None
    if any(value is None for value in values):
        st.error("Complete all yield inputs before generating the prediction.")
        return None
    return {
        "_prediction_type": "yield", "crop": crop, "season": season, "state": state,
        "crop_year": crop_year, "area": area, "production": production,
        "rainfall": rainfall, "fertilizer": fertilizer, "pesticide": pesticide,
    }


def run_prediction(inputs: dict) -> None:
    prediction_type = inputs.pop("_prediction_type")
    with st.spinner("Generating prediction..."):
        generate_prediction(prediction_type, inputs)


def render_shap_explanation(prediction: dict | None) -> None:
    explanation = (prediction or {}).get("explanation", {})
    if explanation.get("type") != "shap":
        return
    factors = explanation.get("top_factors", [])
    with st.expander("Why did the model choose this result?", expanded=False):
        st.write("The chart shows which inputs had the strongest effect on this prediction.")
        if factors:
            chart = pd.DataFrame({
                "Feature": [item["label"] for item in factors],
                "Effect on result": [item["contribution"] for item in factors],
            }).set_index("Feature")
            st.bar_chart(chart)
            st.dataframe(
                [{
                    "Feature": item["label"],
                    "Effect": "Higher" if item["contribution"] >= 0 else "Lower",
                    "Strength": round(abs(item["contribution"]), 3),
                } for item in factors],
                width="stretch",
                hide_index=True,
            )
        st.caption(explanation.get("warning"))


def render_voice_playback() -> None:
    if st.session_state.get("voice_answer_audio"):
        autoplay = bool(st.session_state.get("voice_audio_autoplay"))
        st.audio(
            st.session_state.voice_answer_audio,
            format="audio/mp3",
            autoplay=autoplay,
        )
        st.session_state.voice_audio_autoplay = False


def main() -> None:
    st.set_page_config(page_title="Smart Agriculture Assistant", page_icon="🌾", layout="wide")
    with st.spinner("Preparing agricultural knowledge index..."):
        try:
            index_status = prepare_vector_index()
        except Exception as exc:
            index_status = {"error": f"{type(exc).__name__}: {exc}"}
    initialize_session()
    st.title("🌾 Smart Agriculture Decision Support System")
    active_prediction = get_prediction()
    prediction_ready = active_prediction is not None

    if prediction_ready:
        st.success(f"{active_prediction['type'].title()} prediction is active. Inputs are frozen for this interaction.")
        if st.button("Start New Interaction", type="primary"):
            clear_session()
            st.rerun()
    else:
        st.subheader("Choose a prediction type")
        prediction_type = st.radio(
            "What would you like to predict?", ["Crop", "Fertilizer", "Yield"],
            index=None, horizontal=True, key="prediction_type_choice",
        )
        inputs = None
        if prediction_type == "Crop":
            inputs = crop_form()
        elif prediction_type == "Fertilizer":
            inputs = fertilizer_form()
        elif prediction_type == "Yield":
            st.warning(
                "Model limitation: the current yield model uses reported production and "
                "production-per-area features. Treat it as a retrospective estimate, not "
                "a pre-season forecast, until the model is retrained without target leakage."
            )
            inputs = yield_form()
        if inputs is not None:
            run_prediction(inputs)
            st.rerun()

    with st.sidebar:
        st.header("Session")
        if prediction_ready:
            st.write(f"Prediction: `{active_prediction['type']}`")
            if st.button("Start New Interaction", width="stretch"):
                clear_session()
                st.rerun()
        st.divider()
        st.write(f"LLM via Groq: `{settings.groq_model}`")
        st.write(f"Embeddings: `{settings.embedding_model}`")
        st.write("Vector index: `FAISS`")
        if index_status.get("error"):
            st.error("Knowledge index could not be prepared.")
        else:
            st.write(f"Stored vectors: `{index_status['chunk_count']}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prediction_ready:
        render_shap_explanation(active_prediction)
    render_voice_playback()

    submission = st.chat_input(
        "Type or ask by voice about this prediction...",
        accept_audio=settings.voice_enabled,
        audio_sample_rate=16000,
        disabled=not prediction_ready,
        key="unified_chat_input",
    )
    if submission:
        question = submission if isinstance(submission, str) else submission.text.strip()
        recorded_audio = None if isinstance(submission, str) else submission.audio
        response_language = None
        if recorded_audio is not None:
            try:
                with st.spinner("Listening and detecting language..."):
                    transcription = transcribe_audio(
                        recorded_audio.getvalue(),
                        getattr(recorded_audio, "name", "farmer-question.wav"),
                        language_hint=None,
                    )
                    language = transcription["language"]
                    question = normalize_transcript(transcription["text"], language)
                    response_language = resolve_language(language)
            except VoiceError as exc:
                st.error(str(exc))
                return
        if not question:
            st.warning("No question was detected. Please type or record again.")
            return
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing query and context..."):
                result = answer_question(
                    question,
                    response_language=response_language.name if response_language else None,
                )
            st.markdown(result["answer"])
            if recorded_audio is not None and response_language is not None:
                try:
                    st.session_state.voice_answer_audio = synthesize_speech(
                        result["answer"], response_language.code
                    )
                    st.session_state.voice_audio_autoplay = True
                    st.audio(
                        st.session_state.voice_answer_audio,
                        format="audio/mp3",
                        autoplay=True,
                    )
                    st.session_state.voice_audio_autoplay = False
                except VoiceError as exc:
                    st.warning(f"The text answer is ready, but audio playback failed. {exc}")
            if getattr(settings, "debug", False):
                with st.expander("Technical diagnostics"):
                    st.json(result.get("trace", {}))
                if result.get("rag_sources"):
                    with st.expander("Retrieved knowledge chunks"):
                        for item in result["rag_sources"]:
                            label = item.get("source", "Unknown source")
                            page = f", page {item['page']}" if item.get("page") else ""
                            st.markdown(f"**Chunk {item['rank']}: {label}{page}**")
                            st.write(item["content"])
                            if item.get("source_url"):
                                st.link_button("Open official source", item["source_url"])


if __name__ == "__main__":
    main()
