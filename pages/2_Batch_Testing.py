import pandas as pd
import streamlit as st

from prediction.fertilizer_subprocess import predict_fertilizer
from prediction.real_model import RealCropModel
from prediction.real_predictors import RealYieldModel


st.set_page_config(page_title="Batch Testing | Smart Agriculture", page_icon="📄", layout="wide")

CROP_TEMPLATE = pd.DataFrame([
    {"nitrogen": 90, "phosphorus": 42, "potassium": 43, "temperature": 25,
     "humidity": 80, "ph": 6.5, "rainfall": 200},
    {"nitrogen": 78, "phosphorus": 45, "potassium": 20, "temperature": 24,
     "humidity": 65, "ph": 6.4, "rainfall": 85},
])

FERTILIZER_TEMPLATE = pd.DataFrame([{
    "ph": 6.2, "soil_moisture": 35, "organic_carbon": 1.0,
    "electrical_conductivity": 0.2, "nitrogen": 80, "phosphorus": 40,
    "potassium": 50, "temperature": 28, "humidity": 70, "rainfall": 180,
    "soil_type": "Clay", "crop_type": "Rice", "season": "Kharif",
    "crop_growth_stage": "Vegetative", "irrigation_type": "Canal",
    "previous_crop": "Wheat", "region": "South",
    "fertilizer_used_last_season": 100, "yield_last_season": 3.0,
}])

YIELD_TEMPLATE = pd.DataFrame([{
    "crop": "Rice", "season": "Kharif", "state": "Tamil Nadu",
    "crop_year": 2024, "area": 1.0, "production": 1.0,
    "rainfall": 1200, "fertilizer": 100, "pesticide": 10,
}])

TEMPLATES = {
    "Crop": CROP_TEMPLATE,
    "Fertilizer": FERTILIZER_TEMPLATE,
    "Yield": YIELD_TEMPLATE,
}

NUMERIC_COLUMNS = {
    "Crop": list(CROP_TEMPLATE.columns),
    "Fertilizer": [
        "ph", "soil_moisture", "organic_carbon", "electrical_conductivity",
        "nitrogen", "phosphorus", "potassium", "temperature", "humidity",
        "rainfall", "fertilizer_used_last_season", "yield_last_season",
    ],
    "Yield": ["crop_year", "area", "production", "rainfall", "fertilizer", "pesticide"],
}


def validate_frame(frame: pd.DataFrame, prediction_type: str) -> pd.DataFrame:
    required = list(TEMPLATES[prediction_type].columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The uploaded file contains no records.")
    if len(frame) > 50:
        raise ValueError("Upload at most 50 records per batch.")
    cleaned = frame[required].copy()
    for column in NUMERIC_COLUMNS[prediction_type]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    if cleaned.isna().any().any():
        raise ValueError("Blank values are not allowed in required columns.")
    if prediction_type == "Crop":
        if ((cleaned[["nitrogen", "phosphorus", "potassium", "rainfall"]] < 0).any().any()
                or (cleaned["humidity"] < 0).any() or (cleaned["humidity"] > 100).any()
                or (cleaned["ph"] < 0).any() or (cleaned["ph"] > 14).any()):
            raise ValueError("Crop values are outside the accepted physical ranges.")
    if prediction_type == "Yield" and (cleaned["area"] <= 0).any():
        raise ValueError("Yield area must be greater than zero.")
    return cleaned


def run_batch(frame: pd.DataFrame, prediction_type: str) -> pd.DataFrame:
    results = []
    crop_model = RealCropModel() if prediction_type == "Crop" else None
    yield_model = RealYieldModel() if prediction_type == "Yield" else None
    for row_number, row in frame.iterrows():
        inputs = row.to_dict()
        try:
            if prediction_type == "Crop":
                output = crop_model.predict(inputs)
                result = output.get("crop")
                confidence = output.get("confidence")
            elif prediction_type == "Fertilizer":
                output = predict_fertilizer(inputs)
                result = output.get("display_name") or output.get("prediction")
                confidence = output.get("confidence")
            else:
                inputs["season"] = str(inputs["season"]).strip().ljust(11)
                output = yield_model.predict(inputs)
                result = output.get("prediction")
                confidence = None
            results.append({"row": row_number + 1, "prediction": result, "confidence": confidence, "status": "success"})
        except Exception as exc:
            results.append({"row": row_number + 1, "prediction": None, "confidence": None,
                            "status": f"error: {type(exc).__name__}: {exc}"})
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(results)], axis=1)


st.title("📄 Upload Custom Inputs and Test the System")
st.write("Download a template, preserve its column names, add up to 50 records and upload the completed CSV.")

prediction_type = st.selectbox("Prediction type", list(TEMPLATES))
template = TEMPLATES[prediction_type]
st.download_button(
    f"Download {prediction_type} CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name=f"{prediction_type.lower()}_prediction_template.csv",
    mime="text/csv",
)

with st.expander("Required columns", expanded=False):
    st.code(", ".join(template.columns), language=None)
    st.dataframe(template, hide_index=True, width="stretch")

uploaded = st.file_uploader("Upload completed CSV", type=["csv"])
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        validated = validate_frame(raw, prediction_type)
        st.subheader("Validated input preview")
        st.dataframe(validated, hide_index=True, width="stretch")
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Running predictions..."):
                output = run_batch(validated, prediction_type)
            st.subheader("Batch results")
            summary_columns = ["row", "prediction", "confidence", "status"]
            summary = output[summary_columns].copy()
            summary["confidence"] = summary["confidence"].apply(
                lambda value: "—" if pd.isna(value) else f"{float(value):.1%}"
            )
            st.dataframe(summary, hide_index=True, width="stretch")
            successes = int((output["status"] == "success").sum())
            st.success(f"Completed {successes} of {len(output)} records successfully.")
            with st.expander("View input values with results"):
                st.dataframe(output, hide_index=True, width="stretch")
            st.download_button(
                "Download results as CSV (optional)",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name=f"{prediction_type.lower()}_prediction_results.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"The CSV could not be accepted: {exc}")

st.caption("Uploaded files are processed for the current session and are not added to the agricultural knowledge index.")
