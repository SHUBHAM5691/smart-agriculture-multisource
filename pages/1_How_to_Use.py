import streamlit as st


st.set_page_config(page_title="How to Use | Smart Agriculture", page_icon="📘", layout="wide")

st.title("📘 How to Use the Smart Agriculture Application")
st.write(
    "This page explains how to test crop, fertilizer and yield predictions using "
    "preset examples, manually entered field values, or an uploaded CSV file."
)

st.info(
    "Use the Prediction page for a single prediction or the Batch Testing page "
    "to upload multiple records."
)

st.header("1. Choose a prediction type")
st.markdown(
    """
- **Crop:** recommends a crop from N, P, K, temperature, humidity, soil pH and rainfall.
- **Fertilizer:** recommends a fertilizer class from soil, crop, climate and management inputs.
- **Yield:** provides a retrospective estimate from historical production-related inputs.
"""
)

st.header("2. Select how you want to test")
method_a, method_b, method_c = st.columns(3)
with method_a:
    st.subheader("Preset example")
    st.write("Choose a labelled example and click **Apply preset**. Review the populated values before predicting.")
with method_b:
    st.subheader("Manual inputs")
    st.write("Enter your own measurements directly in the form and click the relevant prediction button.")
with method_c:
    st.subheader("CSV upload")
    st.write("Open **Batch Testing**, download a template, add records, upload it and download the results.")

st.header("3. Preset examples")
st.dataframe(
    [
        {"Prediction": "Crop", "Preset": "Rice-like conditions", "Purpose": "Demonstrates warm, humid and high-rainfall inputs"},
        {"Prediction": "Crop", "Preset": "Maize-like conditions", "Purpose": "Demonstrates moderate-rainfall field inputs"},
        {"Prediction": "Fertilizer", "Preset": "Rice nutrient example", "Purpose": "Demonstrates the complete fertilizer feature set"},
        {"Prediction": "Yield", "Preset": "Rice retrospective example", "Purpose": "Demonstrates historical yield estimation"},
    ],
    hide_index=True,
    width="stretch",
)
st.warning(
    "These example values are only for testing the application. For a real prediction, "
    "enter measurements from your own soil, field and local weather conditions."
)

st.header("4. Understand the result")
st.markdown(
    """
1. Read the predicted crop, fertilizer class or yield estimate.
2. Open **Why did the model choose this result?** to view the SHAP explanation.
3. Ask a typed or voice follow-up question after a prediction is active.
4. Use **Start New Interaction** before running a different prediction.
"""
)

st.header("5. Input notes")
st.markdown(
    """
- Use the units displayed beside each field.
- Select a state under **Location for advisory** when you want state-specific documents or current IMD guidance. Sharing browser coordinates is optional and requires consent.
- Do not enter guessed measurements for real agricultural decisions.
- The yield model currently uses production-derived fields and should be treated as retrospective, not as a pre-season forecast.
- Retrieved guidance may come from national or regional publications. Check the displayed source and confirm regional applicability.
- Verify fertilizer and pesticide decisions using a soil test, current product label and local agricultural expert.
"""
)

st.header("6. Troubleshooting")
st.markdown(
    """
- **Prediction fails:** check that all required values and CSV columns are present.
- **AI response unavailable:** the local prediction remains valid as model output; retry the follow-up after a short wait.
- **Voice input fails:** allow microphone access or type the question.
- **CSV rejected:** download a fresh template and preserve its exact column names.
"""
)
