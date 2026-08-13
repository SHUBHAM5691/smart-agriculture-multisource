from __future__ import annotations

import numpy as np
import pandas as pd
import shap


FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "ph": "Soil pH",
    "Soil_pH": "Soil pH",
    "Annual_Rainfall": "Annual rainfall",
    "Crop_Year": "Crop year",
    "Production_per_Area": "Production per area",
}

CATEGORY_PREFIXES = (
    "Rainfall_Category", "Temperature_Category", "Soil_Type", "Crop_Type",
    "Season", "Crop_Growth_Stage", "Irrigation_Type", "Previous_Crop",
    "Region", "Crop", "State",
)


def readable_feature_name(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix in CATEGORY_PREFIXES:
        marker = f"{prefix}_"
        if name.startswith(marker):
            category = prefix.replace("_", " ")
            value = name[len(marker):].strip()
            return f"{category}: {value}"
    return name.replace("_", " ").strip().title()


def generate_shap_explanation(
    model,
    features: pd.DataFrame,
    subject: str,
    class_index: int | None = None,
    output_space: str = "model output",
) -> dict:
    if len(features) != 1:
        raise ValueError("SHAP explanation requires exactly one transformed input row")

    explanation = shap.TreeExplainer(model)(features)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        selected_class = class_index if class_index is not None else 0
        contributions = values[0, :, selected_class]
        if base_values.ndim >= 2:
            base_value = float(base_values[0, selected_class])
        elif base_values.ndim == 1 and len(base_values) > selected_class:
            base_value = float(base_values[selected_class])
        else:
            base_value = float(base_values.reshape(-1)[0])
    elif values.ndim == 2:
        contributions = values[0]
        base_value = float(base_values.reshape(-1)[0])
    else:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    ranked_indices = np.argsort(np.abs(contributions))[::-1]
    top_factors = []
    for index in ranked_indices[:8]:
        contribution = float(contributions[index])
        if not np.isfinite(contribution):
            continue
        feature_name = str(features.columns[index])
        raw_value = features.iloc[0, index]
        value = float(raw_value) if isinstance(raw_value, (int, float, np.number)) else str(raw_value)
        top_factors.append({
            "feature": feature_name,
            "label": readable_feature_name(feature_name),
            "value": value,
            "contribution": round(contribution, 6),
            "direction": "increases" if contribution >= 0 else "decreases",
        })

    labels = ", ".join(item["label"] for item in top_factors[:3])
    return {
        "type": "shap",
        "method": "TreeExplainer",
        "output_space": output_space,
        "base_value": round(base_value, 6),
        "top_factors": top_factors,
        "summary": f"SHAP values for {subject} have the largest magnitudes for {labels}.",
        "warning": "This explains the model's calculation, not guaranteed real-world cause and effect.",
    }
