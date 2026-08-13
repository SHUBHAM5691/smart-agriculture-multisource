from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

from prediction.preprocessing import preprocess_fertilizer
from prediction.shap_explainer import generate_shap_explanation

FERTILIZER_NAMES = {
    "MOP": "Muriate of Potash",
    "MAP": "Monoammonium Phosphate",
    "DAP": "Diammonium Phosphate",
    "Urea": "Urea",
}


def fertilizer_display_name(label: str | None) -> str | None:
    if label is None:
        return None
    expanded = FERTILIZER_NAMES.get(str(label))
    return f"{label} ({expanded})" if expanded else str(label)


class RealFertilizerModel:
    def __init__(self):
        preprocessing_dir = MODELS_DIR / "preprocessing"
        self.model = joblib.load(MODELS_DIR / "fertilizer" / "best_model.joblib")
        self.scaler = joblib.load(preprocessing_dir / "fertilizer_standard_scaler.joblib")
        self.label_encoder = joblib.load(preprocessing_dir / "fertilizer_target_label_encoder.joblib")
        self.feature_order = joblib.load(preprocessing_dir / "fertilizer_feature_order.joblib")
        self.categorical_cols = [
            "Soil_Type", "Crop_Type", "Season", "Crop_Growth_Stage",
            "Irrigation_Type", "Previous_Crop", "Region",
        ]
        self.feature_encoders = {
            column: joblib.load(
                preprocessing_dir / f"fertilizer_features_{column}_onehot_encoder.joblib"
            )
            for column in self.categorical_cols
        }
        self.numerical_cols = [
            "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level", "Temperature",
            "Humidity", "Rainfall", "Fertilizer_Used_Last_Season", "Yield_Last_Season",
            "NPK_Sum", "NPK_Ratio_N", "NPK_Ratio_P", "NPK_Ratio_K",
        ]

    def _prepare_features(self, inputs: dict) -> pd.DataFrame:
        features = preprocess_fertilizer(inputs)
        for column in self.categorical_cols:
            encoder = self.feature_encoders[column]
            encoded = pd.DataFrame(
                encoder.transform(features[[column]]),
                columns=encoder.get_feature_names_out([column]),
                index=features.index,
            )
            features = pd.concat([features.drop(columns=[column]), encoded], axis=1)
        features = features.reindex(columns=self.feature_order)
        features[self.numerical_cols] = self.scaler.transform(
            features[self.numerical_cols]
        )
        return features.drop(columns=["Recommended_Fertilizer"], errors="ignore")

    def predict(self, field_inputs: dict) -> dict:
        fertilizer_inputs = {
            "Soil_pH": field_inputs.get("ph", 6.5),
            "Soil_Moisture": field_inputs.get("soil_moisture", 35.0),
            "Organic_Carbon": field_inputs.get("organic_carbon", 1.0),
            "Electrical_Conductivity": field_inputs.get("electrical_conductivity", 0.2),
            "Nitrogen_Level": field_inputs.get("nitrogen", 90.0),
            "Phosphorus_Level": field_inputs.get("phosphorus", 42.0),
            "Potassium_Level": field_inputs.get("potassium", 43.0),
            "Temperature": field_inputs.get("temperature", 25.0),
            "Humidity": field_inputs.get("humidity", 80.0),
            "Rainfall": field_inputs.get("rainfall", 200.0),
            "Soil_Type": field_inputs.get("soil_type", "Sandy"),
            "Crop_Type": field_inputs.get("crop_type", "Maize"),
            "Season": field_inputs.get("season", "Kharif"),
            "Crop_Growth_Stage": field_inputs.get("crop_growth_stage", "Vegetative"),
            "Irrigation_Type": field_inputs.get("irrigation_type", "Drip"),
            "Previous_Crop": field_inputs.get("previous_crop", "Wheat"),
            "Region": field_inputs.get("region", "North"),
            "Fertilizer_Used_Last_Season": field_inputs.get("fertilizer_used_last_season", 100.0),
            "Yield_Last_Season": field_inputs.get("yield_last_season", 3.0),
        }
        features = self._prepare_features(fertilizer_inputs)
        encoded_prediction = self.model.predict(features)[0]
        label = str(self.label_encoder.inverse_transform([encoded_prediction])[0])
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        class_index = list(self.model.classes_).index(encoded_prediction)
        explanation = generate_shap_explanation(
            self.model,
            features,
            label,
            class_index=class_index,
            output_space="class score",
        )
        return {
            "prediction": label,
            "display_name": fertilizer_display_name(label),
            "confidence": confidence,
            "model_name": "trained-fertilizer-classifier",
            "model_version": "v2-tuned",
            "explanation": explanation,
        }


class RealYieldModel:
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "yield" / "crop_yield_model.joblib")
        self.feature_order = joblib.load(MODELS_DIR / "preprocessing" / "yield_feature_order.joblib")

    def _prepare_features(self, input_data: dict) -> pd.DataFrame:
        features = pd.DataFrame([input_data]).drop(columns=["Yield"], errors="ignore")
        return features.reindex(columns=self.feature_order).astype(float)

    def _predict(self, features: pd.DataFrame) -> float:
        predicted_log = np.asarray(self.model.predict(features), dtype=float)
        return float(np.maximum(np.expm1(predicted_log), 0)[0])

    def predict(self, input_data: dict) -> dict:
        if "crop_year" not in input_data:
            processed_features = dict(input_data)
            for feature in self.feature_order:
                processed_features.setdefault(feature, 0.0)
            features = self._prepare_features(processed_features)
            prediction = self._predict(features)
            explanation = generate_shap_explanation(
                self.model,
                features,
                "yield estimate",
                output_space="log1p yield",
            )
            return {
                "prediction": prediction,
                "unit": "dataset yield units",
                "input_shape": list(features.shape),
                "model_name": "trained-yield-regressor",
                "model_version": "v2-tuned",
                "explanation": explanation,
            }

        raw_features = {
            "Crop_Year": input_data["crop_year"],
            "Area": input_data["area"],
            "Production": input_data["production"],
            "Annual_Rainfall": input_data["rainfall"],
            "Fertilizer": input_data["fertilizer"],
            "Pesticide": input_data["pesticide"],
            "Production_per_Area": input_data["production"] / input_data["area"],
        }
        raw_features.update({feature: 0.0 for feature in self.feature_order if feature not in raw_features})
        raw_features[f"Crop_{input_data['crop']}"] = 1.0
        raw_features[f"Season_{input_data['season']}"] = 1.0
        raw_features[f"State_{input_data['state']}"] = 1.0
        rainfall = input_data["rainfall"]
        if rainfall < 500:
            rainfall_category = "Very Low"
        elif rainfall < 1000:
            rainfall_category = "Low"
        elif rainfall < 1500:
            rainfall_category = "Moderate"
        elif rainfall < 2000:
            rainfall_category = "High"
        elif rainfall < 2500:
            rainfall_category = "Very High"
        elif rainfall < 3000:
            rainfall_category = "Extreme"
        else:
            rainfall_category = "Torrential"
        raw_features[f"Rainfall_Category_{rainfall_category}"] = 1.0
        features = self._prepare_features(raw_features)
        prediction = self._predict(features)
        explanation = generate_shap_explanation(
            self.model,
            features,
            "yield estimate",
            output_space="log1p yield",
        )
        return {
            "prediction": prediction,
            "unit": "dataset yield units",
            "input_shape": list(features.shape),
            "model_name": "trained-yield-regressor",
            "model_version": "v2-tuned",
            "explanation": explanation,
        }
