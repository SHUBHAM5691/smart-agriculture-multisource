from pathlib import Path

import joblib
import pandas as pd
from prediction.shap_explainer import generate_shap_explanation

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

from prediction.preprocessing import preprocess_crop


class RealCropModel:
    model_name = "trained-crop-classifier"
    model_version = "v2-tuned"

    def __init__(self):
        self.model_path = MODELS_DIR / "crop" / "best_model.joblib"
        self.scaler_path = MODELS_DIR / "preprocessing" / "crop_rec_standard_scaler.joblib"
        self.label_encoder_path = MODELS_DIR / "preprocessing" / "crop_rec_target_label_encoder.joblib"
        self.feature_order_path = MODELS_DIR / "preprocessing" / "crop_rec_feature_order.joblib"
        self.encoder_paths = {
            "Rainfall_Category": MODELS_DIR / "preprocessing" / "crop_rec_features_Rainfall_Category_onehot_encoder.joblib",
            "Temperature_Category": MODELS_DIR / "preprocessing" / "crop_rec_features_Temperature_Category_onehot_encoder.joblib",
        }

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
        self.feature_order = joblib.load(self.feature_order_path)
        self.feature_encoders = {name: joblib.load(path) for name, path in self.encoder_paths.items()}
        self.categorical_cols = ["Rainfall_Category", "Temperature_Category"]
        self.numerical_cols = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
            "NPK_Sum",
            "NPK_Ratio_N",
            "NPK_Ratio_P",
            "NPK_Ratio_K",
        ]

    def _normalize_inputs(self, field_inputs: dict) -> dict:
        aliases = {
            "nitrogen": "N",
            "phosphorus": "P",
            "potassium": "K",
            "temperature": "temperature",
            "humidity": "humidity",
            "ph": "ph",
            "rainfall": "rainfall",
        }
        normalized = {}
        for source_key, target_key in aliases.items():
            if source_key in field_inputs:
                normalized[target_key] = float(field_inputs[source_key])
            elif target_key in field_inputs:
                normalized[target_key] = float(field_inputs[target_key])
        return normalized

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()
        for column in self.categorical_cols:
            if column not in encoded_df.columns:
                continue
            encoder = self.feature_encoders[column]
            transformed = encoder.transform(encoded_df[[column]])
            encoded_columns = pd.DataFrame(
                transformed,
                columns=encoder.get_feature_names_out([column]),
                index=encoded_df.index,
            )
            encoded_df = pd.concat([encoded_df.drop(columns=[column]), encoded_columns], axis=1)
        return encoded_df

    def prepare_features(self, field_inputs: dict) -> pd.DataFrame:
        normalized_inputs = self._normalize_inputs(field_inputs)
        df = preprocess_crop(normalized_inputs)
        df = self._apply_encoders(df)
        df = df.reindex(columns=self.feature_order)
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        return df

    def predict(self, field_inputs: dict) -> dict:
        df = self.prepare_features(field_inputs)
        prediction = int(self.model.predict(df)[0])
        label = str(self.label_encoder.inverse_transform([prediction])[0])

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)[0]
            decoded_labels = self.label_encoder.inverse_transform(self.model.classes_)
            class_probabilities = {
                str(decoded_label): round(float(prob), 3)
                for decoded_label, prob in zip(decoded_labels, probabilities)
            }
            confidence = round(float(max(class_probabilities.values())), 3)
        else:
            class_probabilities = {}
            confidence = None

        class_index = list(self.model.classes_).index(prediction)
        explanation = generate_shap_explanation(
            self.model, df, label, class_index=class_index, output_space="class probability"
        )
        return {
            "crop": label,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "explanation": explanation,
        }
