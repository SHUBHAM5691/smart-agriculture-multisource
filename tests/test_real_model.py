import unittest

from prediction.real_model import RealCropModel


class RealCropModelTests(unittest.TestCase):
    def test_predict_returns_trained_model_output(self):
        model = RealCropModel()
        result = model.predict(
            {
                "nitrogen": 90,
                "phosphorus": 42,
                "potassium": 43,
                "temperature": 25,
                "humidity": 80,
                "ph": 6.5,
                "rainfall": 200,
            }
        )
        self.assertEqual(result["explanation"]["type"], "shap")
        self.assertGreater(len(result["explanation"]["top_factors"]), 0)

        self.assertIn("crop", result)
        self.assertIn("confidence", result)
        self.assertIn("model_name", result)
        self.assertEqual(result["model_version"], "v2-tuned")
        self.assertEqual(model.model_path.name, "best_model.joblib")
        self.assertGreaterEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
