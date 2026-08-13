import unittest

from prediction.fertilizer_subprocess import predict_fertilizer


class FertilizerSubprocessTests(unittest.TestCase):
    def test_real_prediction_completes_in_isolated_process(self):
        result = predict_fertilizer({
            "ph": 6.2, "soil_moisture": 35.0, "organic_carbon": 1.0,
            "electrical_conductivity": 0.2, "nitrogen": 80.0,
            "phosphorus": 40.0, "potassium": 50.0, "temperature": 28.0,
            "humidity": 70.0, "rainfall": 180.0, "soil_type": "Clay",
            "crop_type": "Cotton", "season": "Kharif",
            "crop_growth_stage": "Flowering", "irrigation_type": "Canal",
            "previous_crop": "Cotton", "region": "Central",
            "fertilizer_used_last_season": 100.0, "yield_last_season": 3.0,
        })
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["explanation"]["type"], "shap")
