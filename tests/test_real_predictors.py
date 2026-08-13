import unittest

from prediction.real_predictors import RealFertilizerModel, RealYieldModel, fertilizer_display_name


class RealPredictorTests(unittest.TestCase):
    def test_fertilizer_abbreviations_are_expanded_correctly(self):
        self.assertEqual(fertilizer_display_name("MOP"), "MOP (Muriate of Potash)")
        self.assertEqual(fertilizer_display_name("MAP"), "MAP (Monoammonium Phosphate)")

    def test_fertilizer_model_returns_prediction(self):
        model = RealFertilizerModel()
        result = model.predict(
            {
                "Soil_pH": 6.2,
                "Soil_Moisture": 35.0,
                "Organic_Carbon": 1.0,
                "Electrical_Conductivity": 0.2,
                "Nitrogen_Level": 80,
                "Phosphorus_Level": 40,
                "Potassium_Level": 50,
                "Temperature": 28,
                "Humidity": 70,
                "Rainfall": 180,
                "Soil_Type": "Sandy",
                "Crop_Type": "Maize",
                "Season": "Kharif",
                "Crop_Growth_Stage": "Vegetative",
                "Irrigation_Type": "Drip",
                "Previous_Crop": "Wheat",
                "Region": "North",
                "Fertilizer_Used_Last_Season": 100,
                "Yield_Last_Season": 3.0,
            }
        )
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["model_version"], "v2-tuned")
        self.assertEqual(model.model.__class__.__name__, "LGBMClassifier")
        self.assertEqual(result["explanation"]["type"], "shap")
        self.assertGreater(len(result["explanation"]["top_factors"]), 0)

    def test_yield_model_returns_prediction(self):
        model = RealYieldModel()
        result = model.predict(
            {
                "Crop_Year": 2020,
                "Area": 1000,
                "Production": 5000,
                "Annual_Rainfall": 1200,
                "Fertilizer": 200,
                "Pesticide": 50,
                "Production_per_Area": 5,
                "Crop_Arecanut": 0,
                "Crop_Arhar/Tur": 0,
                "Crop_Bajra": 0,
                "Crop_Banana": 0,
                "Crop_Barley": 0,
                "Crop_Black pepper": 0,
                "Crop_Cardamom": 0,
                "Crop_Cashewnut": 0,
                "Crop_Coconut ": 0,
                "Crop_Coriander": 0,
                "Crop_Cotton(lint)": 0,
                "Crop_Cowpea(Lobia)": 0,
                "Crop_Cumin": 0,
                "Crop_Garlic": 0,
                "Crop_Ginger": 0,
                "Crop_Groundnut": 0,
                "Crop_Horse-gram": 0,
                "Crop_Jowar": 0,
                "Crop_Khesari": 0,
                "Crop_Linseed": 0,
                "Crop_Maize": 1,
                "Crop_Masoor": 0,
                "Crop_Mesta": 0,
                "Crop_Moong(Green Gram)": 0,
                "Crop_Niger seed": 0,
                "Crop_Oats": 0,
                "Crop_Onion": 0,
                "Crop_Pearl millet": 0,
                "Crop_Peas & beans (Pulses)": 0,
                "Crop_Potato": 0,
                "Crop_Ragi": 0,
                "Crop_Rapeseed &Mustard": 0,
                "Crop_Rice": 0,
                "Crop_Safflower": 0,
                "Crop_Sesamum": 0,
                "Crop_Small millets": 0,
                "Crop_Soyabean": 0,
                "Crop_Sunflower": 0,
                "Crop_Tapioca": 0,
                "Crop_Tobacco": 0,
                "Crop_Turmeric": 0,
                "Crop_Urad": 0,
                "Crop_Wheat": 0,
                "State_Andhra Pradesh": 0,
                "State_Arunachal Pradesh": 0,
                "State_Assam": 0,
                "State_Bihar": 0,
                "State_Chhattisgarh": 0,
                "State_Goa": 0,
                "State_Gujarat": 0,
                "State_Haryana": 0,
                "State_Himachal Pradesh": 0,
                "State_Jammu and Kashmir": 0,
                "State_Jharkhand": 0,
                "State_Karnataka": 0,
                "State_Kerala": 0,
                "State_Madhya Pradesh": 0,
                "State_Maharashtra": 0,
                "State_Manipur": 0,
                "State_Meghalaya": 0,
                "State_Mizoram": 0,
                "State_Nagaland": 0,
                "State_Odisha": 0,
                "State_Puducherry": 0,
                "State_Punjab": 0,
                "State_Sikkim": 0,
                "State_Tamil Nadu": 0,
                "State_Telangana": 0,
                "State_Tripura": 0,
                "State_Uttar Pradesh": 0,
                "State_Uttarakhand": 0,
                "State_West Bengal": 0,
                "Rainfall_Category_Extreme": 0,
                "Rainfall_Category_High": 0,
                "Rainfall_Category_Low": 0,
                "Rainfall_Category_Moderate": 1,
                "Rainfall_Category_Torrential": 0,
                "Rainfall_Category_Very High": 0,
                "Rainfall_Category_Very Low": 0,
            },
        )
        self.assertIn("prediction", result)
        self.assertIn("unit", result)
        self.assertEqual(result["model_version"], "v2-tuned")
        self.assertEqual(model.model.__class__.__name__, "XGBRegressor")
        self.assertEqual(result["explanation"]["type"], "shap")
        self.assertGreater(len(result["explanation"]["top_factors"]), 0)


if __name__ == "__main__":
    unittest.main()
