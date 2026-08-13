import unittest

from utils.json_tools import extract_json_object


class JsonToolsTests(unittest.TestCase):
    def test_accepts_valid_object_with_extra_trailing_brace(self):
        text = '{"needs_rag":true,"retrieval_query":"rice cultivation"}}'

        result = extract_json_object(text)

        self.assertTrue(result["needs_rag"])
        self.assertEqual(result["retrieval_query"], "rice cultivation")

    def test_accepts_json_inside_model_commentary(self):
        result = extract_json_object('Result: {"needs_rag": false} done')

        self.assertEqual(result, {"needs_rag": False})


if __name__ == "__main__":
    unittest.main()
