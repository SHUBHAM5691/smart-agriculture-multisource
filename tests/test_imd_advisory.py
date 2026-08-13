import unittest
from unittest.mock import patch

from rag.imd_advisory import identify_state, retrieve_live_advisory, wants_live_advisory


class ImdAdvisoryTests(unittest.TestCase):
    def test_detects_live_request_and_state(self):
        self.assertTrue(wants_live_advisory("What is today's weather advisory for Tamil Nadu?"))
        self.assertEqual(identify_state("forecast for Tamil Nadu"), "Tamil Nadu")

    def test_missing_state_does_not_guess(self):
        docs, error = retrieve_live_advisory("latest agromet advisory", None)
        self.assertEqual(docs, [])
        self.assertIn("state name", error)

    @patch("rag.imd_advisory.fetch_current_state_advisory")
    def test_prediction_state_can_supply_location(self, fetch):
        fetch.return_value = [{"content": "Rice rainfall advisory", "source": "IMD", "live": True}]
        prediction = {"inputs": {"state": "Tamil Nadu"}}
        docs, error = retrieve_live_advisory("current rice advisory", prediction)
        self.assertIsNone(error)
        self.assertEqual(docs[0]["rank"], 1)
        fetch.assert_called_once_with("Tamil Nadu")


if __name__ == "__main__":
    unittest.main()
