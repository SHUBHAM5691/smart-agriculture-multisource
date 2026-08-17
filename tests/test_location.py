import unittest
from unittest.mock import patch

from location.service import normalize_state, reverse_geocode


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return (
            b'{"display_name":"Nagpur, Maharashtra, India",'
            b'"address":{"state":"Maharashtra","state_district":"Nagpur"}}'
        )


class LocationTests(unittest.TestCase):
    def setUp(self):
        reverse_geocode.cache_clear()

    def test_reverse_geocode_extracts_state_and_district(self):
        with patch("location.service.urlopen", return_value=_Response()):
            result = reverse_geocode(21.1458, 79.0882)
        self.assertEqual(result["state"], "Maharashtra")
        self.assertEqual(result["district"], "Nagpur")

    def test_state_alias_is_normalized(self):
        self.assertEqual(
            normalize_state("National Capital Territory of Delhi", ["Delhi", "Punjab"]),
            "Delhi",
        )


if __name__ == "__main__":
    unittest.main()
