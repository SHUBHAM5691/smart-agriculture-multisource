import unittest
from unittest.mock import patch

from rag.retriever import _filter_crop_mismatches, _filter_location_scope, retrieve_with_diagnostics


class RetrieverFilteringTests(unittest.TestCase):
    def test_no_location_excludes_state_specific_chunks(self):
        docs = [
            {"content": "General rice guidance", "state": None},
            {"content": "Tamil Nadu rice guidance", "state": "Tamil Nadu"},
        ]
        self.assertEqual(_filter_location_scope(docs, None), [docs[0]])

    def test_matching_location_keeps_national_and_state_chunks(self):
        docs = [
            {"content": "General rice guidance", "state": None},
            {"content": "Tamil Nadu rice guidance", "state": "Tamil Nadu"},
            {"content": "Punjab rice guidance", "state": "Punjab"},
        ]
        self.assertEqual(_filter_location_scope(docs, "Tamil Nadu"), docs[:2])
    def test_maize_query_retrieves_maize_knowledge(self):
        chunks = [{"content": "MAIZE Maize irrigation and nutrient guidance.", "source": "test"}]
        with patch("rag.retriever.load_chunks", return_value=chunks), patch(
            "rag.retriever.semantic_search", return_value=[(0, 0.8)]
        ):
            docs, error = retrieve_with_diagnostics(
                "maize cultivation irrigation nutrients pests guidance"
            )

        self.assertIsNone(error)
        self.assertTrue(docs)
        self.assertTrue(any("MAIZE" in doc["content"] for doc in docs))

    def test_maize_query_excludes_rice_only_chunk(self):
        docs = [
            {"content": "RICE Rice needs careful water management."},
            {"content": "MAIZE Maize needs good drainage."},
            {"content": "GENERAL FARM PLANNING Start with a soil test."},
            {"content": "Knowledge base for Rice, Wheat, and Maize."},
        ]

        result = _filter_crop_mismatches(docs, "share more knowledge about maize")

        self.assertNotIn(docs[0], result)
        self.assertIn(docs[1], result)
        self.assertIn(docs[2], result)
        self.assertIn(docs[3], result)


if __name__ == "__main__":
    unittest.main()
