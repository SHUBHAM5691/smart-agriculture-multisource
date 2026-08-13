import unittest
from unittest.mock import patch

from rag.retriever import _filter_crop_mismatches, retrieve_with_diagnostics


class RetrieverFilteringTests(unittest.TestCase):
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
