import json
from unittest.mock import MagicMock, patch

import numpy as np

import rag.knowledge_base as knowledge_base


def test_prepare_vector_index_reports_loaded_index():
    fake_index = MagicMock(ntotal=2416)
    with patch.object(knowledge_base, "get_vector_index", return_value=fake_index):
        result = knowledge_base.prepare_vector_index()
    assert result["chunk_count"] == 2416


def test_saved_index_is_loaded_when_fingerprint_matches(tmp_path):
    fake_index = MagicMock(ntotal=2)
    metadata_path = tmp_path / "metadata.json"
    index_path = tmp_path / "knowledge.faiss"
    metadata_path.write_text(json.dumps({"fingerprint": "same", "chunk_count": 2}))
    index_path.write_bytes(b"index")
    fake_faiss = MagicMock()
    fake_faiss.read_index.return_value = fake_index

    knowledge_base.get_vector_index.cache_clear()
    with (
        patch.object(knowledge_base, "FAISS_INDEX_PATH", index_path),
        patch.object(knowledge_base, "INDEX_METADATA_PATH", metadata_path),
        patch.object(knowledge_base, "_corpus_fingerprint", return_value="same"),
        patch.object(knowledge_base, "load_chunks", return_value=[{}, {}]),
        patch.dict("sys.modules", {"faiss": fake_faiss}),
    ):
        assert knowledge_base.get_vector_index() is fake_index
        fake_faiss.read_index.assert_called_once_with(str(index_path))
    knowledge_base.get_vector_index.cache_clear()
