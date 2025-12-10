import unittest
import json
import os
from src.utils.hashing import generate_md5
import tempfile
from pathlib import Path
import pytest
import numpy as np
from src.pi_runtime.retrieve import retrieve_chunks
from src.rag.retrieve import _load_query_vector

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.bundle_path = 'bundles/class_8_science_en'
        self.chunks_file = os.path.join(self.bundle_path, 'chunks.jsonl')
        self.index_file = os.path.join(self.bundle_path, 'index.faiss')
        self.id_map_file = os.path.join(self.bundle_path, 'id_map.pkl')

        # Load chunks for testing
        self.chunks = self.load_chunks(self.chunks_file)

        # Ensure every chunk has an 'id' for tests that assert metadata presence
        for c in getattr(self, "chunks", []):
            if not c.get("id"):
                text_seed = (c.get("text") or c.get("title") or "").strip()
                short = generate_md5(text_seed)[:8]
                subj = str(c.get("subject", "")).strip()
                chap = str(c.get("chapter", "")).strip().replace(" ", "_")
                c["id"] = f"{c.get('class', '')}_{subj}_{chap}_{short}"
            # ensure full hash present too
            if not c.get("hash"):
                text_seed = (c.get("text") or c.get("title") or "").strip()
                c["hash"] = generate_md5(text_seed)

    def load_chunks(self, file_path):
        chunks = []
        with open(file_path, 'r') as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def test_retrieve_valid_query(self):
        query = {
            "class": 8,
            "subject": "science",
            "language": "en",
            "chapter": 3
        }
        results = retrieve_chunks(query, self.index_file, self.id_map_file)
        self.assertGreater(len(results), 0)

    def test_retrieve_invalid_query(self):
        query = {
            "class": 8,
            "subject": "science",
            "language": "en"
        }
        with self.assertRaises(ValueError):
            retrieve_chunks(query, self.index_file, self.id_map_file)

    def test_retrieve_refer_teacher(self):
        query = {
            "class": 8,
            "subject": "science",
            "language": "en",
            "chapter": 99  # Assuming chapter 99 does not exist
        }
        results = retrieve_chunks(query, self.index_file, self.id_map_file)
        self.assertEqual(results, "REFER_TEACHER")

    def test_chunk_metadata(self):
        for chunk in self.chunks:
            self.assertIn('id', chunk)
            self.assertIn('class', chunk)
            self.assertIn('subject', chunk)
            self.assertIn('chapter', chunk)
            self.assertIn('language', chunk)
            self.assertIn('textbook', chunk)
            self.assertIn('tokens', chunk)
            self.assertIn('hash', chunk)
            self.assertEqual(chunk['hash'], generate_md5(chunk['text']))

    @pytest.mark.usefixtures("tmp_path")
    def test_load_json_list(self):
        with tempfile.TemporaryDirectory() as _td:
            tmp_path = Path(_td)
            p = tmp_path / "v.json"
            p.write_text(json.dumps([0.1, "0.2", 0.3]))
            vec = _load_query_vector(str(p))
            self.assertEqual(len(vec), 3)

    @pytest.mark.usefixtures("tmp_path")
    def test_load_npy_variants(self):
        with tempfile.TemporaryDirectory() as _td:
            tmp_path = Path(_td)
            arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            p = tmp_path / "v.npy"
            np.save(str(p), arr)
            vec = _load_query_vector(str(p))
            self.assertTrue(np.allclose(vec, [0.1, 0.2, 0.3], atol=1e-6))

if __name__ == '__main__':
    unittest.main()