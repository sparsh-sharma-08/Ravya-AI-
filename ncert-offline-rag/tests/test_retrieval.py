import unittest
import json
import os
from src.pi_runtime.retrieve import retrieve_chunks
from src.utils.hashing import generate_md5

class TestRetrieval(unittest.TestCase):

    def setUp(self):
        self.bundle_path = 'bundles/class_8_science_en'
        self.chunks_file = os.path.join(self.bundle_path, 'chunks.jsonl')
        self.index_file = os.path.join(self.bundle_path, 'index.faiss')
        self.id_map_file = os.path.join(self.bundle_path, 'id_map.pkl')

        # Load chunks for testing
        self.chunks = self.load_chunks(self.chunks_file)

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

if __name__ == '__main__':
    unittest.main()