import os
import json
import unittest

class TestBundleExport(unittest.TestCase):
    def setUp(self):
        self.bundle_path = 'bundles/class_8_science_en'
        self.manifest_file = os.path.join(self.bundle_path, 'manifest.json')
        self.index_file = os.path.join(self.bundle_path, 'index.faiss')
        self.embeddings_file = os.path.join(self.bundle_path, 'embeddings.bin')
        self.chunks_file = os.path.join(self.bundle_path, 'chunks.jsonl')
        self.id_map_file = os.path.join(self.bundle_path, 'id_map.pkl')
        self.model_file = os.path.join(self.bundle_path, 'model.json')
        self.version_file = os.path.join(self.bundle_path, 'version.txt')

    def test_manifest_exists(self):
        self.assertTrue(os.path.isfile(self.manifest_file), "Manifest file does not exist.")

    def test_index_exists(self):
        self.assertTrue(os.path.isfile(self.index_file), "FAISS index file does not exist.")

    def test_embeddings_exists(self):
        self.assertTrue(os.path.isfile(self.embeddings_file), "Embeddings file does not exist.")

    def test_chunks_exists(self):
        self.assertTrue(os.path.isfile(self.chunks_file), "Chunks file does not exist.")

    def test_id_map_exists(self):
        self.assertTrue(os.path.isfile(self.id_map_file), "ID map file does not exist.")

    def test_model_exists(self):
        self.assertTrue(os.path.isfile(self.model_file), "Model file does not exist.")

    def test_version_exists(self):
        self.assertTrue(os.path.isfile(self.version_file), "Version file does not exist.")

    def test_manifest_content(self):
        with open(self.manifest_file, 'r') as f:
            manifest = json.load(f)
            self.assertIn('class', manifest)
            self.assertIn('subject', manifest)
            self.assertIn('chapter', manifest)
            self.assertIn('language', manifest)
            self.assertIn('embedding_model', manifest)
            self.assertIn('embedding_dim', manifest)
            self.assertIn('chunk_count', manifest)
            self.assertIn('chunk_strategy', manifest)
            self.assertIn('created_at', manifest)
            self.assertIn('version', manifest)
            self.assertIn('hash_strategy', manifest)

if __name__ == '__main__':
    unittest.main()