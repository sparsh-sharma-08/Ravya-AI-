import json
import os
import unittest
from src.ingestion.ingest_jsonl import ingest_jsonl

class TestIngestJSONL(unittest.TestCase):

    def setUp(self):
        self.valid_jsonl_file = 'tests/test_valid.jsonl'
        self.invalid_jsonl_file = 'tests/test_invalid.jsonl'
        self.create_test_files()

    def create_test_files(self):
        valid_data = [
            json.dumps({
                "text": "This is a valid chunk.",
                "class": 8,
                "subject": "science",
                "chapter": 3,
                "language": "en",
                "textbook": "ncert",
                "tokens": 127
            }),
            json.dumps({
                "text": "Another valid chunk.",
                "class": 8,
                "subject": "science",
                "chapter": 3,
                "language": "en",
                "textbook": "ncert",
                "tokens": 130
            })
        ]
        invalid_data = [
            json.dumps({
                "text": "This chunk is missing a class.",
                "subject": "science",
                "chapter": 3,
                "language": "en",
                "textbook": "ncert",
                "tokens": 127
            }),
            json.dumps({
                "text": "This chunk has an invalid class type.",
                "class": "eight",
                "subject": "science",
                "chapter": 3,
                "language": "en",
                "textbook": "ncert",
                "tokens": 130
            })
        ]

        with open(self.valid_jsonl_file, 'w') as f:
            f.write('\n'.join(valid_data))

        with open(self.invalid_jsonl_file, 'w') as f:
            f.write('\n'.join(invalid_data))

    def test_valid_jsonl_ingestion(self):
        result = ingest_jsonl(self.valid_jsonl_file)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['class'], 8)
        self.assertEqual(result[0]['subject'], 'science')

    def test_invalid_jsonl_ingestion(self):
        with self.assertRaises(ValueError):
            ingest_jsonl(self.invalid_jsonl_file)

    def tearDown(self):
        os.remove(self.valid_jsonl_file)
        os.remove(self.invalid_jsonl_file)

if __name__ == '__main__':
    unittest.main()