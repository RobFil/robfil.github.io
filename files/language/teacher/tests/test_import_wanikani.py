import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_wanikani.py"
SPEC = importlib.util.spec_from_file_location("import_wanikani", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ImportWaniKaniTest(unittest.TestCase):
    def test_transforms_vocabulary_and_skips_kanji(self):
        page = {
            "data": [
                {
                    "id": 123,
                    "object": "vocabulary",
                    "data_updated_at": "2026-01-02T03:04:05Z",
                    "data": {
                        "characters": "仕様",
                        "level": 20,
                        "readings": [
                            {"reading": "しよう", "accepted_answer": True, "primary": True}
                        ],
                        "meanings": [
                            {"meaning": "Specification", "accepted_answer": True, "primary": True}
                        ],
                        "parts_of_speech": ["noun"],
                        "document_url": "https://www.wanikani.com/vocabulary/仕様",
                    },
                },
                {"id": 124, "object": "kanji", "data": {"characters": "仕"}},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wanikani.jsonl"
            count = MODULE.write_records(iter([page]), output)
            record = json.loads(output.read_text(encoding="utf-8"))
            metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(record["lemma"], "仕様")
        self.assertEqual(record["reading"], "しよう")
        self.assertEqual(record["translations"]["en"][0]["text"], "Specification")
        self.assertEqual(record["wanikani_level"], 20)
        self.assertEqual(metadata["record_count"], 1)


if __name__ == "__main__":
    unittest.main()
