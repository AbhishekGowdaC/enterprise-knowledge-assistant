import unittest

from app.ingestion.chunking import chunk_records


class TestChunkRecords(unittest.TestCase):

    def test_chunk_indices_are_sequential_per_source(self):
        records = [
            {
                "source": "policy_a.pdf",
                "page": 1,
                "text": "A" * 1500,
            },
            {
                "source": "policy_b.pdf",
                "page": 1,
                "text": "B" * 1000,
            },
            {
                "source": "policy_a.pdf",
                "page": 2,
                "text": "C" * 900,
            },
        ]

        chunks = chunk_records(records, 800, 120)

        policy_a = [
            c["chunk_index"] for c in chunks
            if c["source"] == "policy_a.pdf"
        ]
        policy_b = [
            c["chunk_index"] for c in chunks
            if c["source"] == "policy_b.pdf"
        ]

        self.assertEqual(policy_a, [0, 1, 2, 3, 4])
        self.assertEqual(policy_b, [0, 1])

    def test_chunk_overlap_is_preserved(self):
        records = [{
            "source": "policy.pdf",
            "page": 1,
            "text": "A" * 1500,
        }]

        chunks = chunk_records(records, 800, 120)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            chunks[0]["text"][-120:],
            chunks[1]["text"][:120],
        )

    def test_chunk_metadata_is_preserved(self):
        records = [{
            "source": "policy.pdf",
            "page": 3,
            "file_type": "pdf",
            "text": "Sample policy text",
        }]

        chunks = chunk_records(records)

        self.assertEqual(chunks[0]["source"], "policy.pdf")
        self.assertEqual(chunks[0]["page"], 3)
        self.assertEqual(chunks[0]["file_type"], "pdf")
        self.assertEqual(chunks[0]["chunk_index"], 0)

    def test_empty_text_returns_no_chunks(self):
        records = [{
            "source": "empty.pdf",
            "page": 1,
            "text": "",
        }]

        self.assertEqual(chunk_records(records), [])

    def test_invalid_chunk_size_raises_error(self):
        with self.assertRaises(ValueError):
            chunk_records([], chunk_size=0)

    def test_invalid_overlap_raises_error(self):
        with self.assertRaises(ValueError):
            chunk_records(
                [],
                chunk_size=800,
                chunk_overlap=800,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
