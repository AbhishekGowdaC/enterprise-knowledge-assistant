import unittest
from unittest.mock import patch, MagicMock

from app.retrieval.router import RetrievalRouter
from app.generation.answer import validate_citations


class TestRetrievalRouter(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.qdrant = MagicMock()

        self.question = (
            "How many annual leave days do employees "
            "with less than 3 years of service receive?"
        )

        self.chunk = {
            "id": "chunk-2",
            "source": "employee_handbook (1).pdf",
            "chunk_index": 2,
            "text": (
                "Employees with less than 3 years of service "
                "receive 20 working days of annual leave."
            ),
            "dense_score": 0.91,
        }

    def test_dense_retrieval(self):
        """Dense mode should retrieve and format source-aware context."""

        with patch(
            "app.retrieval.router.dense_search",
            return_value=[self.chunk],
        ) as dense_search:

            with patch(
                "app.retrieval.router.load_chunks"
            ) as load_chunks:

                router = RetrievalRouter(
                    self.model,
                    self.qdrant,
                    mode="dense",
                )

                results = router.retrieve_context(self.question)

                self.assertEqual(len(results), 1)
                self.assertIn("chunk: 2", results[0])
                self.assertIn("20 working days", results[0])

                dense_search.assert_called_once()
                load_chunks.assert_not_called()

    def test_hybrid_retrieval(self):
        """Hybrid mode should use BM25, dense, and RRF."""

        dense_results = [self.chunk]
        keyword_results = [self.chunk]
        exact_results = []

        with patch(
            "app.retrieval.router.load_chunks",
            return_value=[self.chunk],
        ), patch(
            "app.retrieval.router.BM25Okapi"
        ) as bm25_class, patch(
            "app.retrieval.router.dense_search",
            return_value=dense_results,
        ), patch(
            "app.retrieval.router.bm25_search",
            return_value=keyword_results,
        ) as bm25_search, patch(
            "app.retrieval.router.exact_match_search",
            return_value=exact_results,
        ), patch(
            "app.retrieval.router.reciprocal_rank_fusion",
            return_value=[self.chunk],
        ) as fusion:

            bm25_class.return_value = MagicMock()

            router = RetrievalRouter(
                self.model,
                self.qdrant,
                mode="hybrid",
            )

            results = router.retrieve_context(self.question)

            self.assertEqual(len(results), 1)
            self.assertIn("chunk: 2", results[0])

            bm25_search.assert_called_once()
            fusion.assert_called_once()

    def test_dense_low_similarity_returns_empty(self):
        """Dense retrieval should abstain below its similarity threshold."""

        low_score_chunk = {
            **self.chunk,
            "dense_score": 0.20,
        }

        with patch(
            "app.retrieval.router.dense_search",
            return_value=[low_score_chunk],
        ):
            router = RetrievalRouter(
                self.model,
                self.qdrant,
                mode="dense",
                min_similarity=0.45,
            )

            results = router.retrieve_context(self.question)

            self.assertEqual(results, [])

    def test_invalid_mode_raises_error(self):
        """Only dense and hybrid modes should be accepted."""

        with self.assertRaises(ValueError):
            RetrievalRouter(
                self.model,
                self.qdrant,
                mode="invalid",
            )

    def test_hybrid_without_chunks_raises_error(self):
        """Hybrid retrieval requires an indexed document collection."""

        with patch(
            "app.retrieval.router.load_chunks",
            return_value=[],
        ):
            with self.assertRaises(RuntimeError):
                RetrievalRouter(
                    self.model,
                    self.qdrant,
                    mode="hybrid",
                )

    def test_citation_validation_with_retrieved_context(self):
        """A citation must match a retrieved source and chunk."""

        context = [
            "[Source: employee_handbook (1).pdf, chunk: 2]\n"
            "Employees receive 20 working days of annual leave."
        ]

        valid_answer = (
            "Employees receive 20 working days of annual leave.\n"
            "[Source: employee_handbook (1).pdf, chunk: 2]"
        )

        invalid_answer = (
            "Employees receive 20 working days of annual leave.\n"
            "[Source: employee_handbook (1).pdf, chunk: 9]"
        )

        valid, _ = validate_citations(valid_answer, context)
        invalid, _ = validate_citations(invalid_answer, context)

        self.assertTrue(valid)
        self.assertFalse(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
