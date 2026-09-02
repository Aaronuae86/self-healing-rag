"""Lightweight API contract tests that do not instantiate local ML models."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from src.api import QueryRequest, _query_response, create_app
from src.rag import RecoveryAction, RetrievalFailure


class FakeWorkflow:
    def run(self, query: str) -> dict:
        return {
            "final_answer": f"Answer to: {query}",
            "failure_type": RetrievalFailure.HEALTHY,
            "recovery_action": RecoveryAction.PROCEED_TO_GENERATION,
            "retry_count": 0,
            "path": ["RETRIEVE", "RERANK", "DIAGNOSTICS", "HEALTHY", "GENERATE"],
            "rewritten_query": None,
            "classification_reasons": ("evidence is sufficient",),
        }


class ApiContractTests(unittest.TestCase):
    def test_query_validation_rejects_empty_and_whitespace_only_values(self) -> None:
        for query in ("", "   "):
            with self.subTest(query=query), self.assertRaises(ValidationError):
                QueryRequest(query=query)

    def test_query_validation_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(QueryRequest(query="  What is RAG?  ").query, "What is RAG?")

    def test_response_maps_existing_workflow_state(self) -> None:
        response = _query_response(FakeWorkflow().run("What is RAG?"))

        self.assertEqual(response.answer, "Answer to: What is RAG?")
        self.assertEqual(response.failure_type, RetrievalFailure.HEALTHY)
        self.assertFalse(response.abstained)
        self.assertEqual(response.recovery_action, RecoveryAction.PROCEED_TO_GENERATION)
        self.assertEqual(response.retry_count, 0)
        self.assertEqual(response.graph_path[-1], "GENERATE")

    def test_health_is_lightweight_and_queries_reuse_lazy_workflow(self) -> None:
        workflow = FakeWorkflow()
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return workflow

        api = create_app(factory)
        routes = {route.path: route for route in api.routes}
        self.assertIn("GET", routes["/health"].methods)
        self.assertIn("POST", routes["/query"].methods)

        async def check_lifespan() -> None:
            async with api.router.lifespan_context(api):
                self.assertIsNone(api.state.workflow)
                self.assertEqual(factory_calls, 0)

                health_response = routes["/health"].endpoint()
                self.assertEqual(health_response.status, "ok")
                self.assertEqual(factory_calls, 0)

                request = SimpleNamespace(app=api)
                first_response = routes["/query"].endpoint(
                    QueryRequest(query="first question"), request
                )
                second_response = routes["/query"].endpoint(
                    QueryRequest(query="second question"), request
                )

                self.assertEqual(first_response.answer, "Answer to: first question")
                self.assertEqual(second_response.answer, "Answer to: second question")
                self.assertIs(api.state.workflow, workflow)
                self.assertEqual(factory_calls, 1)

        asyncio.run(check_lifespan())


if __name__ == "__main__":
    unittest.main()
