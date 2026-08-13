import unittest
from unittest.mock import patch

from llm.composer import compose_answer
from orchestrator.planner import decide_rag
from orchestrator.workflow import _recent_memory, answer_question, generate_prediction


class WorkflowTests(unittest.TestCase):
    def test_recent_memory_limits_llm_context_without_mutating_session(self):
        messages = [{"role": "user", "content": str(index)} for index in range(12)]
        memory = {"messages": messages, "prediction": {"crop": "maize"}}

        recent = _recent_memory(memory)

        self.assertEqual(len(memory["messages"]), 12)
        self.assertEqual([item["content"] for item in recent["messages"]], [str(i) for i in range(4, 12)])
        self.assertIs(recent["prediction"], memory["prediction"])
    @patch("orchestrator.workflow.add_message")
    @patch("orchestrator.workflow.store_prediction")
    @patch("orchestrator.workflow.RealCropModel")
    def test_crop_prediction_runs_without_rag(self, crop_model, store, add_message):
        crop_model.return_value.predict.return_value = {
            "crop": "rice",
            "confidence": 0.9,
            "class_probabilities": {"rice": 0.9},
            "model_name": "crop-model",
            "model_version": "v1",
            "explanation": {"type": "shap", "top_factors": [], "warning": "warning"},
        }

        with patch("orchestrator.workflow.decide_rag") as rag:
            result = generate_prediction("crop", {"nitrogen": 90})

        rag.assert_not_called()
        store.assert_called_once()
        self.assertEqual(result["prediction"]["crop"], "rice")
        self.assertEqual(add_message.call_count, 2)

    @patch("orchestrator.workflow.add_message")
    @patch("orchestrator.workflow.compose_answer", return_value="Hello")
    @patch("orchestrator.workflow.decide_rag", return_value={"needs_rag": False, "retrieval_query": ""})
    @patch("orchestrator.workflow.get_memory", return_value={"messages": [], "prediction": None})
    def test_question_without_rag_skips_retrieval(self, _memory, rag, _compose, add_message):
        with patch("orchestrator.workflow.retrieve_with_diagnostics") as retrieve:
            result = answer_question("hello")

        rag.assert_called_once()
        retrieve.assert_not_called()
        self.assertFalse(result["trace"]["rag_executed"])
        self.assertEqual(add_message.call_count, 2)

    @patch("orchestrator.workflow.add_message")
    @patch("orchestrator.workflow.compose_answer", return_value="Rice guidance")
    @patch("orchestrator.workflow.retrieve_with_diagnostics", return_value=([{"rank": 1, "content": "doc"}], None))
    @patch("orchestrator.workflow.decide_rag", return_value={"needs_rag": True, "retrieval_query": "rice cultivation"})
    @patch("orchestrator.workflow.get_memory")
    def test_question_with_rag_uses_standalone_query(
        self, memory, _rag, retrieve, _compose, _message
    ):
        memory.return_value = {"messages": [], "prediction": {"type": "crop", "crop": "rice"}}

        result = answer_question("more info about it")

        retrieve.assert_called_once_with("rice cultivation")
        self.assertTrue(result["trace"]["rag_executed"])

    @patch(
        "orchestrator.planner.generate_text",
        return_value='{"needs_rag":true,"retrieval_query":"rice cultivation guidance",'
        '"reason":"knowledge requested"}}',
    )
    def test_rag_planner_handles_trailing_model_noise(self, _generate):
        result = decide_rag("more info about it", {"prediction": {"crop": "rice"}})

        self.assertTrue(result["needs_rag"])
        self.assertEqual(result["retrieval_query"], "rice cultivation guidance")

    @patch("orchestrator.planner.generate_text", side_effect=ConnectionError("Groq unavailable"))
    def test_rag_planner_connection_error_is_visible(self, _generate):
        result = decide_rag("rice cultivation", {"prediction": {"crop": "rice"}})

        self.assertTrue(result["needs_rag"])
        self.assertEqual(result["retrieval_query"], "rice agriculture guidance: rice cultivation")
        self.assertEqual(result["fallback_error"], "ConnectionError: Groq unavailable")

    @patch("orchestrator.workflow.add_message")
    @patch("orchestrator.workflow.compose_answer", return_value="Maize guidance")
    @patch("orchestrator.workflow.retrieve_with_diagnostics", return_value=([], None))
    @patch("orchestrator.workflow.get_memory")
    @patch(
        "orchestrator.planner.generate_text",
        return_value='{"needs_rag":true,"retrieval_query":"maize cultivation irrigation nutrients pests guidance",'
        '"reason":"more crop knowledge requested"}',
    )
    def test_maize_followup_builds_specific_retrieval_query(
        self, _generate, memory, retrieve, _compose, _message
    ):
        memory.return_value = {
            "messages": [{"role": "assistant", "content": "Recommended crop: maize"}],
            "prediction": {"type": "crop", "crop": "maize"},
        }

        answer_question("share more information about it")

        retrieve.assert_called_once_with("maize cultivation irrigation nutrients pests guidance")

    @patch("llm.composer.generate_text", side_effect=TimeoutError("generation timed out"))
    def test_composer_fallback_displays_provider_error(self, _generate):
        prediction = {
            "type": "crop",
            "crop": "Rice",
            "explanation": {"summary": "summary", "top_factors": []},
        }

        answer = compose_answer("why?", {"messages": [], "prediction": prediction}, [])

        self.assertIn("temporarily unavailable", answer)
        self.assertNotIn("TimeoutError", answer)
        self.assertIn("Rice", answer)


if __name__ == "__main__":
    unittest.main()
