import logging

from llm.composer import compose_answer, format_prediction_answer
from memory.session_manager import add_message, get_memory, store_prediction
from orchestrator.planner import decide_rag
from prediction.real_model import RealCropModel
from prediction.real_predictors import RealYieldModel
from prediction.fertilizer_subprocess import predict_fertilizer
from rag.retriever import retrieve_with_diagnostics
from rag.imd_advisory import retrieve_live_advisory, wants_live_advisory

logger = logging.getLogger(__name__)
_CONTEXT_MESSAGE_LIMIT = 8


def _recent_memory(memory: dict) -> dict:
    """Keep full session storage while limiting what is sent to the LLM."""
    return {
        "messages": (memory.get("messages") or [])[-_CONTEXT_MESSAGE_LIMIT:],
        "prediction": memory.get("prediction"),
    }


def generate_prediction(prediction_type: str, inputs: dict) -> dict:
    add_message("user", f"predict {prediction_type}")
    try:
        if prediction_type == "crop":
            output = RealCropModel().predict(inputs)
            prediction = {
                "type": "crop",
                "crop": output["crop"],
                "confidence": output["confidence"],
                "class_probabilities": output["class_probabilities"],
                "model_name": output["model_name"],
                "model_version": output["model_version"],
                "inputs": inputs,
                "explanation": output["explanation"],
            }
        elif prediction_type == "fertilizer":
            output = predict_fertilizer(inputs)
            prediction = {
                "type": "fertilizer",
                "prediction": output.get("prediction"),
                "display_name": output.get("display_name"),
                "confidence": output.get("confidence"),
                "model_name": output.get("model_name"),
                "model_version": output.get("model_version"),
                "inputs": inputs,
                "explanation": output["explanation"],
            }
        elif prediction_type == "yield":
            output = RealYieldModel().predict(inputs)
            prediction = {
                "type": "yield",
                "prediction": output.get("prediction"),
                "unit": output.get("unit"),
                "model_name": output.get("model_name"),
                "model_version": output.get("model_version"),
                "inputs": inputs,
                "explanation": output["explanation"],
            }
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        store_prediction(prediction)
        answer = format_prediction_answer(prediction)
        warning = prediction.get("explanation", {}).get("warning")
        if warning:
            answer += f"\n\n*Note: {warning}*"
        add_message("assistant", answer)
        return {"answer": answer, "prediction": prediction}
    except Exception as exc:
        logger.exception("Prediction failed")
        error = f"{type(exc).__name__}: {exc}"
        answer = "The prediction could not be completed. Please check the inputs and try again."
        add_message("assistant", answer)
        return {"answer": answer, "prediction": None, "error": error}


def answer_question(question: str, response_language: str | None = None) -> dict:
    add_message("user", question)
    try:
        memory = _recent_memory(get_memory())
        rag_decision = decide_rag(question, memory)
        rag_docs, rag_error = [], None

        if wants_live_advisory(question):
            rag_docs, rag_error = retrieve_live_advisory(question, memory.get("prediction"))
        elif rag_decision["needs_rag"]:
            rag_docs, rag_error = retrieve_with_diagnostics(rag_decision["retrieval_query"])

        answer = compose_answer(question, memory, rag_docs, response_language=response_language)
        add_message("assistant", answer)
        return {
            "answer": answer,
            "rag_sources": rag_docs,
            "trace": {
                "rag_decision": rag_decision,
                "rag_executed": bool(rag_docs),
                "rag_error": rag_error,
                "live_imd": bool(rag_docs and rag_docs[0].get("live")),
            },
        }
    except Exception as exc:
        logger.exception("Question answering failed")
        error = f"{type(exc).__name__}: {exc}"
        answer = "I could not answer that question right now. Please try again."
        add_message("assistant", answer)
        return {
            "answer": answer,
            "rag_sources": [],
            "trace": {"error": error},
        }
