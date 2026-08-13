import json
import logging

from langchain_core.prompts import ChatPromptTemplate

from llm.provider import generate_text
from utils.json_tools import extract_json_object

logger = logging.getLogger(__name__)


def _fallback_decision(question: str, memory: dict, error: Exception) -> dict:
    """Use retrieval for non-casual follow-ups when the planner is unavailable."""
    normalized = " ".join(question.lower().split()).strip(".!?")
    casual = normalized in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}
    prediction = memory.get("prediction") or {}
    subject = prediction.get("crop") or prediction.get("display_name") or prediction.get("prediction")
    query = f"{subject} agriculture guidance: {question}" if subject else question
    return {
        "needs_rag": not casual,
        "retrieval_query": query,
        "reason": "Conservative retrieval fallback because the RAG planner is unavailable.",
        "fallback_error": f"{type(error).__name__}: {error}",
    }


def decide_rag(question: str, memory: dict) -> dict:
    prompt = ChatPromptTemplate.from_template('''
Decide whether answering the question requires factual agricultural knowledge
beyond the recent messages and stored prediction.

needs_rag is true for cultivation, irrigation, sowing, nutrients, pests,
diseases, soil, harvesting, weather, and crop-specific guidance. It is false for
greetings and for explaining a prediction from its supplied SHAP explanation.

When needs_rag is true, resolve vague references such as "it", "this", "more
info", and "more details" from memory. retrieval_query must explicitly name the
resolved crop or subject and must be a standalone knowledge-base search query.
Never copy a vague follow-up verbatim. Always write retrieval_query in English,
even when the user's question is in Hindi or another language, because the static
knowledge corpus and its embedding model are English.

Example: if memory is about rice and the question is "more info about it":
{{"needs_rag":true,"retrieval_query":"rice cultivation irrigation nutrients pests diseases harvesting guidance","reason":"cultivation knowledge requested"}}

Return JSON only with needs_rag, retrieval_query, and reason.
Memory: {memory}
Question: {question}
''')
    try:
        raw = generate_text(prompt.format(memory=json.dumps(memory, default=str), question=question))
        result = extract_json_object(raw)
        needs_rag = result.get("needs_rag") is True
        query = str(result.get("retrieval_query") or "").strip()
        if needs_rag and not query:
            query = question
        return {
            "needs_rag": needs_rag,
            "retrieval_query": query,
            "reason": str(result.get("reason") or "Model retrieval decision."),
        }
    except Exception as exc:
        logger.warning("RAG planner fallback used: %s", exc)
        return _fallback_decision(question, memory, exc)
