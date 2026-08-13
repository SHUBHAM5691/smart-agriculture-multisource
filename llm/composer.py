import json
import logging
from llm.provider import generate_text

logger = logging.getLogger(__name__)


def format_prediction_answer(prediction_record: dict | None, intro: str | None = None) -> str:
    fallback_parts = []
    if prediction_record:
        if intro:
            fallback_parts.append(intro)
        if prediction_record.get("type") == "fertilizer":
            fallback_parts.append(
                f"Recommended fertilizer: **{prediction_record.get('display_name') or prediction_record.get('prediction')}**"
            )
        elif prediction_record.get("type") == "yield":
            fallback_parts.append(
                f"Estimated yield: **{prediction_record.get('prediction')} {prediction_record.get('unit', '')}**"
            )
        else:
            fallback_parts.append(f"Recommended crop: **{prediction_record.get('crop')}**")
        explanation = prediction_record["explanation"]
        if prediction_record.get('explanation', {}).get('top_factors'):
            if explanation.get("type") == "shap":
                factors = ", ".join(
                    f"{item['label']} ({'pushed the result higher' if item['contribution'] >= 0 else 'pushed the result lower'})"
                    for item in explanation["top_factors"][:3]
                )
                fallback_parts.append(f"Main model signals: {factors}.")
            else:
                factors = ", ".join(item["label"] for item in explanation["top_factors"][:3])
                fallback_parts.append(f"Main input differences: {factors}.")
    if fallback_parts:
        return "\n\n".join(fallback_parts)
    return "I can recommend a crop, explain a stored recommendation, and answer cultivation questions using the knowledge base."


def compose_answer(
    question: str,
    memory: dict,
    rag_documents: list[dict],
    response_language: str | None = None,
) -> str:
    rag_context = "\n\n".join(
        f"[Chunk {x['rank']}; source={x.get('source')}; state={x.get('state')}; "
        f"year={x.get('publication_year')}; page={x.get('page')}]\n{x['content']}"
        for x in rag_documents
    )
    prompt = f'''You are a smart-agriculture decision-support assistant. Answer clearly and directly.
Rules:
1. Use the supplied prediction only when relevant.
2. When explanation.type is "shap", describe it as model-specific SHAP attribution, not causation.
   When it is "heuristic_placeholder", never call it SHAP or validated explainability.
3. Use retrieved context for cultivation and factual agricultural advice.
4. Do not invent pesticide doses, legal claims, scheme details, or precise facts absent from context.
5. Mention briefly when local variety, region, season, soil test, weather, or extension advice is needed.
6. Use display_name exactly when it is present. Do not invent or expand abbreviations yourself.
7. Stay within agriculture and application capabilities.
8. Use only supplied context and prediction data. Be concise and do not mention internal routing or JSON.
9. For greetings, acknowledgements, casual conversation, or unclear short messages,
   respond naturally and briefly. Do not force the conversation back to farming.
10. Answer the current question rather than narrating or summarizing the conversation.
11. If the user asks for another prediction, tell them to click Start New Interaction
    and submit the appropriate form. Chat must never claim to run a new prediction.
12. Treat state-specific guidance as applicable only to that state. Identify older publication
    years when they materially affect the answer. For pesticide products or doses, explicitly
    tell the user to verify the current label and local regulations.
13. When live IMD context is supplied, say which state it covers and include its source URL.
14. {f"Answer naturally in {response_language}. Keep necessary agricultural terms and units accurate." if response_language else "Answer in the same language and script as the current user question."}
User question: {question}
Memory: {json.dumps(memory, default=str)}
Retrieved context: {rag_context}
'''
    try:
        return generate_text(prompt)
    except Exception as exc:
        logger.exception("Answer composition failed")
        fallback = format_prediction_answer(
            memory.get("prediction"),
            intro="The language model is temporarily unavailable, but the local prediction completed.",
        )
        if rag_documents:
            fallback += "\n\nRetrieved knowledge:\n" + "\n".join(
                f"- {item['content']}" for item in rag_documents[:3]
            )
        return f"{fallback}\n\n*The language model is temporarily unavailable.*"
