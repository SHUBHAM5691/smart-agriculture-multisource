# Smart Agriculture Assistant Architecture

## Runtime flow

The application has two independent workflows.

### Prediction

1. The user selects Crop, Fertilizer, or Yield and submits its form.
2. `generate_prediction()` runs the selected trained model.
3. The model produces a prediction and per-prediction SHAP explanation.
4. The result is displayed and stored in Streamlit session memory.

Prediction does not call the RAG planner, embedding retrieval, or the answer LLM.

### Follow-up question

1. `answer_question()` stores the user message and reads memory.
2. `decide_rag()` uses the question and memory to decide whether agricultural documents are required.
3. When required, local sentence embeddings and FAISS retrieve metadata-rich chunks from `data/knowledge/chunks.jsonl`.
4. Current weather/agromet questions retrieve the matching state bulletin live from IMD; live bulletins are never persisted in FAISS.
4. `compose_answer()` answers using memory and optional retrieved chunks.
5. The assistant response is stored in memory.

## Memory

`memory/session_manager.py` stores only:

```python
{
    "messages": [],
    "prediction": None,
}
```

Starting a new interaction clears both values and all Streamlit widget state.

## Main modules

- `app.py` — Streamlit forms, chat, and result presentation.
- `orchestrator/workflow.py` — prediction and question-answer workflows.
- `orchestrator/planner.py` — RAG decision and standalone retrieval-query generation.
- `prediction/` — trained model adapters and SHAP explanations.
- `rag/` — local semantic embeddings, FAISS search and crop filtering.
- `llm/` — Groq API provider and grounded answer composition.
- `memory/session_manager.py` — two-item Streamlit session memory.

## Safety and limitations

- SHAP explains model behavior, not real-world causation.
- Agricultural guidance is educational and should be adapted to local conditions and extension advice.
- The current yield model uses production-derived inputs and is a retrospective estimate, not a pre-season forecast.
- The bundled estimators were trained with scikit-learn 1.5.1; serving them with another version can produce compatibility warnings.
