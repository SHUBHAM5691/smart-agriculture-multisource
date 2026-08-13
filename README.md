# Smart Agriculture Assistant — Groq Edition

The application has two simple flows:

1. A submitted Crop, Fertilizer, or Yield form runs the selected trained model.
2. The prediction and its SHAP explanation are displayed and stored in memory.
3. Follow-up questions are checked through Groq to decide whether RAG is required.
4. Local sentence embeddings and FAISS return metadata-rich chunks from official agricultural sources only when required.
5. Current weather-sensitive questions retrieve a state bulletin live from IMD instead of embedding stale advisories.
6. The Groq-hosted model answers from the question, stored messages, prediction and retrieved chunks.
7. A turn-based voice layer can transcribe Hindi agricultural speech, let the user correct
   the transcript, retrieve against the English corpus, and speak the answer in Hindi.

## Voice advisory

After a validated prediction is active, use the microphone inside the same chat input used
for typing, speak one question, and submit it. No separate voice panel, language selection,
or transcription button is required. Groq Whisper detects
the language and produces the transcript; an agricultural normalization step standardizes
clear numbers and units without inventing missing values. The recording is automatically
submitted through the normal RAG/prediction-memory workflow, and a multilingual neural TTS
adapter automatically plays the answer in the detected language.

Hindi, Gujarati, Marathi, and English are registered in `voice/service.py`; adding another
language requires a language profile rather than changes to the Streamlit workflow. Voice
questions do not infer required prediction-form values, because silently guessing soil or
field measurements would make model results unreliable.

ASR uses `whisper-large-v3` by default for accuracy on agricultural terminology. Set
`GROQ_ASR_MODEL=whisper-large-v3-turbo` when lower latency and cost matter more. TTS is a
separate network service because Groq TTS does not currently provide a Hindi voice.

Evaluate ASR on consented agricultural recordings with a CSV manifest containing
`audio_path,reference_text,language` (use `hi` for Hindi):

```bash
python scripts/evaluate_asr.py evaluation/hindi_agriculture.csv
```

The script reports per-recording and corpus word error rate (WER). Include crop names,
fertilizer terms, numbers, units, accents, background noise, and both male and female
speakers in the evaluation set; do not commit identifiable farmer recordings without consent.

## Multi-source knowledge base

The source manifest is `data/sources.json`. It currently includes ICAR natural-resource
management, Vikaspedia package-of-practices, and the TNAU Agriculture and Horticulture
2020 guides. The original `data/agriculture_knowledge.txt` remains only as a safe fallback.

Build or refresh the local knowledge corpus:

```bash
python3 scripts/ingest_sources.py
```

To ingest a subset while developing:

```bash
python3 scripts/ingest_sources.py --source tnau_agriculture_2020
```

Generated chunks are written to `data/knowledge/chunks.jsonl`; source PDFs are cached in
`data/downloads/`. Each chunk records the official URL, publication year, page, state,
inferred crops, and topics. At application startup, document embeddings are generated once
and the FAISS index is saved under `data/vector_index/`. Later starts load that saved index.
If the corpus or embedding-model name changes, its fingerprint changes and the application
automatically rebuilds and replaces the stored index. Each question embeds only its query;
it does not regenerate document embeddings.

Current IMD state advisories are requested at question time. A state must be present in the
question or in the active prediction inputs. If IMD is unavailable, diagnostics report the
failure rather than silently substituting an old bulletin.

Streamlit session state stores only `messages` and `prediction`.

## Explainability

The crop Random Forest, fertilizer LightGBM classifier, and yield XGBoost regressor
use real per-prediction SHAP values through `shap.TreeExplainer`. Classification
explanations are selected for the predicted class. Yield SHAP values explain the
model's `log1p(yield)` output. The app displays the largest contributions in a
chart and table. SHAP explains model behavior, not agricultural causation.

The fertilizer model runs in a short-lived worker process. This keeps its native
LightGBM runtime isolated from Streamlit, FAISS, and PyTorch on macOS while preserving
the original trained artifact, preprocessing, probabilities, and SHAP explanation.

The current artifact bundle contains mixed serialization provenance: some
preprocessing objects were created with scikit-learn 1.5.1, while the final crop
estimator was exported with 1.9.0. The deployment therefore pins 1.9.0 to match the
final estimator, but compatibility warnings can still appear for older preprocessing
objects. The prediction smoke tests pass. For a production release, retrain and
export every estimator and preprocessing object in one pinned environment.

The current yield model includes reported production and production-per-area among
its inputs. This creates target leakage for a true pre-season forecast, so the app
labels it as a retrospective estimate. Retrain the model without those features
before using it for forward-looking yield prediction.

## Diagnostics

This build is configured for transparent development diagnostics. Unexpected
failures display their exception type and message instead of a generic model error.
RAG decisions and retrieval failures are available in the
**Technical diagnostics** expander after an interaction.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure the Groq API in `.env`. Never commit this file or send the key in chat:

```env
GROQ_API_KEY=your_real_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TIMEOUT=60
GROQ_MAX_RETRIES=3
GROQ_ASR_MODEL=whisper-large-v3
VOICE_ENABLED=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
RAG_MIN_RELEVANCE=0.25
DEBUG=true
```

Run:

```bash
./run_app.sh
```

This edition runs on port `8503`, so it can run beside the original project on
port `8502`. The `.gitignore` excludes `.env` and `.venv`.

## Example queries

- Which crop should I cultivate?
- Why was that crop recommended?
- How should I irrigate it?
- Which crop should I cultivate and how should I grow it?
- How can you help me in cultivation?
