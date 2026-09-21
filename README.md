# ExpertCall AI

> **Note:** This repository is part of the **Hasamex AI Engineer Technical Case Study**.

## 📌 Project Objective

**ExpertCall AI** is an AI application designed to ingest and analyze 3 expert-call transcripts from the same project. The application extracts answers to 6 core interview-guide questions for each expert, verifies answers with verbatim quotes and exact source timestamps, synthesizes common themes and conflicting viewpoints across all experts, and provides an interactive question-answering interface grounded strictly in transcript evidence to eliminate hallucinations.

---

## 🚀 Planned Features

1. **Transcript Ingestion & Parsing:**
   - Structured parsing and segmenting of the 3 project expert-call transcripts with speaker and timestamp tracking.
2. **Local Embedding & FAISS Semantic Retrieval:**
   - Offline, local dense vector retrieval preserving exact timestamps and metadata.
3. **Interview-Guide Question Answering:**
   - Automated answering of 6 standard interview-guide questions for each expert transcript.
4. **Traceability & Evidence Verification:**
   - Extraction of exact verbatim quotes paired with source timestamps for every key insight and answer.
5. **Cross-Expert Synthesis:**
   - Identification of consensus themes, divergent opinions, and contrasting viewpoints across all 3 experts.
6. **Interactive Cross-Transcript Q&A:**
   - Natural language search and chat querying across all transcripts with source citations.
7. **Anti-Hallucination Guardrails:**
   - Strict retrieval-grounded synthesis ensuring assertions are traceable to transcript evidence.
8. **Streamlit UI:**
   - User-friendly, clean dashboard to review individual expert insights, cross-expert comparisons, and run interactive queries.

---

## 📖 Transcript Ingestion & Deterministic Parser (Step 2)

### 1. Ingested Dataset
The application ingests three real-world case study transcripts located under `data/`:
- `data/Transcript_1_France.txt`: Dr. Jean Martin (Head of Urology, France)
- `data/Transcript_2_Germany.txt`: Anna Keller (Former Hospital Procurement Director, Germany)
- `data/Transcript_3_UK.txt`: Dr. Emily Carter (Consultant Urologist, United Kingdom)

### 2. Structured Evidence Segments
Transcripts are deterministically parsed into strongly typed `EvidenceSegment` units (`src/models.py`) containing:
- `transcript_id`: Unique identifier (e.g., `france_1`, `germany_2`, `uk_3`).
- `expert`: Expert name extracted from header metadata.
- `country`: Market/country (e.g., `France`, `Germany`, `United Kingdom`).
- `role`: Expert's professional title.
- `timestamp`: Verbatim segment timestamp (`MM:SS`).
- `speaker`: Canonical speaker role (`Expert` vs `Interviewer`) alongside raw speaker labels (`Dr. Martin`, `Anna Keller`, `Dr. Carter`).
- `text`: Verbatim, unedited spoken text for that segment.

### 3. Exact Quote & Timestamp Preservation
Every dialogue turn is extracted directly from the raw transcript. Punctuation, capitalization, terminology, and timestamps are preserved with 100% fidelity. No text is normalized, summarized, or rephrased during parsing.

---

## 🔍 Semantic Retrieval Layer with FAISS (Step 3)

### 1. Architecture Flow
```text
Transcript Files (.txt)
         ↓
Deterministic Parser (src/parser.py)
         ↓
Structured EvidenceSegments (42 total segments)
         ↓
Local HuggingFace Embeddings (all-MiniLM-L6-v2)
         ↓
FAISS Vector Index (IndexFlatIP + L2 Normalization)
         ↓
Semantic Similarity Search (Cosine Similarity)
         ↓
Relevant EvidenceSegment with Timestamps & Metadata
```

### 2. Local HuggingFace Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Execution:** Runs 100% locally and offline without sending any transcript text to third-party embedding APIs.
- **Model Caching:** Embeddings model is cached in-process via `get_embedding_model()` to avoid reloading model weights per query.

### 3. FAISS Similarity Metric (Cosine Similarity)
- **Index Type:** `faiss.IndexFlatIP` (Inner Product).
- **Mathematical Equivalence:** Because all segment and query vectors are L2-normalized ($\|u\|_2 = \|v\|_2 = 1$), the inner product is mathematically identical to Cosine Similarity:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$
- Similarity scores range from `-1.0` to `1.0` (where `1.0` represents identical semantic direction).

### 4. Metadata Mapping & Traceability
FAISS stores dense numerical vectors; the aligned mapping maintains a direct 1:1 positional index back to the underlying `EvidenceSegment`. When a vector is retrieved, its full context (`transcript_id`, `expert`, `country`, `role`, `timestamp`, `speaker`, `exact text`) is immediately recovered without loss of information.

### 5. Persistence Strategy
- Vector store artifacts are saved locally to `data/vector_store/`:
  - `index.faiss`: Serialized FAISS vector index binary.
  - `metadata.json`: Serialized evidence segment metadata list.
- **Git Safety:** `data/vector_store/` and all `*.faiss`, `*.index`, and `*.pkl` files are ignored in `.gitignore` to keep the repository clean.
- When existing artifacts are detected, `load_or_build_vector_store()` reloads the persisted index immediately without recomputing embeddings.

> **Note on LLM Generation:** Groq (`GROQ_API_KEY`) is planned for a later step for synthesis and grounded LLM generation. In Step 3, only local embedding and FAISS vector retrieval are implemented. No LLM calls are made.

---

## 📁 Project Structure

```text
expert-call-ai/
├── app.py                      # Streamlit entrypoint
├── requirements.txt            # Project dependencies (Streamlit, FAISS, Sentence-Transformers)
├── README.md                   # Documentation & case study details
├── .gitignore                  # Git ignore rules (ignores .env, data/vector_store/)
├── .env.example                # Environment variable template (GROQ_API_KEY)
├── data/                       # Case study transcript files
│   ├── .gitkeep
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   ├── Transcript_3_UK.txt
│   └── vector_store/           # Local FAISS index & metadata (git-ignored)
├── src/                        # Core application logic and modules
│   ├── __init__.py
│   ├── embeddings.py           # Local Sentence-Transformers embedding wrapper
│   ├── models.py               # EvidenceSegment & Transcript data models
│   ├── parser.py               # Deterministic transcript parser & CLI inspection
│   └── vector_store.py         # FAISS vector index, persistence, and retrieval CLI
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── test_parser.py          # Pytest suite for transcript parser (14 tests)
    └── test_retrieval.py       # Pytest suite for FAISS retrieval (12 tests)
```

---

## ⚙️ Setup & Run Instructions

### Prerequisites
- Python 3.10+ (Python 3.11/3.12 recommended)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Roushan0012/Expert-Call-Ai.git
cd Expert-Call-Ai
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On macOS/Linux
# or
# .venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and supply your GROQ_API_KEY (used in later steps)
```

### 5. Inspect Parsed Transcripts (CLI)
To run the built-in parser inspection tool:
```bash
python -m src.parser
```

### 6. Verify Semantic Retrieval (CLI)
To test semantic search over all 3 transcripts using FAISS:
```bash
python -m src.vector_store
```

### 7. Run Automated Tests
```bash
pytest -v
```

### 8. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the application.
