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
3. **Groq-Powered Grounded Answer Generation:**
   - Strict evidence-grounded RAG synthesis with verbatim citations.
4. **Interview-Guide Question Answering:**
   - Automated answering of 6 standard interview-guide questions for each expert transcript.
5. **Traceability & Evidence Verification:**
   - Extraction of exact verbatim quotes paired with source timestamps for every key insight and answer.
6. **Cross-Expert Synthesis:**
   - Identification of consensus themes, divergent opinions, and contrasting viewpoints across all 3 experts.
7. **Interactive Cross-Transcript Q&A:**
   - Natural language search and chat querying across all transcripts with source citations.
8. **Anti-Hallucination Guardrails:**
   - Strict retrieval-grounded synthesis ensuring assertions are traceable to transcript evidence.
9. **Streamlit UI:**
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

---

## 🧠 Groq-Powered Grounded Answer Generation (Step 4)

### 1. End-to-End RAG Architecture
```text
User Question
      ↓
FAISS Vector Retrieval (src/vector_store.py)
      ↓
Retrieved EvidenceSegments (with timestamps & metadata)
      ↓
Context Construction (src/rag.py)
      ↓
Groq LLM Synthesis (src/llm.py)
      ↓
Structured RAGResponse (Answer + Verifiable Citations)
```

### 2. Separation of Retrieval, Generation, and Citations
A core anti-hallucination principle of this application is strict separation of concerns:
- **Retrieval:** FAISS identifies relevant factual evidence segments from the immutable transcript database.
- **Generation:** Groq LLM synthesizes natural-language answers strictly constrained to the retrieved context.
- **Citations & Quotations:** **The LLM is NOT permitted to invent or modify quotes.** All displayed source quotes and timestamps come directly from the underlying `EvidenceSegment.text` and `EvidenceSegment.timestamp` objects attached to `RAGResponse.sources`.

### 3. Grounding & Anti-Hallucination System Prompt
The Groq model is strictly governed by systemic guardrails:
- Must answer **ONLY** from the supplied transcript context.
- Forbidden from utilizing external pretrained world knowledge.
- Must accurately attribute statements to the specific expert (France: Dr. Jean Martin, Germany: Anna Keller, UK: Dr. Emily Carter).
- Never merges statements from different experts into a false consensus.
- Explicitly highlights cross-market disagreements and divergences.
- If evidence is missing or insufficient, states: *"The provided transcripts do not contain enough evidence to answer this question."*

### 4. Groq Configuration
- **Provider:** Groq API (`GROQ_API_KEY` loaded securely from `.env`).
- **Model:** Configurable via `GROQ_MODEL` (default: `qwen/qwen3.8-27b`).
- **Safety:** API credentials are never logged, exposed, or committed.

---

## 📊 Interview Guide Workflow & Cross-Expert Analysis (Step 5)

### 1. The 6 Interview Guide Questions
The workflow standardizes analysis across all 3 case study transcripts (`src/interview_guide.py`):
1. **Clinical Unmet Need:** What are the primary unmet clinical needs in surgical treatment and patient management?
2. **Current Standard of Care:** What is the current standard of care and how do existing surgical tools perform?
3. **Purchasing & Decision Timeline:** What is the hospital procurement and decision-making timeline for acquiring new medical equipment?
4. **Key Stakeholders & Decision Makers:** Who are the primary stakeholders and decision-makers involved in purchasing evaluations?
5. **Budget & Reimbursement:** What budget, pricing, or reimbursement constraints influence adoption in this market?
6. **Adoption Barriers & Catalysts:** What are the primary barriers to adoption and what catalysts drive clinical acceptance?

### 2. Expert Isolation & Market Attribution
- **Single-Expert Answers:** When querying an expert's perspective (`answer_expert_question()`), retrieval is strictly partitioned using `vector_store.search_by_expert(query, expert=..., k=...)`.
- **Zero Cross-Contamination:** Questions regarding France (Dr. Jean Martin) retrieve exclusively from `france_1`; Germany (Anna Keller) retrieves exclusively from `germany_2`; UK (Dr. Emily Carter) retrieves exclusively from `uk_3`.
- **Market Specifics:** Ensures market-specific nuances (e.g., German hospital procurement boards vs French clinical department heads vs UK NHS trusts) are never conflated.

### 3. Balanced Cross-Expert Retrieval
- **Problem Solved:** Global semantic vector search can allow one verbose transcript or high-keyword section to crowd out other markets.
- **Solution (`search_per_expert`):** Guarantees that retrieval allocates $k$ evidence segments per expert (total $3 \times k$ segments). All 3 countries (France, Germany, UK) contribute equal representation into the synthesis context.
- **Purchasing Timeline Grounding:** Resolves edge cases such as Germany's purchasing timeline (`06:05` Anna Keller: *"Nine to eighteen months is common..."*), which does not contain the literal word "timeline" in its answer. Context pairing and targeted query expansions guarantee this segment is retrieved at rank 1 alongside France (`06:08`: 6–12 months) and the UK (`05:04`: 6–9 months).

### 4. Cross-Expert Synthesis
`compare_experts()` runs grounded comparative analysis:
- **Common Themes:** Grounded synthesis of consensus across all 3 experts (e.g., shared need for ergonomic precision, shared concern over long hospital tender cycles).
- **Contrasting Viewpoints & Disagreements:** Clear, attributed distinctions between clinical department heads (France), procurement/budget directors (Germany), and consultant urologists (UK).
- **Strict Grounding:** The model is forbidden from inventing consensus or projecting viewpoints onto experts who did not express them.

### 5. Verbatim Quote Safety
All quotations, timestamps, and metadata attached to `InterviewQuestionResult` and `CrossExpertAnalysis` are mapped directly from immutable `EvidenceSegment` records. The LLM is never relied upon to generate or hallucinate quotes.

---

## 📁 Project Structure

```text
expert-call-ai/
├── app.py                      # Streamlit entrypoint
├── requirements.txt            # Project dependencies (Streamlit, FAISS, Groq, Sentence-Transformers)
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
│   ├── analysis.py             # Interview guide workflow & cross-expert comparison
│   ├── embeddings.py           # Local Sentence-Transformers embedding wrapper
│   ├── interview_guide.py      # Standard 6 interview guide questions & topics
│   ├── llm.py                  # Groq client wrapper and completion interface
│   ├── models.py               # EvidenceSegment & Transcript data models
│   ├── parser.py               # Deterministic transcript parser & CLI inspection
│   ├── rag.py                  # Grounded RAG answer generation & manual CLI demo
│   └── vector_store.py         # FAISS vector index, expert-filtered retrieval, persistence
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── test_analysis.py        # Pytest suite for interview workflow & comparison (14 tests)
    ├── test_parser.py          # Pytest suite for transcript parser (14 tests)
    ├── test_rag.py             # Pytest suite for Groq grounded RAG (12 tests)
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
# Edit .env and supply your GROQ_API_KEY
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

### 7. Run Manual Groq RAG Verification (CLI)
To test grounded question answering against the real Groq API:
```bash
python -m src.rag
```

### 8. Run Interview Guide & Cross-Expert Analysis (CLI)
To run end-to-end analysis on Interview Guide Question 3 (Purchasing Timelines) across France, Germany, and the UK with Groq synthesis:
```bash
python -m src.analysis
```

### 9. Run Automated Tests
Run the comprehensive 52-test offline verification suite:
```bash
pytest -v
```

### 10. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the application.
