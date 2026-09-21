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

### 1. The 6 Official Interview Guide Questions
The workflow standardizes analysis across all 3 case study transcripts (`src/interview_guide.py`):
1. **Current Adoption of Robotic Surgery:** *How would you describe current adoption of robotic surgery in your market?*
2. **Main Barriers to Adoption:** *What are the main barriers to adoption?*
3. **Hospital Budgets & ROI:** *How important are hospital budgets and ROI in purchasing decisions?*
4. **Surgeon Training & Clinical Outcomes:** *How important are surgeon training and clinical outcomes?*
5. **3–5 Year Adoption Trend:** *What adoption trend do you expect over the next 3–5 years?*
6. **Hospital Decision-Making Timeline:** *What is the typical hospital decision-making timeline for purchasing a new robotic system?*

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

## 🖥️ Streamlit UI & Dashboard Integration (Step 6)

### 1. Application Architecture & Pages
The user interface is built as a responsive, interviewer-friendly Streamlit dashboard structured into 5 dedicated sections:

1. **📊 Overview:**
   - Real-time project statistics derived dynamically from the backend (Transcripts: 3, Experts: 3, Evidence Segments: 42, Model: `all-MiniLM-L6-v2`, Vector Index: `FAISS IndexFlatIP`, LLM: `Groq qwen/qwen3.8-27b`).
   - Commercial market cards detailing the 3 participating experts: France (Dr. Jean Martin), Germany (Anna Keller), and UK (Dr. Emily Carter).
   - Architectural flow diagram outlining deterministic ingestion, dense retrieval, Groq synthesis, and citation verification.

2. **📋 Interview Guide:**
   - Complete support for the 6 official case-study questions.
   - Flexible controls: select an individual market expert (France, Germany, UK) for strict expert isolation, or "All Experts" to render 3 side-by-side comparative cards with clear attribution.
   - Batch analysis mode to evaluate all 6 questions sequentially.
   - Prominent source citation containers for each generated answer.

3. **⚖️ Cross-Expert Analysis:**
   - Structured comparative synthesis across France, Germany, and the UK.
   - Preset comparative topics (budgets, barriers, purchasing timelines, training, 3–5 year adoption) or custom natural language queries.
   - Distinct sections for **Common Themes & Consensus**, **Market Divergences & Differences**, **Synthesis Narrative**, and **Attributed Market Cards**.

4. **💬 Ask Across Calls:**
   - Natural language conversational search and synthesis across all 3 transcripts using grounded RAG.
   - Preset suggested inquiries and interactive custom question input.
   - Session history log preserving queries, answers, and underlying citations across the user session.

5. **📖 Transcript Explorer:**
   - Read-only, immutable viewer for original transcript files.
   - Market selector (France, Germany, UK) with metadata banners.
   - Speaker filter (All Dialogue, Expert Statements Only, Interviewer Questions Only) and instant text keyword search.
   - Chronological dialogue display with `MM:SS` timestamps and speaker tags.

### 2. Source Traceability & Citation UX
Every generated answer features a dedicated **Verified Source Citations** expander displaying:
- Exact `MM:SS` timestamp
- Expert name, professional role, and market flag
- Speaker label and role badge
- Verbatim transcript text enclosed in quotation blocks directly from `EvidenceSegment.text`

### 3. Caching & Performance
- `@st.cache_resource`: Persists the FAISS vector store and Sentence-Transformers embedding model in memory, eliminating redundant index reloads across page navigation.
- `@st.cache_data`: Caches parsed transcript objects and metadata.
- Groq completions are only invoked on explicit user button triggers, preventing unnecessary API calls.

---

## 🧪 End-to-End Evaluation & Grounding Validation (Step 7)

To satisfy the core case-study imperative that **"Every important answer must be traceable to the transcripts and the system must not invent information"**, ExpertCall AI provides a rigorous automated evaluation and grounding verification suite (`src/evaluation.py` and `scripts/evaluate.py`).

### 1. The Seven Grounding Pillars

1. **Source Attribution (100% Verbatim Match):**
   - Every citation attached to an answer is verified character-by-character against the authentic raw transcript segment (`EvidenceSegment.text`).
   - Tampered text, altered phrasing, or fabricated IDs fail validation immediately.

2. **Timestamp Preservation:**
   - Source citations must reflect exact chronological dialogue milestones directly from the raw transcript.
   - Ground truth timestamps (France `01:20`, `02:18`, `06:08`; Germany `02:08`, `06:05`; UK `02:07`, `05:04`) are strictly matched.

3. **Quote Safety & Hallucination Prevention:**
   - Citations and quotation cards rendered in the UI originate strictly from immutable `EvidenceSegment.text` objects in memory.
   - The LLM is structurally prohibited from producing its own quotes or having its generated output treated as transcript citations.

4. **Expert Isolation (Zero Cross-Market Leakage):**
   - Single-expert queries (e.g., France / Dr. Jean Martin) are retrieved through isolated expert-filtered indexes (`vector_store.search_by_expert`).
   - Leaks from other markets (e.g. Germany or UK dialogue in a France answer) are strictly flagged as isolation failures.

5. **Balanced Cross-Expert Representation:**
   - Comparative queries require balanced evidence allocation across France, Germany, and the United Kingdom via `vector_store.search_per_expert(k_per_expert=...)`.
   - Prevents one verbose expert or high-keyword turn from crowding out other markets.

6. **Insufficient Evidence Fallback:**
   - When asked questions about topics outside the transcripts (e.g., exact manufacturer market share, total installed systems count, market size in euros, or 2030 revenue forecasts), the system returns a grounded fallback:
     > *"The provided transcripts do not contain enough evidence to answer this question."*
   - Strictly forbids inventing speculative statistics, percentages, or financial numbers.

7. **Hallucination Resistance to Adversarial Prompts:**
   - Defends against adversarial leading questions containing false premises (e.g., *"Why did Dr. Martin say robotic surgery reduces costs by 50%?"* or *"Since Germany has the fastest adoption, why is that?"*).
   - The system detects unsupported claims, refuses to adopt fabricated facts, and clarifies that the premise is absent from the evidence.

---

### 2. The 15 Golden Benchmark Test Cases

The evaluation suite executes across 15 standardized golden test cases spanning core guide questions, missing-evidence queries, and adversarial challenges:

| ID | Category | Target / Query | Primary Grounding Focus | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| `GOLDEN-01` | Expert Isolation | France Adoption (Q1) | France market isolation | 100% France segments (`01:20`) |
| `GOLDEN-02` | Expert Isolation | Germany Barriers (Q2) | Germany market isolation | 100% Germany segments (`02:08`) |
| `GOLDEN-03` | Cross-Expert | Budgets & ROI (Q3) | Cross-market representation | France, Germany & UK represented |
| `GOLDEN-04` | Timestamp Correctness | Germany Purchasing Timeline (Q6) | Germany milestone retrieval | Milestone `06:05` retrieved |
| `GOLDEN-05` | Cross-Expert | Surgeon Training (Q4) | Cross-market representation | France, Germany & UK represented |
| `GOLDEN-06` | Cross-Expert | 3–5 Year Trend (Q5) | Cross-market representation | France, Germany & UK represented |
| `GOLDEN-07` | Cross-Expert | Purchasing Timelines Comparison | Balanced cross-market timelines | France (`06:08`), Germany (`06:05`), UK (`05:04`) |
| `INSUFFICIENT-01` | Insufficient Evidence | % of French hospitals with robots | Numerical hallucination guard | Grounded fallback, no invented % |
| `INSUFFICIENT-02` | Insufficient Evidence | Manufacturer market share | Entity hallucination guard | Grounded fallback, no invented share |
| `INSUFFICIENT-03` | Insufficient Evidence | 2030 revenue projections | Forward-looking fabrication guard | Grounded fallback, no invented revenue |
| `INSUFFICIENT-04` | Insufficient Evidence | Exact installed count in Germany | Statistical hallucination guard | Grounded fallback, no invented count |
| `INSUFFICIENT-05` | Insufficient Evidence | Total market size in euros | Currency / figure hallucination guard | Grounded fallback, no invented € figure |
| `ADVERSARIAL-01` | Hallucination Resistance | "Since Germany has the fastest adoption..." | False premise resistance | Rejects premise; notes lack of comparative ranking |
| `ADVERSARIAL-02` | Hallucination Resistance | "Why did Dr. Martin say robots reduce costs by 50%?" | Fabricated quote resistance | Refuses fabricated quote; explains 50% not stated |
| `ADVERSARIAL-03` | Hallucination Resistance | "Which expert said robots always improve outcomes?" | Unsupported absolute resistance | Clarifies nuances; refuses "always" generalization |

---

## 🎯 Case Study Requirement Traceability Matrix

Every requirement from the official case study is explicitly satisfied, tested, and verifiable:

| Case Study Requirement | Architecture Implementation | Verification / Test Evidence | Status |
| :--- | :--- | :--- | :--- |
| **1. Upload/read 3 transcripts** | `src/parser.py` deterministically parses France, Germany, and UK transcripts into 42 immutable `EvidenceSegment` units | `tests/test_parser.py` (14 tests passing); CLI `python -m src.parser` | **COMPLETE** |
| **2. Answer interview-guide questions for each expert** | `src/interview_guide.py` defines the 6 official questions; `src/analysis.py` answers each per expert with isolated retrieval | `tests/test_analysis.py` (15 tests passing); `GOLDEN-01`, `GOLDEN-02` | **COMPLETE** |
| **3. Extract useful exact quotes** | `src/models.py` preserves verbatim dialogue; `src/ui/common.py` renders quotes strictly from `EvidenceSegment.text` | `tests/test_rag.py::test_source_text_identical_to_original_segment`; `GOLDEN-01` to `07` | **COMPLETE** |
| **4. Show source timestamp for each answer/quote** | `EvidenceSegment.timestamp` preserved during parsing and displayed on every answer citation card | `tests/test_parser.py::test_timestamps_extracted`; `GOLDEN-04` (`06:05`) | **COMPLETE** |
| **5. Identify common themes and disagreements** | `compare_experts()` in `src/analysis.py` synthesizes consensus themes and market-attributed differences | `tests/test_analysis.py::test_cross_expert_analysis_includes_all_experts`; `GOLDEN-03`, `05`, `06`, `07` | **COMPLETE** |
| **6. Allow questions across all transcripts** | `answer_question()` in `src/rag.py` and *Ask Across Calls* UI page enable arbitrary cross-call natural language queries | `tests/test_rag.py` (12 tests passing); `app.py` Ask page | **COMPLETE** |

---

## 📈 Scaling Strategy: 3 → 30+ Transcripts

| Dimension | Current Implementation (3 Calls) | Production Scaling Architecture (30+ Calls) |
| :--- | :--- | :--- |
| **Vector Storage** | Local FAISS `IndexFlatIP` file (`index.faiss`) | Managed distributed vector database (Qdrant, Milvus, pgvector) |
| **Indexing** | In-memory exact inner-product search | HNSW / IVFFlat indexing with metadata payload filtering |
| **Ingestion** | Synchronous local script execution | Async distributed task queue (Celery, Kafka, AWS SQS) |
| **Embedding Generation** | On-demand single CPU encoding | Batched GPU encoding with micro-batch queuing |
| **Retrieval Strategy** | Dense cosine similarity | Hybrid search (Dense vector + Sparse BM25) + Cross-Encoder Reranking |
| **Partitioning** | Python-level expert filtering | Native database multi-tenancy (partitioned by Project, Market, Date) |
| **Metadata Tracking** | Direct 1:1 positional array in JSON | Relational database (PostgreSQL) with ACID transaction logs |
| **Caching** | Streamlit `@st.cache_resource` in RAM | Distributed semantic cache (Redis) for repeated question embeddings |
| **Access Control** | Single-user local workspace | Enterprise RBAC with project-level read/write permissions |
| **Observability** | Standard Python `logging` | Distributed tracing (OpenTelemetry, LangSmith, TruLens) |
| **Evaluation** | 15 golden cases + 77 unit tests | Automated CI/CD evaluation pipeline with dynamic golden datasets |

*For deep architectural details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 📁 Project Structure

```text
expert-call-ai/
├── app.py                      # Streamlit entrypoint and sidebar router
├── requirements.txt            # Project dependencies (Streamlit, FAISS, Groq, Sentence-Transformers)
├── README.md                   # Comprehensive documentation & case study details
├── .gitignore                  # Git ignore rules (ignores .env, data/vector_store/)
├── .env.example                # Environment variable template (GROQ_API_KEY)
├── data/                       # Case study transcript files
│   ├── .gitkeep
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   ├── Transcript_3_UK.txt
│   └── vector_store/           # Local FAISS index & metadata (git-ignored)
├── docs/                       # Technical specifications & interview documentation
│   ├── ARCHITECTURE.md         # Deep-dive system architecture, model choices & grounding
│   └── DEMO_GUIDE.md           # 5–7 min live demo script & technical interview talking points
├── scripts/                    # Command-line utility and verification runners
│   └── evaluate.py             # Grounding and evaluation benchmark CLI runner
├── src/                        # Core application logic and modules
│   ├── __init__.py
│   ├── analysis.py             # Interview guide workflow & cross-expert comparison
│   ├── embeddings.py           # Local Sentence-Transformers embedding wrapper
│   ├── evaluation.py           # Evaluation framework, 7 grounding pillars, 15 golden cases
│   ├── interview_guide.py      # Official 6 interview guide questions & topics
│   ├── llm.py                  # Groq client wrapper and completion interface
│   ├── models.py               # EvidenceSegment & Transcript data models
│   ├── parser.py               # Deterministic transcript parser & CLI inspection
│   ├── rag.py                  # Grounded RAG answer generation & manual CLI demo
│   ├── vector_store.py         # FAISS vector index, expert-filtered retrieval, persistence
│   └── ui/                     # Modular Streamlit UI presentation layer
│       ├── __init__.py
│       ├── ask.py              # Ask Across Calls conversational interface
│       ├── common.py           # Shared UI utilities, caching, and source cards
│       ├── cross_expert.py     # Cross-Expert Analysis and comparative synthesis
│       ├── explorer.py         # Transcript Explorer with speaker filters
│       ├── interview.py        # Interview Guide 6-question workflow
│       └── overview.py         # Dashboard metrics and expert profile cards
└── tests/                      # Automated test suite (77 tests)
    ├── __init__.py
    ├── test_analysis.py        # Pytest suite for interview workflow & comparison (15 tests)
    ├── test_app.py             # Pytest suite for Streamlit UI & integration (12 tests)
    ├── test_evaluation.py      # Pytest suite for evaluation & grounding validation (12 tests)
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

### 9. Run Grounding & Hallucination Evaluation (CLI)
To execute the automated evaluation benchmark across the 15 Golden Test Cases:
```bash
# Offline deterministic grounding verification
python scripts/evaluate.py

# Live Groq RAG synthesis evaluation
python scripts/evaluate.py --live
```

### 10. Run Automated Tests
Run the comprehensive 77-test offline verification suite:
```bash
pytest -v
```

### 11. Run the Streamlit Application
Launch the interactive web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the application:
- **Overview:** Explore project statistics and participating market experts.
- **Interview Guide:** Select questions 1–6 and explore single-expert or all-expert responses with verified citations.
- **Cross-Expert Analysis:** Compare market perspectives, consensus themes, and divergences.
- **Ask Across Calls:** Interactively query transcripts with grounded RAG.
- **Transcript Explorer:** Inspect immutable raw timestamped dialogue turns directly.

---

## 🎬 Interview Demonstration & Talking Points

For a timed 5–7 minute live interview presentation flow and technical Q&A cheat sheet, refer to:
- **[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)**: Timed demo walkthrough from problem intro to scaling.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Comprehensive architectural breakdown, model choice rationales, and grounding guarantees.
