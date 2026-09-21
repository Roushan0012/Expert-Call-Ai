# ExpertCall AI — System Architecture & Design Specification

> **Technical Case Study:** Hasamex AI Engineer  
> **Repository:** [https://github.com/Roushan0012/Expert-Call-Ai](https://github.com/Roushan0012/Expert-Call-Ai)  
> **Status:** Production-Ready & Verified (77/77 Tests Passing | 15/15 Benchmark Cases Passing)

---

## 1. Executive Summary & Objective

**ExpertCall AI** is a specialized, production-grade retrieval-augmented generation (RAG) platform designed to ingest, process, and analyze expert-call transcripts. The system enables healthcare strategy analysts and commercial procurement teams to:
1. Ingest and deterministically parse multiple commercial expert transcripts while preserving exact speaker turns, timestamps, and verbatim dialogue.
2. Automatically answer standard 6-question Interview Guides for each individual expert with strict market isolation.
3. Synthesize cross-expert consensus themes, strategic divergences, and commercial differences across all transcripts without false consensus.
4. Allow freeform, cross-transcript question answering with strict citation grounding.
5. Provide mathematical and architectural guarantees against hallucination, unsupported claims, and fabricated quotes.

---

## 2. End-to-End System Architecture

```text
                                 ┌───────────────────────────┐
                                 │       User / Analyst      │
                                 └─────────────┬─────────────┘
                                               │ (Web Interface / CLI)
                                               ▼
                                 ┌───────────────────────────┐
                                 │   Streamlit Web UI Layer  │
                                 │      (app.py, src/ui/)    │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Application/Analysis Layer│
                                 │     (src/analysis.py)     │
                                 └──────┬─────────────┬──────┘
                                        │             │
                    Single-Expert Query │             │ Cross-Expert Query
                    (Filtered k=3)      │             │ (Balanced k=2/expert)
                                        ▼             ▼
                                 ┌───────────────────────────┐
                                 │   FAISS Retrieval Engine  │
                                 │   (src/vector_store.py)   │
                                 └─────────────┬─────────────┘
                                               │ Cosine Similarity Search
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Immutable EvidenceSegments│
                                 │      (src/models.py)      │
                                 └──────┬─────────────┬──────┘
                                        │             │
                Exact Segments          │             │ Direct Pointer
                Passed into Context     │             │ For Verified Citations
                                        ▼             │
                                 ┌──────────────┐     │
                                 │   Groq LLM   │     │
                                 │ (src/rag.py) │     │
                                 └──────┬───────┘     │
                                        │             │
                                        ▼             ▼
                                 ┌───────────────────────────┐
                                 │ Grounded Synthesis Answer │
                                 │             +             │
                                 │ Verified Source Citations │
                                 └───────────────────────────┘
```

---

## 3. Component Breakdown & Responsibilities

### 3.1 Transcript Ingestion & Deterministic Parser (`src/parser.py`)
- **Role:** Ingests raw text transcripts from `data/` (`Transcript_1_France.txt`, `Transcript_2_Germany.txt`, `Transcript_3_UK.txt`).
- **Mechanism:** Deterministic regex parsing extracts metadata headers (`Expert`, `Title/Role`, `Market/Country`) and splits dialogue into turn-by-turn units bounded by timestamps (`[MM:SS]`).
- **Fidelity Guarantee:** Raw transcript text, capitalization, and punctuation are preserved with 100% fidelity. No text is normalized, summarized, or paraphrased during parsing.

### 3.2 EvidenceSegment Domain Model (`src/models.py`)
- **Role:** Strongly typed, immutable data structures representing factual units of evidence.
- **Fields:**
  - `segment_id`: Unique identifier (`france_1_seg_002`).
  - `transcript_id`: Source file identifier (`france_1`).
  - `expert`: Expert name (`Dr. Jean Martin`).
  - `country`: Commercial market (`France`).
  - `role`: Professional position (`Head of Urology, University Hospital`).
  - `timestamp`: Verbatim MM:SS timestamp (`01:20`).
  - `speaker`: Standardized speaker category (`Expert` vs `Interviewer`).
  - `text`: Verbatim spoken dialogue turn.

### 3.3 Embedding Model (`src/embeddings.py`)
- **Role:** Maps text segments and user queries into dense numerical vector representations.
- **Implementation:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **Execution:** Runs 100% locally and offline. Vectors are L2-normalized upon creation.

### 3.4 FAISS Vector Database (`src/vector_store.py`)
- **Role:** In-process dense vector indexing and sub-millisecond similarity retrieval.
- **Index Type:** `faiss.IndexFlatIP` (Inner Product).
- **Mathematical Equivalence:** Because all segment vectors and query vectors are L2-normalized ($\|u\|_2 = \|v\|_2 = 1$), the inner product is mathematically identical to Cosine Similarity:
  $$\text{Cosine Similarity}(u, v) = u \cdot v$$
- **Metadata Aligned Array:** A parallel in-memory array preserves a 1:1 index mapping between FAISS vector IDs and `EvidenceSegment` objects, ensuring zero metadata loss.

### 3.5 Specialized Retrieval Layer (`src/vector_store.py`)
- **Global Search (`vector_store.search`):** Top-$k$ semantic similarity across all indexed segments.
- **Expert-Isolated Search (`search_by_expert`):** Restricts candidates strictly to segments matching a target expert/country, preventing cross-market leakage.
- **Balanced Multi-Expert Search (`search_per_expert`):** Allocates an equal budget of $k$ segments per expert, preventing verbose transcripts from monopolizing context in comparative questions.

### 3.6 Groq LLM Synthesis Layer (`src/llm.py`, `src/rag.py`)
- **Role:** Natural language synthesis constrained to retrieved evidence.
- **Provider & Model:** Groq API running `qwen/qwen3.8-27b` (configurable via `GROQ_MODEL`).
- **Prompt Engineering:** Strict closed-world system instructions forbidding external knowledge, speculation, or quote fabrication.

### 3.7 Interview Guide & Cross-Expert Analysis Layer (`src/analysis.py`, `src/interview_guide.py`)
- **Role:** Executes the 6 official case study questions across all 3 market transcripts.
- **Capabilities:**
  - Standardizes the 6 case-study questions.
  - Generates market-isolated analyses for France, Germany, and the UK.
  - Synthesizes consensus themes and market-specific disagreements.
  - Automatically handles query expansions (e.g. mapping "timeline" queries to Anna Keller's `06:05` purchasing dialogue).

### 3.8 Citation & Verification Layer (`src/ui/common.py`)
- **Role:** Renders verified citation cards in the UI.
- **Non-Negotiable Guardrail:** **The LLM is NEVER permitted to generate or format quotes.** Citations originate exclusively from the `sources: List[EvidenceSegment]` field attached to response objects, ensuring every quote displayed to the user is authentic transcript text.

### 3.9 Grounding & Evaluation Framework (`src/evaluation.py`, `scripts/evaluate.py`)
- **Role:** Continuous automated validation of system grounding across 7 core pillars and 15 golden benchmark cases.
- **Execution:** Supports both deterministic offline validation (zero API usage) and live Groq evaluation.

---

## 4. Technology Stack & Model Choice Rationale

### 4.1 Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Local & Offline:** Operates entirely within the local Python runtime. No sensitive transcript data is transmitted to third-party embedding APIs.
- **Resource Efficiency:** 22.7 million parameters (~80MB model size). Delivers sub-15ms vector encoding on standard laptop CPUs with minimal memory overhead.
- **Semantic Efficacy:** High performance on MTEB retrieval benchmarks for short-to-medium conversational segments.
- **Fixed Dimensionality:** 384-dimensional dense vectors provide an optimal balance between semantic granularity and vector index memory footprint.

### 4.2 Vector Search: `faiss-cpu`
- **Blazing Fast In-Memory Execution:** C++ optimized matrix multiplication enables nearest-neighbor retrieval in microseconds.
- **Clean Persistence:** Simple serialisation to local disk (`index.faiss` and `metadata.json`) without requiring dedicated server infrastructure, background daemon processes, or external ports.
- **Architectural Match:** For 3 expert transcripts (42 segments), flat exhaustive indexing (`IndexFlatIP`) guarantees 100% recall with zero approximation error.
- **Clear Conceptual Scalability:** The exact same interface scales to millions of vectors using FAISS IVF or HNSW indices.

### 4.3 LLM Provider: Groq Cloud (`qwen/qwen3.8-27b`)
- **Inference Speed:** Groq LPU architecture delivers ~300 tokens/second, ensuring responsive analysis and interactive dashboard exploration.
- **Instruction Adherence:** Excels at complex constraint following, source attribution, and structured JSON/markdown synthesis.
- **Configurability:** Fully configurable through the `GROQ_MODEL` environment variable.
- **Boundary of Authority:** **The LLM is strictly an answer synthesizer, NOT the source of truth.** All truth resides exclusively in the retrieved `EvidenceSegment` objects.

---

## 5. Hallucination Prevention Architecture

Hallucination reduction is achieved through an 8-stage defense-in-depth pipeline:

| Stage | Mechanism | Hallucination Risk Mitigated |
| :--- | :--- | :--- |
| **1. Immutable Ingestion** | Regex-based deterministic parsing | Paraphrasing or distortion of source text during parsing |
| **2. Aligned Vector Indexing** | Positional 1:1 mapping between FAISS vector ID and `EvidenceSegment` | Metadata detachment or context loss |
| **3. Isolated Retrieval** | `search_by_expert()` filter masks | Cross-market leakage (e.g., German procurement opinions attributed to French clinicians) |
| **4. Balanced Allocation** | `search_per_expert()` equal quota allocation | Single-expert dominance in cross-transcript questions |
| **5. Closed-World Prompting** | Strict systemic prompt forbidding external knowledge | LLM relying on pre-trained parametric world knowledge |
| **6. Grounded Fallback** | Automated fallback triggered when evidence is missing | Fabricated statistics, percentages, market sizes, or dates |
| **7. Structural Citation Isolation** | UI citation cards rendered directly from `EvidenceSegment.text` | LLM hallucinating fake quotes, fake timestamps, or altered wording |
| **8. Automated Verification** | 15-case benchmark (`scripts/evaluate.py`) + 77 pytest unit tests | Undetected regression in grounding or attribution |

### Verified Evaluation Results
- **Golden Benchmark Pass Rate:** **15 / 15 passed (100.0%)**
- **Automated Pytest Suite:** **77 / 77 passed (100.0%)**
- **Source Attribution:** 100% verbatim character match against raw transcripts.
- **Milestone Timestamps Verified:** France (`01:20`, `02:18`, `06:08`), Germany (`02:08`, `06:05`), UK (`02:07`, `05:04`).

---

## 6. Scaling Strategy: From 3 to 30+ Transcripts

### 6.1 Current Architecture Baseline
- **Scale:** 3 transcripts, 42 evidence segments, 1 project.
- **Storage:** In-process FAISS `IndexFlatIP` + local JSON metadata.
- **Retrieval:** Linear in-memory filtering by expert name.
- **Synthesis:** Direct prompt context packing (under 3,000 tokens).

### 6.2 Scaling Workflow for 30+ Transcripts
When scaling from 3 to 30+ transcripts (approx. 500–1,500 evidence segments across multiple projects and markets):

```text
30+ Raw Transcripts (.txt / .json / .vtt)
               ↓
Parallelized Deterministic Parser (Celery / Redis Queue)
               ↓
Normalized EvidenceSegments with UUIDs & Project Metadata
               ↓
Batch Embedding Generation (sentence-transformers / GPU batching)
               ↓
Persistent Vector Database (Qdrant / Milvus / pgvector)
               ↓
Hybrid Search: Dense Vectors (HNSW) + Sparse BM25 Keywords
               ↓
Cross-Encoder Reranker (bge-reranker-large)
               ↓
Partitioned Retrieval (Filter by Project, Country, Specialty)
               ↓
Top-K Balanced Evidence Context
               ↓
Groq LLM Synthesis + Verifiable Source Citations
```

### 6.3 Implementation vs. Production Scaling Comparison

| Dimension | Current Implementation (Case Study) | Potential Production Scaling (30+ Calls) |
| :--- | :--- | :--- |
| **Vector Storage** | Local FAISS `IndexFlatIP` file (`index.faiss`) | Managed distributed vector DB (Qdrant, Milvus, pgvector) |
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

---

## 7. Requirement-to-Implementation Traceability Matrix

| Case Study Requirement | Architecture Implementation | Verification / Test Evidence | Status |
| :--- | :--- | :--- | :--- |
| **1. Upload/read 3 transcripts** | `src/parser.py` deterministically parses France, Germany, and UK transcripts into 42 `EvidenceSegment` objects | `tests/test_parser.py` (14 tests passing); `python -m src.parser` | **COMPLETE** |
| **2. Answer interview-guide questions for each expert** | `src/interview_guide.py` defines the 6 official questions; `src/analysis.py` answers each per expert with isolated retrieval | `tests/test_analysis.py` (15 tests passing); `GOLDEN-01`, `GOLDEN-02` | **COMPLETE** |
| **3. Extract useful exact quotes** | `src/models.py` preserves verbatim dialogue; `src/ui/common.py` renders quotes strictly from `EvidenceSegment.text` | `tests/test_rag.py::test_source_text_identical_to_original_segment`; `GOLDEN-01` to `07` | **COMPLETE** |
| **4. Show source timestamp for each answer/quote** | `EvidenceSegment.timestamp` preserved during parsing and displayed on every answer citation card | `tests/test_parser.py::test_timestamps_extracted`; `GOLDEN-04` (`06:05`) | **COMPLETE** |
| **5. Identify common themes and disagreements** | `compare_experts()` in `src/analysis.py` synthesizes consensus themes and market-attributed differences | `tests/test_analysis.py::test_cross_expert_analysis_includes_all_experts`; `GOLDEN-03`, `05`, `06`, `07` | **COMPLETE** |
| **6. Allow questions across all transcripts** | `answer_question()` in `src/rag.py` and *Ask Across Calls* UI page enable arbitrary cross-call natural language queries | `tests/test_rag.py` (12 tests passing); `app.py` Ask page | **COMPLETE** |
