# ExpertCall AI — Interview Demonstration Guide & Technical Talking Points

> **Hasamex AI Engineer Technical Case Study**  
> **Repository:** [https://github.com/Roushan0012/Expert-Call-Ai](https://github.com/Roushan0012/Expert-Call-Ai)  
> **Duration:** 5–7 Minute Live Demo Script + Technical Q&A Cheat Sheet

---

## Part 1: Recommended 5–7 Minute Live Demonstration Script

```text
[0:00 - 0:30] Problem Context & Objective
[0:30 - 1:15] System Architecture & Grounding Guarantees
[1:15 - 2:15] Overview Dashboard & Ingested Transcripts
[2:15 - 3:15] Interview Guide: The 6 Official Questions
[3:15 - 4:00] Timestamp Traceability & Exact Quote Verification
[4:00 - 4:45] Cross-Expert Synthesis: Consensus vs. Disagreements
[4:45 - 5:30] Ask Across Calls: Grounded Cross-Transcript Q&A
[5:30 - 6:15] Anti-Hallucination Guardrails & Insufficient Evidence Handling
[6:15 - 7:00] Scaling to 30+ Transcripts & Production Readiness
```

---

### [0:00 – 0:30] Introduce the Problem
- **Pitch:**  
  *"When healthcare strategy consultants and procurement teams conduct expert calls on medical devices (like surgical robotics), they face two core operational bottlenecks: First, manually extracting answers to standard interview questions across multiple markets is tedious. Second, standard AI chatbots hallucinate quotes, mix up experts, or invent statistics."*
- **Solution:**  
  *"ExpertCall AI solves this with a deterministic parsing and grounded RAG architecture that analyzes 3 commercial transcripts from France, Germany, and the UK. Every insight is traceable to exact dialogue turns, exact timestamps, and verified verbatim quotes with 100% anti-hallucination validation."*

---

### [0:30 – 1:15] Show the Architecture
- **Navigate to:** `docs/ARCHITECTURE.md` or the diagram on the **Overview** dashboard.
- **Key Talking Points:**
  - *"We deliberately separated retrieval, generation, and citation rendering."*
  - *"1. **Local Parsing & Embeddings:** Transcripts are parsed deterministically into 42 immutable `EvidenceSegment` units. Dense vectors are generated locally using `all-MiniLM-L6-v2` (no third-party API dependencies or data leakage)."*
  - *"2. **FAISS Vector Index:** We use `IndexFlatIP` with L2 normalization, yielding exact cosine similarity retrieval in microseconds."*
  - *"3. **Groq LPU Synthesis:** Answers are synthesized rapidly using `qwen/qwen3.8-27b` under strict closed-world prompt guardrails."*
  - *"4. **Quote Safety Guardrail:** The LLM generates the analytical narrative, but is NEVER allowed to invent quotes. All citation cards in the UI pull directly from raw transcript data structures in memory."*

---

### [1:15 – 2:15] Overview Dashboard
- **Action:** Open `http://localhost:8501`, stay on the **📊 Overview** page.
- **Showcase:**
  - Dynamic system metrics: **3 Transcripts**, **3 Experts**, **42 Evidence Segments**, **384-dim Embeddings**, **FAISS IndexFlatIP**, **Groq LLM**.
  - Market Expert Cards:
    - **France:** Dr. Jean Martin (Head of Urology, Academic Medical Center)
    - **Germany:** Anna Keller (Former Hospital Procurement Director, Private Hospital Group)
    - **United Kingdom:** Dr. Emily Carter (Consultant Urologist, NHS Foundation Trust)
  - Highlight how diverse perspectives (clinical department head vs. procurement director vs. NHS consultant) provide balanced commercial insights.

---

### [2:15 – 3:15] Interview Guide (Official 6 Questions)
- **Action:** Click on **📋 Interview Guide** in the sidebar.
- **Showcase:**
  - Point out that the system exposes the **exact 6 official case study questions**:
    1. *Current adoption of robotic surgery*
    2. *Main barriers to adoption*
    3. *Hospital budgets and ROI*
    4. *Surgeon training and clinical outcomes*
    5. *3–5 year adoption trend*
    6. *Hospital decision-making timeline*
  - Select **Question 6:** *"What is the typical hospital decision-making timeline for purchasing a new robotic system?"*
  - Set Expert to **All Experts** and click **Analyze Question Across Experts**.
  - Highlight the 3 side-by-side market cards:
    - **France:** 6 to 12 months (driven by clinical board and administrative approvals).
    - **Germany:** 9 to 18 months (protracted procurement board reviews and capex cycles).
    - **United Kingdom:** 6 to 9 months (business case approval within NHS trust committees).

---

### [3:15 – 4:00] Timestamp Traceability & Exact Quotes
- **Action:** Expand the **Verified Source Citations** expander beneath the Germany answer.
- **Showcase:**
  - Show the milestone timestamp **`06:05`** for Anna Keller.
  - Show the verbatim transcript quote:
    > *"Nine to eighteen months is common. In private groups, it can be slightly faster if there is strong leadership support, but you still need formal business cases, surgeon consensus, and board approval."*
  - **Key Talking Point:**  
    *"Notice how the word 'timeline' does not literally appear in Anna Keller's spoken words at 06:05. Our retrieval layer pairs question-answer dialogue turns and applies targeted query expansions, retrieving the exact ground-truth evidence at rank 1 with 100% precision."*

---

### [4:00 – 4:45] Cross-Expert Analysis (Themes vs. Divergences)
- **Action:** Click on **⚖️ Cross-Expert Analysis** in the sidebar.
- **Showcase:**
  - Select preset topic: *"Hospital Budgets & ROI"* or *"Purchasing Timelines"*.
  - Point out the two structured output containers:
    - **Common Themes (Consensus):** High initial capital expenditure and annual maintenance contracts require detailed multi-year procedure volume justification across all three healthcare systems.
    - **Market Disagreements & Nuances:**
      - France: Emphasizes department head influence and clinical utilization metrics.
      - Germany: Strictly governed by multi-stakeholder procurement committees where CFOs and capex limits dictate timing.
      - UK: Bound by rigid NHS Trust capital allocation ceilings and public tender requirements.
  - **Key Talking Point:**  
    *"The RAG layer enforces balanced retrieval using `search_per_expert(k=2)`. This prevents one market from crowding out the others."*

---

### [4:45 – 5:30] Ask Across Calls (Interactive Grounded Q&A)
- **Action:** Click on **💬 Ask Across Calls** in the sidebar.
- **Showcase:**
  - Type or click preset query: *"How does surgeon training affect the adoption of robotic surgery across markets?"*
  - Click **Search & Generate Answer**.
  - Show how the synthesis brings together Dr. Martin (France `04:02`), Anna Keller (Germany `04:06`), and Dr. Carter (UK `03:45`), backed by source cards with exact timestamps and roles.

---

### [5:30 – 6:15] Anti-Hallucination & Insufficient Evidence Guardrails
- **Action:** In **Ask Across Calls**, enter an unsupported query:
  - Query: *"What is the exact market share of Intuitive Surgical in euros across Europe?"*
  - Click **Search & Generate Answer**.
  - Show the clean grounded fallback:
    > *"The provided transcripts do not contain enough evidence to answer this question."*
  - **Key Talking Point:**  
    *"Notice that the system did not invent market share percentages or fabricate euro revenue figures. Our closed-world prompt and insufficient-evidence fallback protect users from parametric hallucinations."*
  - Mention the evaluation benchmark:
    *"We validated this across 15 golden test cases and 77 unit tests, achieving a 100% pass rate."*

---

### [6:15 – 7:00] Scaling to 30+ Transcripts & Wrap-Up
- **Action:** Summarize the path to enterprise production.
- **Key Talking Points:**
  - *"To scale this from 3 to 30+ transcripts, we maintain our core domain models (`EvidenceSegment`) but upgrade three infrastructure layers:"*
    - *"1. **Distributed Vector Database:** Migrate from local FAISS `IndexFlatIP` to Qdrant or Milvus with native metadata payload filtering."*
    - *"2. **Hybrid Search & Reranking:** Pair dense embeddings with sparse BM25 and a cross-encoder reranker (`bge-reranker-large`)."*
    - *"3. **Async Ingestion Pipeline:** Decouple transcript ingestion with Celery/Redis background workers and batch GPU embedding generation."*
  - *"Thank you! I'm happy to answer any technical questions about the architecture or implementation."*

---

## Part 2: Technical Interview Q&A Talking Points

### 1. Why FAISS?
- **Answer:**  
  *FAISS (Facebook AI Similarity Search) is an ultra-fast, C++-optimized vector library with minimal dependencies. For our 3-transcript dataset (42 segments), FAISS `IndexFlatIP` provides exact, exhaustive nearest-neighbor search with zero approximation error. It saves directly to disk (`index.faiss`) without requiring an external database process or port.*

### 2. Why local embeddings instead of OpenAI / Cohere embeddings?
- **Answer:**  
  *Three reasons: (1) **Data privacy:** Healthcare and commercial procurement expert calls often contain confidential or proprietary information; running locally guarantees zero transcript data leaves the machine during embedding. (2) **Zero API latency/costs:** Eliminates network overhead, rate limits, and external service failure modes. (3) **100% Offline testability:** Our entire 77-test suite runs deterministically offline in 14 seconds.*

### 3. Why `sentence-transformers/all-MiniLM-L6-v2`?
- **Answer:**  
  *It provides an ideal trade-off for short-to-medium conversational turns. At 22.7M parameters (~80MB), it executes in milliseconds on standard laptop CPUs while generating dense 384-dimensional vectors that capture semantic similarity with high fidelity on MTEB benchmarks.*

### 4. Why Groq?
- **Answer:**  
  *Groq's Language Processing Unit (LPU) architecture delivers extraordinary inference speed (~300 tokens/sec), allowing complex multi-segment cross-transcript comparative analyses to complete in under 2 seconds. The LLM is used strictly for linguistic synthesis, not as the ground-truth store.*

### 5. Why RAG instead of passing all transcripts directly into a long-context LLM?
- **Answer:**  
  *While 3 transcripts could fit in a 128k context window, doing so has major flaws: (1) **Context crowding & attention degradation:** LLMs struggle to recall exact details from the middle of large contexts ("lost in the middle"). (2) **Expert conflation:** Long prompts frequently lead the model to merge different experts' viewpoints. (3) **Cost and Latency:** Passing entire transcripts for every single query is slow and expensive. (4) **Scalability:** Passing all text breaks completely when scaling to 30 or 100 transcripts. RAG provides deterministic, inspectable retrieval.*

### 6. How do you prevent hallucinations?
- **Answer:**  
  *We use defense-in-depth across 8 stages: deterministic parsing, immutable segment models, isolated vector retrieval, balanced quota allocation per expert, strict closed-world prompt instructions, automated insufficient-evidence fallbacks, structural citation separation (UI quotes come from `EvidenceSegment`, not the LLM), and automated benchmark testing.*

### 7. How do you guarantee quote accuracy?
- **Answer:**  
  *We structurally eliminate LLM quote generation. When the UI renders a quote card, it reads directly from `EvidenceSegment.text` in memory. The LLM is never prompted or trusted to generate verbatim quotes.*

### 8. How are timestamps preserved?
- **Answer:**  
  *During parsing, the deterministic regex extracts the `[MM:SS]` timestamp from each speaker turn and binds it permanently to the `EvidenceSegment.timestamp` field. When evidence is retrieved, its timestamp is passed alongside it and displayed in verified citation containers.*

### 9. How do you prevent expert leakage (cross-contamination)?
- **Answer:**  
  *When answering single-expert questions (e.g. France / Dr. Jean Martin), we use `vector_store.search_by_expert(query, expert=...)` which applies an in-memory filter mask before scoring. Segments from Germany or the UK are physically excluded from the retrieval candidate set.*

### 10. How does cross-expert comparison work without bias?
- **Answer:**  
  *Standard similarity search can allow a verbose expert or high-frequency keyword section to dominate top-k results. We built `search_per_expert(query, k_per_expert=2)` which retrieves exactly 2 segments per expert across all 3 countries (6 total). The LLM is then prompted with clearly delineated expert sections and instructed to identify common themes and specific attributed differences.*

### 11. What happens when evidence is insufficient?
- **Answer:**  
  *If a query asks about topics not covered in the transcripts (such as market size in euros or manufacturer market share), the system returns a standardized fallback: 'The provided transcripts do not contain enough evidence to answer this question.' Our evaluation suite specifically tests 5 insufficient-evidence queries to ensure no figures or percentages are fabricated.*

### 12. How would you scale this system from 3 to 30+ transcripts?
- **Answer:**  
  *We would maintain the same domain boundaries while replacing the storage and ingestion layer:
  1. Store vectors in a distributed database like Qdrant or Milvus with payload filtering for `project_id`, `market`, and `expert_id`.
  2. Implement hybrid search combining dense HNSW embeddings with sparse BM25, followed by a cross-encoder reranker.
  3. Move ingestion to an asynchronous Celery/Redis queue with batch GPU embedding generation.
  4. Add semantic caching in Redis to prevent redundant LLM generations for identical queries.*

### 13. What would you improve before deploying to enterprise production?
- **Answer:**  
  *1. **Security & Access Control:** Add OAuth2/SSO with role-based access control (RBAC) so analysts only see transcripts for their assigned projects.  
  2. **LLM Observability:** Instrument tracing via OpenTelemetry and TruLens/LangSmith to track latency, token usage, and grounding metrics.  
  3. **Audio-to-Text Pipeline:** Integrate Whisper or Deepgram with diarization to process raw MP3/WAV audio recordings into timestamped transcripts automatically.  
  4. **Document Export:** Add PDF/Word report generation for executive summaries.*
