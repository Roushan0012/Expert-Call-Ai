# ExpertCall AI

> **Note:** This repository is part of the **Hasamex AI Engineer Technical Case Study**.

## 📌 Project Objective

**ExpertCall AI** is an AI application designed to ingest and analyze 3 expert-call transcripts from the same project. The application extracts answers to 6 core interview-guide questions for each expert, verifies answers with verbatim quotes and exact source timestamps, synthesizes common themes and conflicting viewpoints across all experts, and provides an interactive question-answering interface grounded strictly in transcript evidence to eliminate hallucinations.

---

## 🚀 Planned Features

1. **Transcript Ingestion & Parsing:**
   - Structured parsing and segmenting of the 3 project expert-call transcripts with speaker and timestamp tracking.
2. **Interview-Guide Question Answering:**
   - Automated answering of 6 standard interview-guide questions for each expert transcript.
3. **Traceability & Evidence Verification:**
   - Extraction of exact verbatim quotes paired with source timestamps for every key insight and answer.
4. **Cross-Expert Synthesis:**
   - Identification of consensus themes, divergent opinions, and contrasting viewpoints across all 3 experts.
5. **Interactive Cross-Transcript Q&A:**
   - Natural language search and chat querying across all transcripts with source citations.
6. **Anti-Hallucination Guardrails:**
   - Strict retrieval-grounded synthesis ensuring assertions are traceable to transcript evidence.
7. **Streamlit UI:**
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

### 4. Importance of Deterministic Parsing for Traceability
In high-stakes technical and clinical analyses, AI systems must never hallucinate evidence. Deterministic, rule-based parsing creates immutable evidence records linked to exact timestamps. When downstream systems surface a quote or assertion, it directly maps back to a verifiable line in the original transcript.

> **Note on LLM Integration:** Groq (`GROQ_API_KEY`) is selected as the LLM provider for downstream retrieval-augmented generation (RAG) and question answering. In this step, only deterministic ingestion and parsing are implemented; no external LLM API calls or vector embeddings are performed.

---

## 📁 Project Structure

```text
expert-call-ai/
├── app.py                      # Streamlit entrypoint
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation & case study details
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment variable template (GROQ_API_KEY)
├── data/                       # Case study transcript files
│   ├── .gitkeep
│   ├── Transcript_1_France.txt
│   ├── Transcript_2_Germany.txt
│   └── Transcript_3_UK.txt
├── src/                        # Core application logic and modules
│   ├── __init__.py
│   ├── models.py               # EvidenceSegment & Transcript data models
│   └── parser.py               # Deterministic transcript parser & CLI inspection
└── tests/                      # Automated test suite
    ├── __init__.py
    └── test_parser.py          # Pytest suite for transcript parser
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
To run the built-in parser inspection tool and verify parsed evidence segments:
```bash
python -m src.parser
```

### 6. Run Automated Tests
```bash
pytest -v
```

### 7. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the application.
