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

## 📁 Initial Project Structure

```text
expert-call-ai/
├── app.py              # Streamlit entrypoint
├── requirements.txt    # Project dependencies
├── README.md           # Documentation & case study details
├── .gitignore          # Git ignore rules
├── .env.example        # Environment variable template
├── data/               # Raw and processed transcript storage
├── src/                # Core application logic and modules
│   └── __init__.py
└── tests/              # Automated unit and integration tests
    └── __init__.py
```

---

## ⚙️ Initial Setup & Run Instructions

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
# Edit .env and configure the required API keys as needed
```

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the application.
