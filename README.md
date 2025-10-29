# medSim – Virtual Patient Prototype

![It Works](assets/It%20Works.png)

A simple virtual patient simulator for medical students. Uses RAG over the PMC-CaseReport dataset, ChromaDB as a vector store, sentence-transformers embeddings, and Ollama running `llama3.2:3b` for responses. UI built with Streamlit.

Dataset: https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport

## Features
- RAG with ChromaDB (persistent local store)
- Sentence-transformers embeddings (`all-MiniLM-L6-v2` by default)
- Local LLM via Ollama (`llama3.2:3b`)
- Streamlit chat interface with citations
- Virtual Patient mode 
- Ethics/HIPAA policy applied to prompts
- Simple CLI to build or refresh the index

## Quickstart

1) Install system prerequisites
- Python 3.10+
- Install Ollama (https://ollama.com) and pull the model:

```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest
ollama pull llama3.2:3b
```

2) Create a virtual environment and install Python dependencies

```bash
cd medSim
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3) Configure environment (optional)

```bash
cp .env.example .env
# Edit values if needed
```

4) Build the RAG index (you can limit rows to speed up)

```bash
python scripts/build_index.py --limit 5000
```

5) Run the Streamlit app

```bash
# Ensure imports resolve
export PYTHONPATH=$PWD

streamlit run src/app/streamlit_app.py
# or headless/network:
# streamlit run src/app/streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

## Project Structure

```
MedSimuli/
  ├─ data/
  │   └─ chroma/                # ChromaDB persistent store (created on first index build)
  ├─ scripts/
  │   └─ build_index.py         # CLI for building the index
  └─ src/
      ├─ app/
      │   └─ streamlit_app.py   # Streamlit chat UI
      ├─ rag/
      │   ├─ dataset_loader.py  # Load PMC-CaseReport from HF
      │   ├─ chunker.py         # Text chunking utilities
      │   ├─ indexer.py         # Chroma index builder
      │   ├─ retriever.py       # Chroma retriever
      │   ├─ llm.py             # Ollama LLM wrapper
      │   └─ evaluator.py       # Rubric-based evaluator 
      └─ config.py              # Config and constants
```

## Using the app
- In the sidebar, choose Mode:
  - Study (RAG QA): ask general questions; answers cite `[PMC_id]` sources
  - Virtual Patient: click “New patient” to sample a case; ask first‑person history questions
- The sidebar also shows a read‑only Ethics/HIPAA policy in effect.

## Notes
- This is a prototype for demonstration only; not clinical or production-grade.
- The dataset includes generated QA pairs; we primarily index the case `context` for retrieval. The model is instructed to cite sources as `[PMC_id]`.
- If you get missing model errors, ensure `ollama serve` is running and the `llama3.2:3b` model is available.

### Security/Ethics
- Prompts include a concise Ethics/HIPAA policy (minimum‑necessary PHI, respectful language, no third‑party PHI).
- Evaluation module exists (`src/rag/evaluator.py`).

## Dataset Citation
- PMC-CaseReport dataset by Chaoyi Wu et al.: https://huggingface.co/datasets/chaoyi-wu/PMC-CaseReport
