import os

# Storage / Chroma
PERSIST_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pmc_casereport")

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Dataset
HF_DATASET = os.getenv("HF_DATASET", "chaoyi-wu/PMC-CaseReport")
HF_SPLIT = os.getenv("HF_SPLIT", "train")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))          # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))     # characters

# Retrieval / Generation defaults
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Ollama
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You are a helpful virtual patient simulator for medical students. "
    "Use the provided medical case snippets to answer the user's question. "
    "Cite sources as [PMC_id]. If the answer is uncertain or missing, explicitly say you don't know."
)

# Ethics / HIPAA policy (applied in all prompts)
ETHICS_POLICY = (
    "Ethics/HIPAA: Use professional, respectful, non-discriminatory language. "
    "Follow minimum-necessary PHI: avoid requesting or revealing unnecessary identifiers (e.g., SSN, exact address, financial IDs). "
    "Do not disclose third-party PHI. Ask sensitive topics only when clinically relevant and with consent. "
    "Avoid leading/coercive questions. If unsure, ask for clarification rather than inventing details."
)
