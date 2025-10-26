from typing import Iterable, Dict, Any, Optional
import os

from datasets import load_dataset

from src.config import HF_DATASET, HF_SPLIT


def load_pmc_dataset(limit: Optional[int] = None, hf_token: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """
    Stream rows from the PMC-CaseReport dataset. We read fields needed for RAG:
    - PMC_id (string)
    - context (case report text)

    Optionally limit to first N rows for quick prototyping.
    """
    token = hf_token or os.getenv("HF_TOKEN") or None
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, token=token)

    count = 0
    for row in ds:  # Hugging Face returns dict-like rows
        pmc_id = row.get("PMC_id")
        context = row.get("context")
        if not pmc_id or not context:
            continue
        yield {
            "pmc_id": str(pmc_id),
            "context": str(context),
        }
        count += 1
        if limit is not None and count >= limit:
            break
