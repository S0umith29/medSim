from typing import List, Dict, Any, Optional
import random

import chromadb
from chromadb.config import Settings

from src.config import PERSIST_DIR, COLLECTION_NAME, DEFAULT_TOP_K


class ChromaRetriever:
    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection_name)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, pmc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        where = {"pmc_id": pmc_id} if pmc_id else None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
            where=where,
        )
        docs: List[Dict[str, Any]] = []
        if not results or not results.get("documents"):
            return docs
        docs_list = results["documents"][0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]
        for text, meta, dist in zip(docs_list, metas_list, dists_list):
            docs.append({
                "text": text,
                "pmc_id": meta.get("pmc_id"),
                "chunk_index": meta.get("chunk_index"),
                "score": float(dist),
            })
        return docs

    def sample_pmc_id(self, sample_limit: int = 1000) -> Optional[str]:
        """
        Sample a random case by selecting a random metadata row among first-chunk entries.
        """
        try:
            res = self.collection.get(where={"chunk_index": 0}, include=["metadatas"], limit=sample_limit)
            metas = res.get("metadatas") or []
            if not metas:
                return None
            flat = metas  # already a flat list per chroma get
            candidates = [m.get("pmc_id") for m in flat if m and m.get("pmc_id")]
            if not candidates:
                return None
            return random.choice(candidates)
        except Exception:
            return None
