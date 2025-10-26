from typing import Iterable, Dict, Any
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.rag.chunker import split_text


class ChromaIndexer:
    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def reset_collection(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def add_documents(self, rows: Iterable[Dict[str, Any]]):
        batch_texts = []
        batch_ids = []
        batch_metadatas = []

        row_counter = 0
        for row in rows:
            pmc_id = row["pmc_id"]
            context = row["context"]
            chunks = split_text(context, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{pmc_id}-{row_counter}-{idx}"
                batch_texts.append(chunk)
                batch_ids.append(doc_id)
                batch_metadatas.append({"pmc_id": pmc_id, "chunk_index": idx, "row_index": row_counter})

            row_counter += 1

            # flush in moderate batches to avoid memory spikes
            if len(batch_texts) >= 256:
                self._flush(batch_texts, batch_ids, batch_metadatas)
                batch_texts, batch_ids, batch_metadatas = [], [], []

        if batch_texts:
            self._flush(batch_texts, batch_ids, batch_metadatas)

    def _flush(self, texts, ids, metadatas):
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)
