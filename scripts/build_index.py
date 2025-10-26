#!/usr/bin/env python
import argparse
import os

from src.rag.dataset_loader import load_pmc_dataset
from src.rag.indexer import ChromaIndexer
from src.config import PERSIST_DIR, COLLECTION_NAME


def main():
    parser = argparse.ArgumentParser(description="Build Chroma index from PMC-CaseReport")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset rows")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    args = parser.parse_args()

    indexer = ChromaIndexer(persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME)

    if args.reset:
        print("Resetting collection…")
        indexer.reset_collection()

    print("Loading dataset…")
    rows = load_pmc_dataset(limit=args.limit)

    print("Indexing…")
    indexer.add_documents(rows)

    print("Done. Index stored at:", os.path.abspath(PERSIST_DIR))


if __name__ == "__main__":
    main()
