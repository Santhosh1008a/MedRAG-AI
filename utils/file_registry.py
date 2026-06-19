"""
utils/file_registry.py
======================
Provides SHA-256 based duplicate-PDF detection for the ingestion pipeline.

A JSON registry is persisted to disk so that already-indexed documents are
remembered across application restarts.  Only the ingestion path reads or
writes this module; retrieval, reranking, and the LLM are unaffected.
"""

import hashlib
import json
import os
from typing import Dict, Any

# Default location for the on-disk registry
DEFAULT_REGISTRY_PATH = "./data/processed_files.json"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: str) -> str:
    """
    Return the SHA-256 hex digest of the file at *file_path*.

    Reads in 8 KB chunks so large PDFs do not exhaust memory.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def load_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    """
    Load the processed-files registry from *registry_path*.

    Returns an empty dict if the file does not yet exist.

    Registry schema::

        {
            "<sha256_hex>": {
                "filename": "<original filename>",
                "num_chunks": <int>
            },
            ...
        }
    """
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry: Dict[str, Any], registry_path: str = DEFAULT_REGISTRY_PATH) -> None:
    """Persist *registry* to *registry_path* as pretty-printed JSON."""
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_file(
    registry: Dict[str, Any],
    file_hash: str,
    filename: str,
    num_chunks: int,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> None:
    """
    Add *file_hash* → metadata to *registry* and immediately persist to disk.

    Args:
        registry:      The in-memory registry dict (mutated in-place).
        file_hash:     SHA-256 hex digest of the file.
        filename:      Human-readable filename for logging / UI feedback.
        num_chunks:    Number of chunks produced during chunking.
        registry_path: Where to persist the JSON registry.
    """
    registry[file_hash] = {
        "filename": filename,
        "num_chunks": num_chunks,
    }
    save_registry(registry, registry_path)


def is_duplicate(registry: Dict[str, Any], file_hash: str) -> bool:
    """Return *True* if *file_hash* is already present in *registry*."""
    return file_hash in registry


def clear_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> None:
    """
    Delete the on-disk registry file and reset state.

    Called when the user clears the FAISS index so that previously-indexed
    hashes are no longer treated as duplicates.
    """
    if os.path.exists(registry_path):
        os.remove(registry_path)
