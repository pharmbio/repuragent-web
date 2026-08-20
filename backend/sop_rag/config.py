'''Paths and settings for the SOP retrieval-augmented index.

Source PDFs live under `DATA_ROOT/SOP`; the built index (Chroma vectors plus a
file-backed docstore of the original chunks) lives under
`MEMORY_ROOT/sop_documents` and is shipped with the repository, so a fresh clone
can answer SOP questions without re-indexing.
'''

from __future__ import annotations

from pathlib import Path

from backend.utils.storage_paths import get_data_root, get_memory_root

DATA_DIR = get_data_root()
MEMORY_DIR = get_memory_root()

SOP_DATA_DIR = DATA_DIR / "SOP"
SOP_MEMORY_DIR = MEMORY_DIR / "sop_documents"
CHROMA_PERSIST_PATH = SOP_MEMORY_DIR / "chroma_db" / "sop_rag"
DOCSTORE_PATH = SOP_MEMORY_DIR / "docstore.pkl"
DOCSTORE_DIR = SOP_MEMORY_DIR / "docstore"

COLLECTION_NAME = "sop_rag"
ID_KEY = "doc_id"

EMBEDDING_MODEL = "text-embedding-3-small"

# `unstructured` hi-res partitioning: expensive, but it is what keeps tables and
# figures out of the text chunks. Only used when rebuilding the index.
PDF_PROCESSING_CONFIG = {
    "strategy": "hi_res",
    "infer_table_structure": True,
    "extract_image_block_types": ["Image"],
    "extract_image_block_to_payload": True,
    "chunking_strategy": "by_title",
    "max_characters": 10000,
    "combine_text_under_n_chars": 2000,
    "new_after_n_chars": 6000,
}

LLM_CONFIG = {
    "summarization_model": "gpt-5-mini-2025-08-07",
    "image_description_model": "gpt-5-mini-2025-08-07",
    "rag_response_model": "gpt-5-mini-2025-08-07",
}

RETRIEVAL_CONFIG = {
    "default_k": 4,
    "search_type": "similarity",
}


def ensure_directories() -> None:
    for directory in (SOP_DATA_DIR, SOP_MEMORY_DIR, CHROMA_PERSIST_PATH):
        directory.mkdir(parents=True, exist_ok=True)


def index_exists() -> bool:
    '''True when both halves of the index are present on disk.

    Returns:
    ----------
    present (boolean): True when both the vector store and the docstore are on disk.
    '''

    return Path(CHROMA_PERSIST_PATH).exists() and Path(DOCSTORE_DIR).exists()
