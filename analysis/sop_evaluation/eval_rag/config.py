from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EVAL_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `unstructured` and `chromadb` both want writable caches; keep them out of the
# repository the way the production indexer does.
TMP_ROOT = Path(gettempdir()) / "repuragent-sop-eval"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_ROOT / "xdg-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(TMP_ROOT / "numba-cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Same reason the app turns gradio telemetry off: this handles unpublished
# research data, and the posthog client in chromadb 0.6 raises on every event.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

SOP_DIR = EVAL_DIR / "SOPs"
QUESTIONS_PATH = EVAL_DIR / "questions.csv"

CACHE_DIR = EVAL_DIR / "cache"
INDEX_DIR = EVAL_DIR / "index"
RESULTS_DIR = EVAL_DIR / "results"
FIGURES_DIR = EVAL_DIR / "figures"

PARENTS_CACHE_PATH = CACHE_DIR / "parents.json"
QUERY_EMBEDDING_DIR = CACHE_DIR / "query_embeddings"


def ensure_directories() -> None:
    '''Create every directory this evaluation writes into.'''
    for directory in (CACHE_DIR, INDEX_DIR, RESULTS_DIR, FIGURES_DIR, QUERY_EMBEDDING_DIR):
        directory.mkdir(parents=True, exist_ok=True)



DEFAULT_SETTINGS = {
    "kind": "parent_child",
    "embedding_model": "text-embedding-3-large",
    "chunk_size": 400,
    "chunk_overlap": 50,
    "search_type": "similarity",
    "bm25_weight": 0.5,
    "dense_weight": 0.5,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "fetch_k": 50,
    "fuse_func": "rrf",
    "max_results": 5,
    "rrf_c": 60,
}

# FETCH_K_MAX bounds every cached candidate list, so any `fetch_k <=
# FETCH_K_MAX` is answered by slicing rather than by a second retrieval pass.
FETCH_K_MAX = 100
CHILD_POOL = FETCH_K_MAX * 4

# Fuzzy match threshold for "the gold snippet is inside this passage".
EVIDENCE_MATCH_THRESHOLD = 90


def canonical_source(name: Optional[str]) -> str:
    '''Filename in a form that compares equal across questions.csv and disk.'''

    text = str(name or "").strip().rsplit("/", 1)[-1]
    return text.replace(" ", "_").lower()


@dataclass(frozen=True)
class IndexSpec:
    '''One built index.

    Parameters:
    ---------
    kind (str): `parent_child` (sections stored, children embedded) or `basic`
        (flat chunks stored and embedded, so a hit returns the chunk itself).
    embedding_model (str): the OpenAI embedding model for the dense arm.
    chunk_size (int): child chunk size for `parent_child`; chunk size for `basic`.
    chunk_overlap (int): overlap for the same splitter.
    source_prefix (boolean): keep the `[Source: <filename>]` line the production
        indexer writes at the top of every parent. False is the ablation.
    '''

    kind: str = "parent_child"
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 400
    chunk_overlap: int = 50
    source_prefix: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("basic", "parent_child"):
            raise ValueError(f"Unknown index kind '{self.kind}'.")

    @property
    def slug(self) -> str:
        model = self.embedding_model.replace("text-embedding-3-", "")
        parts = [self.kind, model, f"c{self.chunk_size}", f"o{self.chunk_overlap}"]
        if not self.source_prefix:
            parts.append("nosrc")
        return "_".join(parts)

    @property
    def root(self) -> Path:
        return INDEX_DIR / self.slug

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma_db"

    @property
    def docstore_path(self) -> Path:
        return self.root / "docstore.json"

    @property
    def bm25_corpus_path(self) -> Path:
        return self.root / "bm25_corpus.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def collection_name(self) -> str:
        # The embedding model and child size are part of the collection name for
        # the same reason as in production: two embedding spaces must never end
        # up in one collection.
        return f"sop_eval_{self.slug}"


DEFAULT_SPEC = IndexSpec(
    kind="parent_child",
    embedding_model=DEFAULT_SETTINGS["embedding_model"],
    chunk_size=DEFAULT_SETTINGS["chunk_size"],
    chunk_overlap=DEFAULT_SETTINGS["chunk_overlap"],
)

# The flat-chunk baseline. 1000/200 is the ordinary RecursiveCharacterTextSplitter
# default territory, and is what "basic RAG" means here: one embedding per chunk,
# and the chunk itself is what a hit returns.
BASIC_SPEC = IndexSpec(
    kind="basic",
    embedding_model=DEFAULT_SETTINGS["embedding_model"],
    chunk_size=1000,
    chunk_overlap=200,
)

__all__ = [
    "BASIC_SPEC",
    "CHILD_POOL",
    "DEFAULT_SETTINGS",
    "DEFAULT_SPEC",
    "EVIDENCE_MATCH_THRESHOLD",
    "FETCH_K_MAX",
    "FIGURES_DIR",
    "INDEX_DIR",
    "IndexSpec",
    "PARENTS_CACHE_PATH",
    "QUERY_EMBEDDING_DIR",
    "QUESTIONS_PATH",
    "RESULTS_DIR",
    "SOP_DIR",
    "canonical_source",
    "ensure_directories",
]
