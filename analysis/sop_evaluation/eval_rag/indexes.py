from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid5, NAMESPACE_URL

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval_rag.config import IndexSpec, ensure_directories
from eval_rag.parsing import SOURCE_PREFIX_TEMPLATE, load_sections, sections_to_documents

MANIFEST_VERSION = 1
EMBED_BATCH = 256


def get_embeddings(model: str) -> OpenAIEmbeddings:
    '''The embedding function for `model`.'''
    from app.config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; the dense arm cannot be built.")
    return OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)


def _stable_id(spec: IndexSpec, position: int, text: str) -> str:
    '''A parent id that is the same every time this index is rebuilt.

    Deterministic ids make two indices comparable at the document level, which
    is what lets a per-query result cached against one index be lined up with
    another, and what makes a rebuild a no-op rather than a re-shuffle.
    '''

    return str(uuid5(NAMESPACE_URL, f"{spec.slug}|{position}|{text[:200]}"))


def build_stored_documents(
    spec: IndexSpec,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Document], List[Document]]:
    '''The documents this index stores, and the documents it embeds.

    Parameters:
    ---------
    spec (IndexSpec): which index to build.
    sections (list): the cached sections, loaded when omitted.

    Returns:
    ----------
    parents (list): what a hit returns; each carries a `doc_id`.
    children (list): what gets embedded; each carries its parent's `doc_id`.
    '''

    from backend.sop_rag.config import ID_KEY

    sections = load_sections() if sections is None else sections
    # Built without the source line, which is re-attached below.
    documents = sections_to_documents(sections, source_prefix=False)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=spec.chunk_size, chunk_overlap=spec.chunk_overlap
    )
    # `basic` stores the embedded chunk itself; `parent_child` stores the whole
    # section, which nothing splits further.
    units = child_splitter.split_documents(documents) if spec.kind == "basic" else documents

    parents = []
    for position, unit in enumerate(units):
        metadata = dict(unit.metadata or {})
        text = unit.page_content
        if spec.source_prefix:
            text = SOURCE_PREFIX_TEMPLATE.format(name=metadata.get("source")) + text
        metadata[ID_KEY] = _stable_id(spec, position, text)
        parents.append(Document(page_content=text, metadata=metadata))

    if spec.kind == "basic":
        children = [
            Document(page_content=parent.page_content, metadata=dict(parent.metadata))
            for parent in parents
        ]
    else:
        children = [
            Document(
                page_content=child.page_content,
                metadata={
                    ID_KEY: parent.metadata[ID_KEY],
                    "source": parent.metadata.get("source"),
                    "page": parent.metadata.get("page"),
                },
            )
            for parent in parents
            for child in child_splitter.split_documents([parent])
        ]

    return parents, children


def open_chroma(spec: IndexSpec, embeddings: Any = None) -> Any:
    '''Open (or create) this index's Chroma collection.'''
    from langchain_community.vectorstores import Chroma

    spec.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=spec.collection_name,
        embedding_function=embeddings or get_embeddings(spec.embedding_model),
        persist_directory=str(spec.chroma_dir),
    )


def write_docstore(spec: IndexSpec, parents: List[Document]) -> None:
    '''Persist the stored documents as JSON, keyed by `doc_id`.'''
    from backend.sop_rag.config import ID_KEY

    payload = {
        "manifest_version": MANIFEST_VERSION,
        "documents": {
            str(parent.metadata[ID_KEY]): {
                "page_content": parent.page_content,
                "metadata": parent.metadata,
            }
            for parent in parents
        },
    }
    spec.docstore_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = spec.docstore_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    temporary.replace(spec.docstore_path)


def read_docstore(spec: IndexSpec) -> InMemoryStore:
    '''The stored documents, as a `BaseStore` a ParentDocumentRetriever can use.'''
    if not spec.docstore_path.exists():
        raise FileNotFoundError(f"No docstore for {spec.slug}. Call build_index(spec) first.")
    payload = json.loads(spec.docstore_path.read_text(encoding="utf-8"))
    store = InMemoryStore()
    store.mset(
        [
            (key, Document(page_content=entry["page_content"], metadata=entry["metadata"]))
            for key, entry in payload["documents"].items()
        ]
    )
    return store


def index_exists(spec: IndexSpec) -> bool:
    '''Whether this index is already built and its manifest matches.'''
    if not (spec.manifest_path.exists() and spec.docstore_path.exists()):
        return False
    if not spec.bm25_corpus_path.exists():
        return False
    try:
        manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return (
        manifest.get("manifest_version") == MANIFEST_VERSION
        and manifest.get("slug") == spec.slug
        and int(manifest.get("children") or 0) > 0
    )


def build_index(
    spec: IndexSpec,
    sections: Optional[List[Dict[str, Any]]] = None,
    embed: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    '''Build (or reuse) one index.

    Parameters:
    ---------
    spec (IndexSpec): which index to build.
    sections (list): the cached sections, loaded when omitted.
    embed (boolean): embed the children. False builds only the docstore and the
        BM25 corpus, which is all a sparse-arm-only experiment needs and costs
        nothing.
    verbose (boolean): print progress.

    Returns:
    ----------
    manifest (dict): what was built — parents, children, characters.
    '''

    from backend.sop_rag.bm25_corpus import write_bm25_corpus

    ensure_directories()
    if index_exists(spec):
        return json.loads(spec.manifest_path.read_text(encoding="utf-8"))

    spec.root.mkdir(parents=True, exist_ok=True)
    parents, children = build_stored_documents(spec, sections)

    write_docstore(spec, parents)
    total = write_bm25_corpus(docstore=read_docstore(spec), path=spec.bm25_corpus_path)

    embedded = 0
    if embed:
        store = open_chroma(spec)
        embedded = store._collection.count()
        if not embedded:
            for start in range(0, len(children), EMBED_BATCH):
                store.add_documents(children[start : start + EMBED_BATCH])
                if verbose:
                    print(
                        f"    {spec.slug}: embedded "
                        f"{min(start + EMBED_BATCH, len(children))}/{len(children)}",
                        flush=True,
                    )
            embedded = store._collection.count()

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "slug": spec.slug,
        "kind": spec.kind,
        "embedding_model": spec.embedding_model,
        "chunk_size": spec.chunk_size,
        "chunk_overlap": spec.chunk_overlap,
        "source_prefix": spec.source_prefix,
        "parents": len(parents),
        "children": embedded or len(children),
        "embedded": bool(embed),
        "bm25_documents": total,
        "parent_chars": sum(len(parent.page_content) for parent in parents),
        "child_chars": sum(len(child.page_content) for child in children),
    }
    spec.manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if verbose:
        print(
            f"  built {spec.slug}: {manifest['parents']} stored docs, "
            f"{manifest['children']} vectors",
            flush=True,
        )
    return manifest


def build_indices(specs: List[IndexSpec], embed: bool = True, verbose: bool = True) -> "Any":
    '''Build several indices and report what each one holds.

    Parameters:
    ---------
    specs (list): the indices to build; already-built ones are reused.
    embed (boolean): embed the children.
    verbose (boolean): print progress.

    Returns:
    ----------
    summary (DataFrame): one row per index.
    '''

    import pandas as pd

    sections = load_sections()
    return pd.DataFrame(
        [build_index(spec, sections, embed=embed, verbose=verbose) for spec in specs]
    )


__all__ = [
    "build_index",
    "build_indices",
    "build_stored_documents",
    "get_embeddings",
    "open_chroma",
    "read_docstore",
]
