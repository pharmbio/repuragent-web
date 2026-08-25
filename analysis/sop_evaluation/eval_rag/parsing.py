from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.documents import Document

from eval_rag.config import PARENTS_CACHE_PATH, SOP_DIR, ensure_directories

CACHE_VERSION = 1
SOURCE_PREFIX_TEMPLATE = "[Source: {name}]\n\n"


def parse_sop_folder() -> Dict[str, Any]:
    '''Parse every PDF in `SOPs/` into parent sections.

    Returns:
    ----------
    payload (dict): `{cache_version, sections: [...]}` where each section
        carries `raw_text`, `source`, `page`, `element_type` and `order`.
    '''

    from backend.sop_rag.sop_indexer import (
        _create_image_describer,
        build_parent_documents,
        discover_pdf_files,
    )

    pdfs = discover_pdf_files(SOP_DIR)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {SOP_DIR}.")

    describer = _create_image_describer()

    sections: List[Dict[str, Any]] = []
    for path in pdfs:
        print(f"  parsing {path.name} ...", flush=True)
        parents = build_parent_documents(path, describer)
        for order, parent in enumerate(parents):
            metadata = dict(parent.metadata or {})
            # Strip the `[Source: <filename>]` line the production indexer
            # prepends; `sections_to_documents` puts it back when asked to.
            text = parent.page_content
            prefix = SOURCE_PREFIX_TEMPLATE.format(name=path.name)
            if text.startswith(prefix):
                text = text[len(prefix) :]
            sections.append(
                {
                    "raw_text": text,
                    "source": metadata.get("source", path.name),
                    "page": metadata.get("page"),
                    "element_type": metadata.get("element_type"),
                    "order": order,
                }
            )
        print(f"    {len(parents)} sections", flush=True)

    return {"cache_version": CACHE_VERSION, "sections": sections}


def load_sections() -> List[Dict[str, Any]]:
    '''The parsed sections, parsing (and caching) them on the first call.

    Returns:
    ----------
    sections (list): one dict per parent section.
    '''

    ensure_directories()

    if PARENTS_CACHE_PATH.exists():
        payload = json.loads(PARENTS_CACHE_PATH.read_text(encoding="utf-8"))
        if payload.get("cache_version") == CACHE_VERSION and payload.get("sections"):
            return payload["sections"]
        print(f"  cache at {PARENTS_CACHE_PATH.name} is stale; re-parsing")

    payload = parse_sop_folder()
    temporary = PARENTS_CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    temporary.replace(PARENTS_CACHE_PATH)
    print(f"  cached {len(payload['sections'])} sections in {PARENTS_CACHE_PATH}")
    return payload["sections"]


def sections_to_documents(
    sections: List[Dict[str, Any]],
    source_prefix: bool = True,
) -> List[Document]:
    '''Turn cached sections into Documents, optionally re-attaching the source line.

    Parameters:
    ---------
    sections (list): what `load_sections` returned.
    source_prefix (boolean): put `[Source: <filename>]` back at the top of each
        section, as the production indexer does.

    Returns:
    ----------
    documents (list): one Document per section, carrying `source`, `page`,
        `element_type` and the section's position in its document.
    '''

    documents: List[Document] = []
    for section in sections:
        name = section["source"]
        text = section["raw_text"]
        if source_prefix:
            text = SOURCE_PREFIX_TEMPLATE.format(name=name) + text
        metadata = {
            "source": name,
            "element_type": section.get("element_type"),
            "order": section.get("order"),
        }
        if section.get("page") is not None:
            metadata["page"] = int(section["page"])
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


__all__ = ["SOURCE_PREFIX_TEMPLATE", "load_sections", "sections_to_documents"]
