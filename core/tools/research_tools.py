from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, List

import requests
from langchain_core.tools import tool

from app.config import LIBRARIAN_MAX_RESULTS, logger
from backend.librarian_search import search_literature
from backend.utils.cancellation import ExecutionCancelled
from backend.sop_rag.config import RETRIEVAL_CONFIG
from backend.sop_rag.sop_retriever import get_sop_retriever

# The tool's default, and the retriever's, are the same number by construction.
DEFAULT_SOP_RESULTS = RETRIEVAL_CONFIG["max_results"]
# How many librarian passages reach the transcript. The pipeline's cost does not
# depend on it, so it is a context budget rather than a search setting.
DEFAULT_LIBRARIAN_RESULTS = LIBRARIAN_MAX_RESULTS

@dataclass
class LitSenseObject:
    '''Dataclass for a single result from the LitSense API.'''

    text: str
    score: float
    annotations: List[str]
    pmid: int
    pmcid: str
    section: str


class PyLitSense:
    '''Python wrapper for the LitSense API.'''

    def __init__(self, base_url: str = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/") -> None:
        self.base_url = base_url.rstrip("/") + "/"

    def query(
        self,
        query_str: str,
        *,
        rerank: bool = True,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        mode: str = "passages",
    ) -> List[LitSenseObject]:
        '''Query LitSense API for passages or sentences.

        Parameters:
        ---------
        query_str (str): what to search for, in free text.
        rerank (boolean): whether to let LitSense re-rank by relevance.
        limit (int): how many results to return.
        min_score (float): drop results scoring below this.
        mode (str): `passages` or `sentences`, the granularity LitSense returns.

        Returns:
        ----------
        results (list): the matching `LitSenseObject`s.
        '''

        path = "passages/" if mode == "passages" else "sentences/"
        url = f"{self.base_url}{path}"

        params = {"query": query_str, "rerank": rerank, "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        results = [LitSenseObject(**result) for result in response.json()]

        if min_score is not None:
            results = [result for result in results if result.score >= min_score]

        return results


_MARKUP = re.compile(r"<[^>]+>")


def _strip_markup(text: str) -> str:
    '''Drop the inline markup Europe PMC leaves in a title or abstract.

    Its abstracts are JATS-derived and carry the tags with them — a real passage
    comes back as `<h4>Results</h4>The pooled analysis ...` and a heterogeneity
    statistic as `<i>I</i> <sup>2</sup>`. Upstream passes them straight through,
    and an agent told to quote its evidence verbatim would put them in the report.

    Parameters:
    ---------
    text (str): the snippet or title as the librarian returned it.

    Returns:
    ----------
    text (str): the same wording with tags removed and whitespace collapsed.
    '''

    return " ".join(_MARKUP.sub(" ", str(text or "")).split())


def _short_authors(authors: str) -> str:
    '''Shorten Europe PMC's author string to a citable "First A, et al.".

    Parameters:
    ---------
    authors (str): the full comma-separated author string.

    Returns:
    ----------
    authors (str): the first author, with "et al." when there were more. A consortium paper otherwise spends 40 lines of the transcript on names nobody reads.
    '''

    names = [name.strip() for name in str(authors or "").split(",") if name.strip()]
    if not names:
        return "Unknown authors"
    return names[0] if len(names) == 1 else f"{names[0]}, et al."


def _format_librarian_passages(passages) -> str:
    '''Format librarian evidence passages for the agent.

    Parameters:
    ---------
    passages (list): the passage dicts `search_literature` returned.

    Returns:
    ----------
    text (str): one block per paper — its citation, its identifiers, and the sentences the relevance judge cited, which are what may be quoted.
    '''

    sections: List[str] = []
    for idx, passage in enumerate(passages, start=1):
        citation = " · ".join(
            part for part in (passage.get("journal"), passage.get("year")) if part
        )
        identifiers = " | ".join(
            f"{label}: {value}"
            for label, value in (("PMID", passage.get("pmid")), ("DOI", passage.get("doi")))
            if value
        )
        snippets = [clean for text in passage.get("evidence_snippets") or [] if (clean := _strip_markup(text))]
        evidence = "\n".join(f"- {text}" for text in snippets) or "- (no sentence extracted)"

        lines = [f"\n--- Passage #{idx} ---", f"Title: {_strip_markup(passage.get('title')) or 'No title'}"]
        lines.append(f"Authors: {_short_authors(passage.get('authors'))}")
        if citation:
            lines.append(f"Published: {citation}")
        if identifiers:
            lines.append(identifiers)
        lines.append("Full text read: " + ("yes" if passage.get("has_fulltext") else "abstract only"))
        lines.append(f"Evidence:\n{evidence}\n")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def _format_sop_results(documents) -> str:
    '''Format SOP documents for display.

    Parameters:
    ---------
    documents (list): the SOP chunks the retriever returned.

    Returns:
    ----------
    text (str): the chunks rendered for the agent, each headed by its source and page.
    '''

    result_lines: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source") or metadata.get("filename") or "Unknown"
        heading = os.path.basename(str(source))
        page = metadata.get("page")
        if page is not None:
            heading = f"{heading}, page {page}"

        result_lines.extend(
            [
                f"\n--- Document {idx} ---\n",
                f"Source: {heading}\n",
                f"Content: {getattr(doc, 'page_content', '')}\n",
            ]
        )

    return "".join(result_lines)

@tool
def literature_search_litsense(query: str, limit: int = 5) -> str:
    '''Search PubMed passages via the LitSense API. Use it to check whether something has been reported, to confirm a fact 
    in passing, or to find the wording a field uses before searching properly.

    Parameters:
    ---------
    query (str): what to search for, in free text.
    limit (int): how many passages to return.

    Returns:
    ----------
    results (str): the matching passages with their citations, or a message when there are none.
    '''

    try:
        engine = PyLitSense()
        results = engine.query(query, limit=limit)

        if not results:
            return f"No relevant literature found for '{query}'. Please try a broader query."

        result_sections: List[str] = []
        for idx, result in enumerate(results, start=1):
            section = (
                f"\n--- Passage #{idx} ---\n"
                f"PMID: {result.pmid}\n"
                f"Content: {result.text}\n"
            )
            result_sections.append(section)

        return "".join(result_sections)

    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error in literature_search_litsense: %s", exc)
        return f"Error retrieving literature for '{query}': {exc}. Please try again."


@tool
def literature_search_librarian(query: str, limit: int = DEFAULT_LIBRARIAN_RESULTS) -> str:
    '''Answer a research question from the literature, with the EMBL AI Librarian
    over Europe PMC. Use it to check whether something has been reported, to confirm a fact 
    in passing, or to find the wording a field uses before searching properly.

    Parameters:
    ---------
    query (str): the research question in natural language. Give it the whole question — the disease, the drug or target, and what you need to know about them — rather than keywords: it plans its own queries, and a bare keyword gives it nothing to plan from. Do not write Europe PMC field syntax yourself.
    limit (int): how many papers' evidence to return. This bounds the result, not the work: the search costs the same either way, so lower it only when you need the transcript kept small.

    Returns:
    ----------
    results (str): one block per paper — citation, PMID/DOI, and the evidence sentences, which are quotable verbatim. A message instead when the search found nothing.
    '''

    try:
        passages = search_literature(query, limit=limit)

        if not passages:
            return (
                f"No literature found for '{query}'. The question may be too narrow, or "
                "phrased in terms the literature does not use — try it more broadly."
            )

        return _format_librarian_passages(passages)

    except ExecutionCancelled:
        # Propagate: the run is being torn down, and reporting Stop as a failed
        # search would have the supervisor retry it.
        raise
    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error in literature_search_librarian: %s", exc)
        return f"Error retrieving literature for '{query}': {exc}. Please try again."


@tool
def protocol_search_sop(query: str, max_results: int = DEFAULT_SOP_RESULTS) -> str:
    '''Search the SOP corpus for protocols, standards and regulatory procedures.
    Use this to ground any claim that has to follow a documented procedure —
    assay protocols, reporting standards, regulatory definitions and thresholds.
    Returns the original passages with their source filenames, so the wording can
    be quoted rather than paraphrased.

    Parameters:
    ---------
    query (str): What you need the procedure for, in the terms the document would use.
    max_results (int): How many passages to return. Leave at the default unless you have a reason: a passage is a whole section of a document, so asking for more costs proportionally more of your context. Ask for fewer (2-3) when you need one specific threshold or definition, and for more (8-10) when you are surveying what a procedure covers and do not yet know which section holds it.

    Returns:
    ----------
    results (str): the matching SOP passages with their source filenames, or a message when the index holds nothing relevant.
    '''

    try:
        documents = get_sop_retriever().query(query, max_results=max_results)

        if not documents:
            return f"No SOP content found for query '{query}'."

        return _format_sop_results(documents)

    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error processing SOP query '%s': %s", query, exc)
        return f"Error retrieving SOP content for '{query}': {exc}. Please try again."


__all__ = [
    "DEFAULT_LIBRARIAN_RESULTS",
    "DEFAULT_SOP_RESULTS",
    "LitSenseObject",
    "PyLitSense",
    "literature_search_librarian",
    "literature_search_litsense",
    "protocol_search_sop",
]
