'''Retrieval over the SOP corpus (regulatory guidance, protocols, standards).

A `MultiVectorRetriever`: short summaries are embedded in Chroma, the original
chunks live in a file-backed docstore, and a hit returns the original rather than
its summary — which matters when a regulatory clause has to be quoted as written.

**Obtain one through `get_sop_retriever()`.** Constructing a retriever reopens
Chroma and walks the docstore, and `protocol_search_sop` is called on nearly every
grounding step, so building a fresh one per query made SOP search several times
slower than the retrieval itself. Call `clear_sop_retriever_cache()` after
rebuilding the index on disk.
'''

from __future__ import annotations

import json
import threading
from base64 import b64decode
from typing import Any, Dict, List, Optional

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import OPENAI_API_KEY, logger
from backend.sop_rag.config import (
    CHROMA_PERSIST_PATH,
    COLLECTION_NAME,
    DOCSTORE_DIR,
    EMBEDDING_MODEL,
    ID_KEY,
    LLM_CONFIG,
    RETRIEVAL_CONFIG,
    index_exists,
)


def _require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured; cannot run SOP retrieval.")
    return OPENAI_API_KEY


class SOPRetriever:
    def __init__(self) -> None:
        self.retriever: Optional[MultiVectorRetriever] = None
        self.rag_chain = None
        self._api_key = _require_openai_api_key()
        self._initialize()

    def _initialize(self) -> None:
        if not index_exists():
            raise FileNotFoundError(
                f"SOP index not found at {CHROMA_PERSIST_PATH} / {DOCSTORE_DIR}. "
                "Run `python -m backend.sop_rag.sop_indexer` to build it."
            )

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=self._api_key),
            persist_directory=str(CHROMA_PERSIST_PATH),
        )
        count = vectorstore._collection.count()
        if count == 0:
            raise ValueError(
                "SOP vector store is empty. Run `python -m backend.sop_rag.sop_indexer`."
            )

        docstore = LocalFileStore(str(DOCSTORE_DIR))
        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=ID_KEY,
            search_kwargs={"k": RETRIEVAL_CONFIG["default_k"]},
        )
        logger.info("SOP retriever ready: %s embedded summaries", count)
        self.rag_chain = self._create_rag_chain(self.retriever)

    # --- retrieval ------------------------------------------------------------

    def search(self, question: str) -> List[Document]:
        '''Documents relevant to `question`, as Documents rather than raw bytes.

        Parameters:
        ---------
        question (str): what to retrieve SOP material for.

        Returns:
        ----------
        documents (list): the relevant chunks as Documents rather than raw docstore bytes.
        '''

        if self.retriever is None:
            raise ValueError("SOP retriever not initialized")
        return self.as_documents(self.retriever.invoke(question))

    @staticmethod
    def as_documents(retrieved_items: List[Any]) -> List[Document]:
        '''Turn docstore payloads back into Documents.

        The docstore holds JSON-serialised documents as bytes, so a hit needs
        decoding before its metadata (the source filename, which the caller cites)
        is reachable.

        Parameters:
        ---------
        retrieved_items (list): raw docstore payloads.

        Returns:
        ----------
        documents (list): the same payloads as Documents.
        '''

        documents: List[Document] = []
        for item in retrieved_items:
            if isinstance(item, bytes):
                try:
                    payload = json.loads(item.decode("utf-8"))
                    documents.append(
                        Document(
                            page_content=payload["page_content"],
                            metadata=payload.get("metadata") or {},
                        )
                    )
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    documents.append(Document(page_content=item.decode("utf-8", errors="replace")))
            elif isinstance(item, Document):
                documents.append(item)
            elif hasattr(item, "page_content"):
                documents.append(item)
            else:
                documents.append(Document(page_content=str(item)))
        return documents

    # Kept for callers that used the private name.
    _convert_bytes_to_docs = as_documents

    def get_sources(self, question: str) -> List[str]:
        '''Distinct source filenames behind the hits for `question`.

        Parameters:
        ---------
        question (str): what to retrieve SOP material for.

        Returns:
        ----------
        sources (list): the distinct source filenames behind the hits, for citation.
        '''

        sources = []
        for doc in self.search(question):
            filename = (getattr(doc, "metadata", None) or {}).get("filename")
            if filename:
                sources.append(str(filename).rsplit("/", 1)[-1])
        return sorted(set(sources))

    # --- generation -----------------------------------------------------------

    def _create_rag_chain(self, retriever):
        return {
            "context": retriever | RunnableLambda(self._parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self._build_prompt)
                | ChatOpenAI(model=LLM_CONFIG["rag_response_model"], api_key=self._api_key)
                | StrOutputParser()
            )
        )

    def _parse_docs(self, docs: List[Any]) -> Dict[str, List[Any]]:
        '''Split base64 images from text chunks.

        Parameters:
        ---------
        docs (list): the retrieved payloads, which mix text and base64 images.

        Returns:
        ----------
        grouped (dict): the two kinds separated, since only text goes into the prompt.
        '''

        images: List[str] = []
        texts: List[Document] = []
        for doc in self.as_documents(docs):
            try:
                b64decode(doc.page_content, validate=True)
                images.append(doc.page_content)
            except Exception:
                texts.append(doc)
        return {"images": images, "texts": texts}

    def _build_prompt(self, kwargs: Dict[str, Any]) -> ChatPromptTemplate:
        docs_by_type = kwargs["context"]
        question = kwargs["question"]
        context_text = "\n\n".join(doc.page_content for doc in docs_by_type["texts"])

        prompt_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Answer the question using only the following context, which may "
                    "include text, tables and images.\n"
                    f"Context: {context_text}\n"
                    f"Question: {question}\n"
                ),
            }
        ]
        for image in docs_by_type["images"]:
            prompt_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            )
        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def query(self, question: str) -> Dict[str, Any]:
        if self.rag_chain is None:
            raise ValueError("RAG chain not initialized")
        return self.rag_chain.invoke(question)


_retriever: Optional[SOPRetriever] = None
_retriever_lock = threading.Lock()


def get_sop_retriever() -> SOPRetriever:
    '''The shared retriever, built at most once per process.

    Returns:
    ----------
    retriever (SOPRetriever): the shared instance, built at most once per process.
    '''

    global _retriever
    if _retriever is not None:
        return _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = SOPRetriever()
        return _retriever


def clear_sop_retriever_cache() -> None:
    '''Forget the cached retriever, e.g. after rebuilding the index on disk.'''

    global _retriever
    with _retriever_lock:
        _retriever = None


__all__ = ["SOPRetriever", "clear_sop_retriever_cache", "get_sop_retriever"]
