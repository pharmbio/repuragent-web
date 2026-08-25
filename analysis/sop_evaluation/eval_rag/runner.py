from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.sop_rag.bm25_corpus import read_bm25_corpus
from backend.sop_rag.config import ID_KEY
from backend.sop_rag.sop_retriever import (
    SCORE_FUSIONS,
    EnsembleSOPRetriever,
    preprocess_bm25_text,
)

from eval_rag.config import (
    CHILD_POOL,
    FETCH_K_MAX,
    IndexSpec,
    DEFAULT_SETTINGS,
    QUERY_EMBEDDING_DIR,
    canonical_source,
    ensure_directories,
)
from eval_rag.indexes import get_embeddings, open_chroma, read_docstore
from eval_rag.metrics import (
    ALL_METRICS,
    QueryOutcome,
    count_tokens,
    evidence_matches,
    summarize,
)

CHILDREN_PER_PARENT = 4
DEFAULT_KS = (1, 3, 5, 10)


# Query embeddings, cached per embedding model
def embed_questions(questions: Sequence[str], model: str) -> List[List[float]]:
    '''Embed the question set once per embedding model and cache the vectors.

    Parameters:
    ---------
    questions (list): the question strings, in question-set order.
    model (str): the OpenAI embedding model.

    Returns:
    ----------
    vectors (list): one embedding per question.
    '''

    ensure_directories()
    cache_path = QUERY_EMBEDDING_DIR / f"{model.replace('/', '_')}.json"
    cached: Dict[str, List[float]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    missing = [text for text in questions if text not in cached]
    if missing:
        print(f"  embedding {len(missing)} question(s) with {model}", flush=True)
        embeddings = get_embeddings(model)
        for start in range(0, len(missing), 256):
            batch = missing[start : start + 256]
            for text, vector in zip(batch, embeddings.embed_documents(batch)):
                cached[text] = vector
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(cached), encoding="utf-8")
        temporary.replace(cache_path)

    return [cached[text] for text in questions]


# The two arms, cached
@dataclass
class Candidates:
    '''One arm's ranking for one query: parallel `doc_ids` and `scores`.'''

    doc_ids: List[str]
    scores: List[float]


class DenseArm:
    '''The dense arm of one index, with its child pool cached per query.'''

    def __init__(self, spec: IndexSpec) -> None:
        self.spec = spec
        self.store = open_chroma(spec)
        children = self.store._collection.count()
        if children == 0:
            raise RuntimeError(
                f"The Chroma collection for {spec.slug} is empty; build the index first."
            )
        # Never ask for more children than the collection holds. Chroma usually clamps, but
        # asking hnswlib for close to every element raises "Cannot return the results in a
        # contigious 2D array. Probably ef or M is too small" instead, which is a property of
        # the graph rather than of the query and so appears and disappears between rebuilds.
        self.child_pool = max(1, min(CHILD_POOL, children))
        self._pool: Dict[int, List[Tuple[Dict[str, Any], float]]] = {}

    def _children(self, index: int, vector: Sequence[float]) -> List[Tuple[Dict[str, Any], float]]:
        pool = self._pool.get(index)
        if pool is None:
            hits = self.store.similarity_search_by_vector_with_relevance_scores(
                list(vector), k=self.child_pool
            )
            pool = [(dict(document.metadata or {}), float(distance)) for document, distance in hits]
            self._pool[index] = pool
        return pool

    def candidates(
        self,
        index: int,
        vector: Sequence[float],
        fetch_k: int,
        child_k: int,
    ) -> Candidates:
        '''The dense ranking for question `index`, as parents.

        Parameters:
        ---------
        index (int): which question.
        vector (list): its cached embedding.
        fetch_k (int): how many parents to keep.
        child_k (int): how many children to fold up. This is not cosmetic:
            production pulls `fetch_k * CHILDREN_PER_PARENT` children for a
            score fusion (`_dense_with_scores`) but only `fetch_k` children for
            a rank fusion (`_dense_ranked`, whose search_kwargs are `{"k":
            fetch_k}`), so the dense arm is a *different list* depending on which
            fusion function is asked for.

        Returns:
        ----------
        candidates (Candidates): up to `fetch_k` parents, best first.
        '''

        best: Dict[str, float] = {}
        order: List[str] = []
        for metadata, distance in self._children(index, vector)[: min(child_k, self.child_pool)]:
            parent_id = (metadata or {}).get(ID_KEY)
            if parent_id is None:
                continue
            similarity = -float(distance)
            if parent_id not in best:
                best[parent_id] = similarity
                order.append(parent_id)
                if len(order) >= fetch_k:
                    break
            elif similarity > best[parent_id]:
                best[parent_id] = similarity

        kept = order[:fetch_k]
        return Candidates(kept, [best[parent_id] for parent_id in kept])


class SparseArm:
    '''The BM25 arm of one index, tokenised once and re-parameterised for free.'''

    def __init__(self, spec: IndexSpec, tokenizer: Optional[Callable] = None) -> None:
        self.tokenizer = tokenizer or preprocess_bm25_text
        documents = read_bm25_corpus(spec.bm25_corpus_path)
        self.doc_ids = [str((document.metadata or {}).get(ID_KEY)) for document in documents]
        self.corpus = [self.tokenizer(document.page_content) for document in documents]
        self._models: Dict[Tuple[float, float], Any] = {}

    def _model(self, k1: float, b: float) -> Any:
        key = (float(k1), float(b))
        model = self._models.get(key)
        if model is None:
            from rank_bm25 import BM25Okapi

            model = BM25Okapi(self.corpus, k1=float(k1), b=float(b))
            self._models[key] = model
        return model

    def candidates(self, query: str, k1: float, b: float, fetch_k: int) -> Candidates:
        '''The BM25 ranking for `query` under these parameters.'''
        scores = np.asarray(self._model(k1, b).get_scores(self.tokenizer(query)), dtype=float)
        order = np.argsort(-scores, kind="stable")[:fetch_k]
        return Candidates([self.doc_ids[i] for i in order], [float(scores[i]) for i in order])


# Fusion, with no index behind it
class Fuser(EnsembleSOPRetriever):
    '''The production fusion functions with no index behind them.'''

    def __init__(self, fuse_func: str, fetch_k: int, rrf_c: int) -> None:
        super().__init__(fuse_func=fuse_func, fetch_k=fetch_k, max_results=1)
        self.rrf_c = int(rrf_c)

    def _initialize(self) -> None:
        return None


_FUSERS: Dict[Tuple[str, int, int], Fuser] = {}


def _fuser(fuse_func: str, fetch_k: int, rrf_c: int) -> Fuser:
    '''One `Fuser` per `(fuse_func, fetch_k, rrf_c)`; they hold no state.'''
    key = (fuse_func, int(fetch_k), int(rrf_c))
    fuser = _FUSERS.get(key)
    if fuser is None:
        fuser = Fuser(fuse_func=fuse_func, fetch_k=int(fetch_k), rrf_c=int(rrf_c))
        _FUSERS[key] = fuser
    return fuser


# The evaluator
class Evaluator:
    '''Everything one index needs to be swept, with the expensive parts cached.

    Parameters:
    ---------
    spec (IndexSpec): the built index to evaluate.
    questions (DataFrame): the question set.
    tokenizer (callable): the BM25 tokeniser, defaulting to the production one.
    dense (boolean): open the dense arm. False for a sparse-only experiment, so
        an index built with `embed=False` can still be evaluated.
    sparse (boolean): open the sparse arm.
    '''

    def __init__(
        self,
        spec: IndexSpec,
        questions: "Any",
        tokenizer: Optional[Callable] = None,
        dense: bool = True,
        sparse: bool = True,
    ) -> None:
        self.spec = spec
        self.questions = questions.reset_index(drop=True)
        self.texts = self.questions["question"].tolist()
        self.snippets = self.questions["evidence_snippet"].tolist()
        self.evidence_files = [canonical_source(name) for name in self.questions["evidence_file"]]

        self.dense_arm = DenseArm(spec) if dense else None
        self.sparse_arm = SparseArm(spec, tokenizer) if sparse else None
        self.vectors = embed_questions(self.texts, spec.embedding_model) if dense else None

        self._documents = read_docstore(spec)
        self._content: Dict[str, str] = {}
        self._source: Dict[str, str] = {}
        self._tokens: Dict[str, int] = {}
        self._relevant: Dict[Tuple[int, str], int] = {}
        self._stubs: Dict[str, Any] = {}

    # Document-level memos
    def _load(self, doc_id: str) -> None:
        if doc_id in self._content:
            return
        document = self._documents.mget([doc_id])[0]
        text = "" if document is None else document.page_content
        metadata = {} if document is None else (document.metadata or {})
        self._content[doc_id] = text
        self._source[doc_id] = canonical_source(metadata.get("source"))
        self._tokens[doc_id] = count_tokens(text)

    def _stub(self, doc_id: str) -> Any:
        '''A Document carrying only `doc_id` — all the fusion code reads.'''
        from langchain_core.documents import Document

        stub = self._stubs.get(doc_id)
        if stub is None:
            stub = Document(page_content="", metadata={ID_KEY: doc_id})
            self._stubs[doc_id] = stub
        return stub

    def _is_relevant(self, index: int, doc_id: str) -> int:
        '''Whether this document answers question `index`, memoised.'''

        key = (index, doc_id)
        hit = self._relevant.get(key)
        if hit is None:
            self._load(doc_id)
            hit = int(evidence_matches(self.snippets[index], self._content[doc_id]))
            self._relevant[key] = hit
        return hit

    def _outcome(self, index: int, doc_ids: Sequence[str]) -> QueryOutcome:
        '''Score one ranked list of `doc_id`s against question `index`.'''
        outcome = QueryOutcome()
        wanted = self.evidence_files[index]
        for doc_id in doc_ids:
            self._load(doc_id)
            outcome.relevance.append(self._is_relevant(index, doc_id))
            outcome.source_match.append(int(self._source[doc_id] == wanted))
            outcome.tokens.append(self._tokens[doc_id])
        return outcome

    def _fuse(
        self,
        sparse: Candidates,
        dense: Candidates,
        bm25_weight: float,
        dense_weight: float,
        fuse_func: str,
        fetch_k: int,
        rrf_c: int,
    ) -> List[str]:
        '''Combine two candidate lists with the production fusion code.'''
        fuser = _fuser(fuse_func, fetch_k, rrf_c)
        weights = [bm25_weight, dense_weight]
        if fuse_func in SCORE_FUSIONS:
            fused = fuser.fuse(
                weights,
                scored_lists=[
                    list(zip((self._stub(i) for i in sparse.doc_ids), sparse.scores)),
                    list(zip((self._stub(i) for i in dense.doc_ids), dense.scores)),
                ],
            )
        else:
            fused = fuser.fuse(
                weights,
                ranked_lists=[
                    [self._stub(i) for i in sparse.doc_ids],
                    [self._stub(i) for i in dense.doc_ids],
                ],
            )
        return [document.metadata[ID_KEY] for document, _ in fused]

    # Running a configuration
    def run(
        self,
        arms: str = "both",
        fetch_k: int = DEFAULT_SETTINGS["fetch_k"],
        bm25_weight: float = DEFAULT_SETTINGS["bm25_weight"],
        dense_weight: float = DEFAULT_SETTINGS["dense_weight"],
        bm25_k1: float = DEFAULT_SETTINGS["bm25_k1"],
        bm25_b: float = DEFAULT_SETTINGS["bm25_b"],
        fuse_func: str = DEFAULT_SETTINGS["fuse_func"],
        rrf_c: int = DEFAULT_SETTINGS["rrf_c"],
        depth: int = 10,
    ) -> List[QueryOutcome]:
        '''Retrieve and score every question under one configuration.

        Parameters:
        ---------
        arms (str): `both`, `dense` or `sparse`.
        fetch_k (int): candidates per arm before fusion.
        bm25_weight (float): the sparse arm's fusion weight.
        dense_weight (float): the dense arm's fusion weight.
        bm25_k1 (float): BM25 term-frequency saturation.
        bm25_b (float): BM25 length normalisation.
        fuse_func (str): the fusion function.
        rrf_c (int): the RRF constant.
        depth (int): how many results to keep per question, i.e. the largest `k`
            any metric will be reported at.

        Returns:
        ----------
        outcomes (list): one `QueryOutcome` per question.
        '''

        if arms not in ("both", "dense", "sparse"):
            raise ValueError(f"Unknown arms '{arms}'. Choose both, dense or sparse.")
        if fetch_k > FETCH_K_MAX:
            raise ValueError(f"fetch_k={fetch_k} exceeds FETCH_K_MAX={FETCH_K_MAX}.")
        if arms != "sparse" and self.dense_arm is None:
            raise RuntimeError("This evaluator was built without a dense arm.")
        if arms != "dense" and self.sparse_arm is None:
            raise RuntimeError("This evaluator was built without a sparse arm.")

        # Production's dense arm pulls a child pool sized for the fusion it is
        # feeding; see `DenseArm.candidates`.
        child_k = fetch_k * CHILDREN_PER_PARENT if fuse_func in SCORE_FUSIONS else fetch_k

        outcomes: List[QueryOutcome] = []
        for index in range(len(self.texts)):
            if arms == "sparse":
                doc_ids = self.sparse_arm.candidates(
                    self.texts[index], bm25_k1, bm25_b, fetch_k
                ).doc_ids
            else:
                dense = self.dense_arm.candidates(
                    index, self.vectors[index], fetch_k, child_k
                )
                if arms == "dense":
                    doc_ids = dense.doc_ids
                else:
                    sparse = self.sparse_arm.candidates(
                        self.texts[index], bm25_k1, bm25_b, fetch_k
                    )
                    doc_ids = self._fuse(
                        sparse, dense, bm25_weight, dense_weight, fuse_func, fetch_k, rrf_c
                    )
            outcomes.append(self._outcome(index, doc_ids[:depth]))
        return outcomes

    def evaluate(
        self,
        ks: Sequence[int] = DEFAULT_KS,
        metrics: Sequence[str] = ALL_METRICS,
        **kwargs: Any,
    ) -> Dict[str, float]:
        '''`run` followed by `summarize`: one configuration's numbers.

        `metrics` narrows what comes back; an optimisation sweep passes just the
        one it selects on.
        '''

        return summarize(self.run(depth=max(ks), **kwargs), ks, metrics=metrics)


_EVALUATORS: Dict[Tuple[Any, ...], Evaluator] = {}


def evaluator_for(
    spec: IndexSpec,
    questions: "Any",
    tokenizer: Optional[Callable] = None,
    dense: bool = True,
    sparse: bool = True,
) -> Evaluator:
    '''The `Evaluator` for these settings, built at most once per session.

    Building one opens Chroma, tokenises the BM25 corpus and reads the question
    embeddings, so a notebook that sweeps 12 indices and then revisits three of
    them should not pay for them twice.
    '''

    key = (spec.slug, getattr(tokenizer, "__name__", None), dense, sparse)
    evaluator = _EVALUATORS.get(key)
    if evaluator is None:
        evaluator = Evaluator(
            spec, questions=questions, tokenizer=tokenizer, dense=dense, sparse=sparse
        )
        _EVALUATORS[key] = evaluator
    return evaluator


# Diagnostics on the corpus
def corpus_stats(spec: IndexSpec) -> Dict[str, Any]:
    '''How big the searchable corpus of one index is.

    Parameters:
    ---------
    spec (IndexSpec): a built index.

    Returns:
    ----------
    stats (dict): documents, characters, and the random-ranking recall at 1/5/10.
    '''

    documents = read_bm25_corpus(spec.bm25_corpus_path)
    lengths = [len(document.page_content) for document in documents]
    total = len(documents)
    stats: Dict[str, Any] = {
        "index": spec.slug,
        "documents": total,
        "chars_total": int(np.sum(lengths)),
        "chars_median": float(np.median(lengths)),
        "chars_max": int(np.max(lengths)),
        "tokens_median": float(np.median([count_tokens(d.page_content) for d in documents])),
    }
    for k in (1, 5, 10):
        stats[f"random_recall@{k}"] = min(1.0, k / total)
    return stats


# Sweeping: two shapes, and the difference between them is the whole reason
# there are two. `sweep` varies the retrieval settings over a fixed index;
# `sweep_indices` varies the index under fixed retrieval settings.
def sweep(
    evaluator: Evaluator,
    configurations: Iterable[Dict[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
    metrics: Sequence[str] = ALL_METRICS,
) -> "Any":
    '''Run a list of configurations against one index and tabulate the results.

    Parameters:
    ---------
    evaluator (Evaluator): the index to sweep.
    configurations (iterable): dicts of `Evaluator.run` arguments. `label` names
        the row and `meta` is a dict of extra columns; neither is passed to
        `run`.
    ks (list): the depths to report every metric at.
    metrics (list): which metrics to emit.

    Returns:
    ----------
    results (DataFrame): one row per configuration, metrics as columns.
    '''

    import pandas as pd

    rows = []
    for position, configuration in enumerate(configurations, start=1):
        settings = dict(configuration)
        label = settings.pop("label", None) or f"config_{position}"
        meta = settings.pop("meta", None) or {}
        rows.append(
            {
                "label": label,
                "index": evaluator.spec.slug,
                **settings,
                **meta,
                **evaluator.evaluate(ks=ks, metrics=metrics, **settings),
            }
        )
    return pd.DataFrame(rows)


SPEC_COLUMNS = ("kind", "embedding_model", "chunk_size", "chunk_overlap", "source_prefix")


def sweep_indices(
    specs: Sequence[IndexSpec],
    questions: "Any",
    run_kwargs: Optional[Dict[str, Any]] = None,
    ks: Sequence[int] = DEFAULT_KS,
    metrics: Sequence[str] = ALL_METRICS,
    tokenizer: Optional[Callable] = None,
    dense: bool = True,
    sparse: bool = True,
    labels: Optional[Sequence[str]] = None,
) -> "Any":
    '''Run one retrieval configuration against several indices.

    Parameters:
    ---------
    specs (list): the built indices to compare.
    questions (DataFrame): the question set.
    run_kwargs (dict): the retrieval configuration, as `Evaluator.run` arguments.
    ks (list): the depths to report every metric at.
    metrics (list): which metrics to emit.
    tokenizer (callable): the BM25 tokeniser.
    dense (boolean): open the dense arm.
    sparse (boolean): open the sparse arm. False skips the BM25 corpus entirely.
    labels (list): one label per spec, defaulting to the slug.

    Returns:
    ----------
    results (DataFrame): one row per index; the spec's fields plus the metrics,
        plus `documents` and `median_doc_tokens` so a score can be read next to
        the granularity that produced it.
    '''

    import pandas as pd

    run_kwargs = dict(run_kwargs or {})
    rows = []
    for position, spec in enumerate(specs):
        evaluator = evaluator_for(
            spec, questions=questions, tokenizer=tokenizer, dense=dense, sparse=sparse
        )
        stats = corpus_stats(spec)
        rows.append(
            {
                "label": labels[position] if labels else spec.slug,
                "index": spec.slug,
                **{column: getattr(spec, column) for column in SPEC_COLUMNS},
                "documents": stats["documents"],
                "median_doc_tokens": stats["tokens_median"],
                **evaluator.evaluate(ks=ks, metrics=metrics, **run_kwargs),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_KS",
    "Evaluator",
    "corpus_stats",
    "evaluator_for",
    "sweep",
    "sweep_indices",
]
