'''Retrieval metrics for the SOP question set.

Each question has exactly one gold passage, given as an `evidence_snippet`
quoted out of one `evidence_file`. So "did retrieval work" is a
graded-relevance question with a single relevant document, and the metrics
follow from that:

* **recall@k** — the gold snippet appears in one of the top k passages. With one
  relevant document this is also precision-oriented hit-rate; it is the
  objective this notebook selects on.
* **nDCG@k** — the same event, discounted by *where* it landed. Two
  configurations can both find the passage while one puts it at rank 1 and the
  other at rank 5, and only nDCG separates them.
* **MRR@k** — the same idea, read as "how far down the list does the agent have
  to look".
* **source_recall@k** — did any returned passage come from the right *document*.
  A cheap, fuzzy-matching-independent sanity check: it catches a configuration
  that retrieves the right SOP but the wrong section, which recall@k scores as a
  flat miss.
* **tokens@k** — what the k passages cost in the specialist's context. Recall
  rises monotonically with k and cost rises with it, so k cannot be chosen from
  recall alone.

Matching is fuzzy because the gold snippet is quoted from the PDF while the
passage came through `unstructured` — hyphenation, ligatures, table
linearisation and whitespace all differ. `rapidfuzz.fuzz.partial_ratio > 90`
on whitespace-normalised lowercase text is the reference project's rule and is
kept unchanged, so the two evaluations' numbers are comparable.
'''

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from eval_rag.config import EVIDENCE_MATCH_THRESHOLD

_WHITESPACE = re.compile(r"\s+")
_ENCODER = None

ALL_METRICS = ("recall", "ndcg", "mrr", "source_recall", "tokens")


def normalize(text: str) -> str:
    '''Lowercase, collapse whitespace: the form both sides of a match are in.'''
    return _WHITESPACE.sub(" ", str(text or "").lower().strip())


def evidence_matches(snippet: str, passage: str) -> bool:
    '''Whether `passage` contains `snippet`, allowing for PDF-extraction drift.'''
    from rapidfuzz import fuzz

    return fuzz.partial_ratio(normalize(snippet), normalize(passage)) > EVIDENCE_MATCH_THRESHOLD


def count_tokens(text: str) -> int:
    '''Tokens `text` would cost in an OpenAI context, via `cl100k_base`.'''
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODER.encode(str(text or "")))


@dataclass
class QueryOutcome:
    '''What one configuration retrieved for one question, scored but not summarised.

    Only these three vectors are kept rather than the passages themselves: a
    sweep of 77 configurations would otherwise hold tens of thousands of
    passages in memory for no gain.

    Parameters:
    ---------
    relevance (list): 1 where the returned passage holds the gold snippet.
    source_match (list): 1 where the returned passage came from the gold file.
    tokens (list): the token cost of each returned passage.
    '''

    relevance: List[int] = field(default_factory=list)
    source_match: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)


def dcg(relevances: Sequence[float]) -> float:
    '''Discounted cumulative gain of a relevance vector, in rank order.'''
    values = np.asarray(list(relevances), dtype=float)
    if values.size == 0:
        return 0.0
    gains = (2.0**values) - 1.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum(gains / discounts))


def ndcg(relevances: Sequence[float]) -> float:
    '''DCG over the DCG of the best possible ordering of the same relevances.'''
    values = list(relevances)
    ideal = dcg(sorted(values, reverse=True))
    if ideal == 0:
        return 0.0
    return dcg(values) / ideal


def reciprocal_rank(relevances: Sequence[float]) -> float:
    '''1/rank of the first relevant document, or 0 when there is none.'''
    for rank, value in enumerate(relevances, start=1):
        if value:
            return 1.0 / rank
    return 0.0


def metrics_at_k(outcome: QueryOutcome, k: int) -> Dict[str, float]:
    '''Every metric for one question at depth `k`.

    Parameters:
    ---------
    outcome (QueryOutcome): what was retrieved for this question.
    k (int): the depth to score at; the outcome is truncated to it.

    Returns:
    ----------
    metrics (dict): recall, ndcg, mrr, source_recall and tokens at `k`.
    '''

    relevance = outcome.relevance[:k]
    return {
        "recall": float(any(relevance)),
        "ndcg": ndcg(relevance),
        "mrr": reciprocal_rank(relevance),
        "source_recall": float(any(outcome.source_match[:k])),
        "tokens": float(sum(outcome.tokens[:k])),
    }


def summarize(
    outcomes: Iterable[QueryOutcome],
    ks: Sequence[int],
    metrics: Sequence[str] = ALL_METRICS,
) -> Dict[str, float]:
    '''Mean of each requested metric over a set of questions, at each depth in `ks`.

    Parameters:
    ---------
    outcomes (iterable): one `QueryOutcome` per question.
    ks (list): the depths to report.
    metrics (list): which metrics to emit. Narrowing this is what keeps an
        optimisation sweep's output to the one column that decides it: a results
        table carrying five metrics invites the reader to pick whichever one
        agrees with them, which is the thing a single objective exists to stop.

    Returns:
    ----------
    summary (dict): `{"recall@3": ...}` for the requested metrics and depths.
    '''

    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metric(s) {sorted(unknown)}; choose from {ALL_METRICS}.")

    outcomes = list(outcomes)
    if not outcomes:
        return {}
    summary: Dict[str, float] = {"n": float(len(outcomes))}
    for k in ks:
        rows = [metrics_at_k(outcome, k) for outcome in outcomes]
        for metric in metrics:
            summary[f"{metric}@{k}"] = float(np.mean([row[metric] for row in rows]))
    return summary


def paired_bootstrap(
    baseline: Sequence[QueryOutcome],
    candidate: Sequence[QueryOutcome],
    k: int = 5,
    metric: str = "recall",
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    '''Confidence interval on the difference between two configurations.

    With 550 questions and a recall around 0.8, one configuration's *own*
    standard error is about sqrt(0.8 * 0.2 / 550) = 0.017, so an unpaired
    comparison cannot see a small difference. Paired resampling is much tighter,
    because the two configurations answer the *same* questions and most
    questions are answered the same way by both: only the questions they
    disagree on carry any information.

    Parameters:
    ---------
    baseline (list): one `QueryOutcome` per question, from the reference configuration.
    candidate (list): the same questions under the configuration being tested.
    k (int): the depth to score at.
    metric (str): `recall`, `ndcg`, `mrr` or `source_recall`.
    resamples (int): bootstrap resamples.
    confidence (float): interval width.
    seed (int): the resampler's seed, so the interval is reproducible.

    Returns:
    ----------
    report (dict): the two means, their difference, the interval, and
        `significant` — whether the interval excludes zero.
    '''

    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must cover the same questions.")

    left = np.array([metrics_at_k(outcome, k)[metric] for outcome in baseline], dtype=float)
    right = np.array([metrics_at_k(outcome, k)[metric] for outcome in candidate], dtype=float)
    difference = right - left

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(resamples, len(difference)))
    means = difference[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])

    return {
        f"{metric}@{k}_baseline": float(left.mean()),
        f"{metric}@{k}_candidate": float(right.mean()),
        "difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_disagree": int(np.count_nonzero(difference)),
        "significant": bool(low > 0 or high < 0),
    }


__all__ = [
    "ALL_METRICS",
    "QueryOutcome",
    "count_tokens",
    "evidence_matches",
    "metrics_at_k",
    "paired_bootstrap",
    "summarize",
]
