'''Retrieval evaluation for the SOP search machine.

Self-contained under `analysis/sop_evaluation/`: it imports code from
`backend.sop_rag` (the parse, the BM25 tokeniser, the fusion functions) but
builds its own indices here and never reads or writes the production index.
'''

from eval_rag import config  # noqa: F401  (puts REPO_ROOT on sys.path)

__all__ = ["config"]
