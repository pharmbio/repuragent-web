from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from statistics import fmean
from typing import Any, Dict, Optional, List, Union, Tuple

import pandas as pd
import requests
from langchain_core.tools import tool

from app.config import logger
from backend.sop_rag.sop_retriever import SOPRetriever
from backend.utils.output_paths import task_file_path
from backend.utils.storage_paths import get_data_root
from backend.utils.kgg_tools import _get_target_association_score_for_disease, OPENTARGETS_GRAPHQL_URL
from kgg.kgg_apiutils import RetMech, Ret_chembl_protein, chembl2gene2path, chembl2uniprot
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.http_errors import BaseHttpException, HttpApplicationError
import time
import random
from tqdm import tqdm

@dataclass
class LitSenseObject:
    """Dataclass for a single result from the LitSense API."""

    text: str
    score: float
    annotations: List[str]
    pmid: int
    pmcid: str
    section: str


class PyLitSense:
    """Python wrapper for the LitSense API."""

    def __init__(self, base_url: str = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/") -> None:
        self.base_url = base_url.rstrip("/") + "/"

    def query(
        self,
        query_str: str,
        *,
        rerank: bool = False,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        mode: str = "passages",
    ) -> List[LitSenseObject]:
        """Query LitSense API for passages or sentences."""
        path = "passages/" if mode == "passages" else "sentences/"
        url = f"{self.base_url}{path}"

        params = {"query": query_str, "rerank": rerank, "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        results = [LitSenseObject(**result) for result in response.json()]

        if min_score is not None:
            results = [result for result in results if result.score >= min_score]

        return results


def _format_sop_results(documents) -> str:
    """Format SOP documents for display."""
    result_lines: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        filename = "Unknown"
        if hasattr(doc, "metadata"):
            filename = os.path.basename(doc.metadata.get("filename", "Unknown"))

        result_lines.extend(
            [
                f"\n--- Document {idx} ---\n",
                f"Source: {filename}\n",
                f"Content: {getattr(doc, 'page_content', '')}\n",
            ]
        )

    return "".join(result_lines)


def _retry_chembl_request(func, *, max_attempts=3, base_delay=1.0, jitter=0.25, context=""):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except (HttpApplicationError, BaseHttpException, requests.exceptions.RequestException) as exc:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            suffix = f" [{context}]" if context else ""
            logger.warning("ChEMBL request failed%s (attempt %s/%s): %s", suffix, attempt, max_attempts, exc)
            time.sleep(delay)


MAP_TARGET_IDS_QUERY = """
    query MapTargets($terms: [String!]!) {
      mapIds(queryTerms: $terms, entityNames: ["target"]) {
        mappings {
          term
          hits {
            id
            name
            entity
          }
        }
      }
    }
"""


@lru_cache(maxsize=1)
def _load_symbol_to_ensg_map() -> Dict[str, str]:
    mapping_file = get_data_root() / "api_related_data" / "DruggableProtein_annotation_OT.csv"
    try:
        df = pd.read_csv(mapping_file)
        if "approvedSymbol" not in df.columns or "ENSG" not in df.columns:
            return {}
        return dict(df[["approvedSymbol", "ENSG"]].dropna().values)
    except Exception as exc:
        logger.debug("Unable to load protein mapping file %s: %s", mapping_file, exc)
        return {}


def _lookup_ensg_id(symbol: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    if symbol in cache:
        return cache[symbol]
    symbol_map = _load_symbol_to_ensg_map()
    ensg_id = symbol_map.get(symbol)
    if ensg_id:
        cache[symbol] = ensg_id
        return ensg_id
    variables = {"terms": [symbol]}
    try:
        response = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": MAP_TARGET_IDS_QUERY, "variables": variables},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        mappings = (((payload.get("data") or {}).get("mapIds") or {}).get("mappings") or [])
        for mapping in mappings:
            if mapping.get("term") != symbol:
                continue
            for hit in mapping.get("hits", []):
                if hit.get("entity") == "target" and str(hit.get("id", "")).startswith("ENSG"):
                    ensg_id = hit.get("id")
                    cache[symbol] = ensg_id
                    return ensg_id
    except Exception as exc:
        logger.debug("mapIds lookup failed for %s: %s", symbol, exc)
    cache[symbol] = None
    return None


def _normalize_chembl_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def _chembl_ids_from_input(chembl_input: Union[str, List[str]]) -> List[str]:
    ids: List[str] = []
    if isinstance(chembl_input, str) and os.path.isfile(chembl_input):
        ext = os.path.splitext(chembl_input)[-1].lower()
        if ext != ".csv":
            raise ValueError("Only CSV files are supported for chembl_id input.")
        df = pd.read_csv(chembl_input)
        columns = {col.lower(): col for col in df.columns}
        if "chembl_id" not in columns:
            raise ValueError("CSV file must contain a 'chembl_id' column.")
        for value in df[columns["chembl_id"]].tolist():
            normalized = _normalize_chembl_id(value)
            if normalized:
                ids.append(normalized)
    elif isinstance(chembl_input, str):
        ids = [item.strip().upper() for item in chembl_input.split(",") if item.strip()]
    elif isinstance(chembl_input, list):
        for item in chembl_input:
            normalized = _normalize_chembl_id(item)
            if normalized:
                ids.append(normalized)
    else:
        raise ValueError("Input must be a ChEMBL ID, list of IDs, or CSV file path.")

    deduped: List[str] = []
    seen = set()
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_reactome_pathways(target_info: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Optional[str]]]]:
    gene_symbol: Optional[str] = None
    pathways: List[Dict[str, Optional[str]]] = []
    for item in target_info:
        if not isinstance(item, dict):
            continue
        if "component_synonym" in item:
            candidate = str(item.get("component_synonym") or "").strip()
            if candidate:
                gene_symbol = candidate
            continue
        if item.get("xref_src_db") != "Reactome":
            continue
        pathway_name = str(item.get("xref_name") or "").strip()
        pathway_id = str(item.get("xref_id") or "").strip()
        if not pathway_name and not pathway_id:
            continue
        pathways.append(
            {
                "name": pathway_name or pathway_id,
                "id": pathway_id or None,
                "url": f"https://reactome.org/content/detail/{pathway_id}" if pathway_id else None,
            }
        )
    return gene_symbol, pathways



def RetAct(chemblIds) -> dict:
    """Function to retrieve associated assays from ChEMBL

    :param chemblIds:
    :return:
    """
    GetAct = new_client.activity
    getTar = new_client.target
    ActList = []
    filtered_list=['assay_chembl_id','assay_type','pchembl_value','target_chembl_id',
                   'target_organism','bao_label','target_type']
    for chembl in tqdm(chemblIds, desc='Retrieving bioassays from ChEMBL'):
        def _fetch_acts():
            return list(GetAct.filter(
                molecule_chembl_id=chembl,
                pchembl_value__isnull=False,
                assay_type_iregex='(B|F)',
                target_organism='Homo sapiens'
            ).only(filtered_list))

        acts = _retry_chembl_request(_fetch_acts, context=f"activity {chembl}")
        data = []
        for d in acts:
            if float(d.get('pchembl_value')) < 5:
                continue      
            if (d.get('bao_label') != 'single protein format'):
                continue
            tar = d.get('target_chembl_id')   
            tar_dict = _retry_chembl_request(lambda: getTar.get(tar), context=f"target {tar}")
            try:
                if tar_dict['target_type'] in ('CELL-LINE', 'UNCHECKED'):
                    continue
            except KeyError:
                continue
            data.append(d)
        ActList.append(list(data))
    named_ActList = dict(zip(chemblIds, ActList))
    named_ActList = {
        k: v
        for k, v in named_ActList.items()
        if v
    }
    return named_ActList

@tool
def getProteinsforDrugs(
    disease_id: str,
    chembl_input: Union[str, List[str]],
) -> Dict[str, Any]:
    """
    Retrieve drug-protein pairs and disease relevance scores for each protein.

    Inputs can be a single ChEMBL ID, a list of ChEMBL IDs, or a CSV path with a
    'chembl_id' column. A disease identifier (EFO/MONDO) is required.
    """
    if not disease_id:
        raise ValueError("disease_id is required.")

    ids = _chembl_ids_from_input(chembl_input)
    if not ids:
        raise ValueError("No valid ChEMBL IDs provided.")

    mech = RetMech(ids)
    act = RetAct(ids)
    target_ids = Ret_chembl_protein(mech) + Ret_chembl_protein(act)
    if target_ids:
        chembl2gene = chembl2uniprot(target_ids)
        mech = chembl2gene2path(chembl2gene, mech)
        act = chembl2gene2path(chembl2gene, act)

    rows: List[Dict[str, Any]] = []
    ensg_cache: Dict[str, Optional[str]] = {}

    for drug_id in ids:
        proteins: set[str] = set()
        for entry in mech.get(drug_id, []):
            protein = entry.get("Protein")
            if protein:
                proteins.add(protein)
        for entry in act.get(drug_id, []):
            protein = entry.get("Protein")
            if protein:
                proteins.add(protein)

        if not proteins:
            rows.append(
                {
                    "chembl_id": drug_id,
                    "protein_symbol": None,
                    "ensg_id": None,
                    "disease_id": disease_id,
                    "disease_association_score": None,
                }
            )
            continue

        for protein in sorted(proteins):
            ensg_id = _lookup_ensg_id(protein, ensg_cache)
            if ensg_id:
                score = _get_target_association_score_for_disease(ensg_id, disease_id)
            else:
                score = None
            if score is None:
                score = 0.0
            rows.append(
                {
                    "chembl_id": drug_id,
                    "protein_symbol": protein,
                    "ensg_id": ensg_id,
                    "disease_id": disease_id,
                    "disease_association_score": score,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "chembl_id",
            "protein_symbol",
            "ensg_id",
            "disease_id",
            "disease_association_score",
        ],
    )
    output_path = task_file_path("drug_protein_scores.csv")
    df.to_csv(output_path, index=False)

    return {
        "success": True,
        "data": {
            "summary": {
                "total_drugs": len(ids),
                "total_pairs": len(df),
                "unique_proteins": df["protein_symbol"].nunique() if not df.empty else 0,
            },
            "analysis_recommendation": (
                f"Use the full dataset at {output_path} for complete analysis."
            ),
        },
        "output_file": str(output_path),
        "message": f"Generated {len(df)} drug-protein rows for {len(ids)} drugs."
    }


@tool
def getMechanismofActionforDrugs(
    disease_id: str,
    chembl_input: Union[str, List[str]],
    min_protein_score: float = 0.0,
    aggregate_by_moa: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve drug-mechanism-of-action records with optional disease relevance scores.

    Inputs can be a single ChEMBL ID, a list of ChEMBL IDs, or a CSV path with a
    'chembl_id' column. A disease identifier (EFO/MONDO) is required.
    """
    if not disease_id:
        raise ValueError("disease_id is required.")
    if min_protein_score < 0:
        raise ValueError("min_protein_score must be >= 0.")

    ids = _chembl_ids_from_input(chembl_input)
    if not ids:
        raise ValueError("No valid ChEMBL IDs provided.")

    mech = RetMech(ids)
    target_ids = Ret_chembl_protein(mech)
    if target_ids:
        chembl2gene = chembl2uniprot(target_ids)
        mech = chembl2gene2path(chembl2gene, mech)

    rows: List[Dict[str, Any]] = []
    ensg_cache: Dict[str, Optional[str]] = {}

    for drug_id in ids:
        mech_entries = mech.get(drug_id, [])
        kept_rows = 0

        for entry in mech_entries:
            moa = entry.get("mechanism_of_action")
            action_type = entry.get("action_type")
            target_id = entry.get("target_chembl_id")
            protein = entry.get("Protein")

            ensg_id = _lookup_ensg_id(protein, ensg_cache) if protein else None
            if ensg_id:
                score = _get_target_association_score_for_disease(ensg_id, disease_id)
            else:
                score = None
            if score is None:
                score = 0.0

            if score < min_protein_score:
                continue

            rows.append(
                {
                    "chembl_id": drug_id,
                    "mechanism_of_action": moa,
                    "action_type": action_type,
                    "target_chembl_id": target_id,
                    "protein_symbol": protein,
                    "ensg_id": ensg_id,
                    "disease_id": disease_id,
                    "protein_association_score": score,
                    "moa_association_score": None,
                    "moa_protein_count": None,
                    "moa_protein_symbols": None,
                }
            )
            kept_rows += 1

        if not mech_entries or kept_rows == 0:
            rows.append(
                {
                    "chembl_id": drug_id,
                    "mechanism_of_action": None,
                    "action_type": None,
                    "target_chembl_id": None,
                    "protein_symbol": None,
                    "ensg_id": None,
                    "disease_id": disease_id,
                    "protein_association_score": None,
                    "moa_association_score": None,
                    "moa_protein_count": None,
                    "moa_protein_symbols": None,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "chembl_id",
            "mechanism_of_action",
            "action_type",
            "target_chembl_id",
            "protein_symbol",
            "ensg_id",
            "disease_id",
            "protein_association_score",
            "moa_association_score",
            "moa_protein_count",
            "moa_protein_symbols",
        ],
    )

    if aggregate_by_moa and not df.empty:
        valid = df.dropna(subset=["mechanism_of_action"])
        if not valid.empty:
            grouped = valid.groupby(["chembl_id", "mechanism_of_action"], dropna=False)
            agg_df = grouped.agg(
                protein_scores=("protein_association_score", list),
                protein_symbols=("protein_symbol", list),
            ).reset_index()
            agg_df["moa_association_score"] = agg_df["protein_scores"].apply(
                lambda scores: fmean(scores) if scores else 0.0
            )
            agg_df["moa_protein_symbols"] = agg_df["protein_symbols"].apply(
                lambda symbols: ";".join(sorted({s for s in symbols if s}))
                if symbols
                else None
            )
            agg_df["moa_protein_count"] = agg_df["protein_symbols"].apply(
                lambda symbols: len({s for s in symbols if s}) if symbols else 0
            )
            df = df.merge(
                agg_df[
                    [
                        "chembl_id",
                        "mechanism_of_action",
                        "moa_association_score",
                        "moa_protein_count",
                        "moa_protein_symbols",
                    ]
                ],
                on=["chembl_id", "mechanism_of_action"],
                how="left",
            )

    output_path = task_file_path("drug_mechanism_of_action_scores.csv")
    df.to_csv(output_path, index=False)

    unique_moas = df["mechanism_of_action"].nunique() if not df.empty else 0
    return {
        "success": True,
        "data": {
            "summary": {
                "total_drugs": len(ids),
                "total_rows": len(df),
                "unique_mechanisms": unique_moas,
                "min_protein_score": min_protein_score,
                "aggregate_by_moa": aggregate_by_moa,
            },
            "analysis_recommendation": (
                f"Use the full dataset at {output_path} for complete analysis."
            ),
        },
        "output_file": str(output_path),
        "message": f"Generated {len(df)} drug-mechanism rows for {len(ids)} drugs."
    }


@tool
def getPathwaysforDrugs(
    disease_id: str,
    chembl_input: Union[str, List[str]],
    min_protein_score: float = 0.3,
) -> Dict[str, Any]:
    """
    Retrieve drug-pathway associations and a pathway-level disease relevance score.

    Inputs can be a single ChEMBL ID, a list of ChEMBL IDs, or a CSV path with a
    'chembl_id' column. A disease identifier (EFO/MONDO) is required.
    """
    if not disease_id:
        raise ValueError("disease_id is required.")
    if min_protein_score < 0:
        raise ValueError("min_protein_score must be >= 0.")

    ids = _chembl_ids_from_input(chembl_input)
    if not ids:
        raise ValueError("No valid ChEMBL IDs provided.")

    mech = RetMech(ids)
    act = RetAct(ids)
    target_ids = Ret_chembl_protein(mech) + Ret_chembl_protein(act)
    target_to_pathways = chembl2uniprot(target_ids) if target_ids else {}

    rows: List[Dict[str, Any]] = []
    ensg_cache: Dict[str, Optional[str]] = {}
    score_cache: Dict[str, float] = {}

    for drug_id in ids:
        target_set: set[str] = set()
        for entry in mech.get(drug_id, []):
            target = entry.get("target_chembl_id")
            if target:
                target_set.add(target)
        for entry in act.get(drug_id, []):
            target = entry.get("target_chembl_id")
            if target:
                target_set.add(target)

        pathway_map: Dict[str, Dict[str, Any]] = {}

        for target_id in sorted(target_set):
            target_info = target_to_pathways.get(target_id)
            if not target_info:
                continue
            gene_symbol, pathways = _extract_reactome_pathways(target_info)
            if not gene_symbol or not pathways:
                continue
            if gene_symbol in score_cache:
                protein_score = score_cache[gene_symbol]
            else:
                ensg_id = _lookup_ensg_id(gene_symbol, ensg_cache)
                protein_score = (
                    _get_target_association_score_for_disease(ensg_id, disease_id)
                    if ensg_id
                    else None
                )
                if protein_score is None:
                    protein_score = 0.0
                score_cache[gene_symbol] = protein_score

            if protein_score < min_protein_score:
                continue

            for pathway in pathways:
                pathway_key = pathway.get("id") or pathway.get("name") or ""
                if not pathway_key:
                    continue
                entry = pathway_map.setdefault(
                    pathway_key,
                    {
                        "pathway_name": pathway.get("name"),
                        "pathway_id": pathway.get("id"),
                        "pathway_url": pathway.get("url"),
                        "protein_scores": {},
                    },
                )
                entry["protein_scores"][gene_symbol] = protein_score

        if not pathway_map:
            rows.append(
                {
                    "chembl_id": drug_id,
                    "pathway_name": None,
                    "pathway_id": None,
                    "pathway_url": None,
                    "disease_id": disease_id,
                    "pathway_association_score": None,
                    "protein_count": 0,
                    "protein_symbols": None,
                }
            )
            continue

        for entry in sorted(pathway_map.values(), key=lambda x: x["pathway_name"] or ""):
            protein_scores = list(entry["protein_scores"].values())
            pathway_score = fmean(protein_scores) if protein_scores else 0.0
            proteins = sorted(entry["protein_scores"].keys())
            rows.append(
                {
                    "chembl_id": drug_id,
                    "pathway_name": entry["pathway_name"],
                    "pathway_id": entry["pathway_id"],
                    "pathway_url": entry["pathway_url"],
                    "disease_id": disease_id,
                    "pathway_association_score": pathway_score,
                    "protein_count": len(proteins),
                    "protein_symbols": ";".join(proteins) if proteins else None,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "chembl_id",
            "pathway_name",
            "pathway_id",
            "pathway_url",
            "disease_id",
            "pathway_association_score",
            "protein_count",
            "protein_symbols",
        ],
    )
    output_path = task_file_path("drug_pathway_scores.csv")
    df.to_csv(output_path, index=False)

    unique_pathways = df["pathway_name"].nunique() if not df.empty else 0
    return {
        "success": True,
        "data": {
            "summary": {
                "total_drugs": len(ids),
                "total_pairs": len(df),
                "unique_pathways": unique_pathways,
                "min_protein_score": min_protein_score,
            },
            "analysis_recommendation": (
                f"Use the full dataset at {output_path} for complete analysis."
            ),
        },
        "output_file": str(output_path),
        "message": f"Generated {len(df)} drug-pathway rows for {len(ids)} drugs."
    }


@tool
def literature_search_pubmed(query: str, limit: int = 5) -> str:
    """Search scientific literature via the LitSense API."""
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
        logger.error("Error in literature_search_pubmed: %s", exc)
        return f"Error retrieving literature for '{query}': {exc}. Please try again."


@tool
def protocol_search_sop(query: str) -> str:
    """Search SOP documents for protocols and regulatory procedures."""
    retriever = SOPRetriever()

    try:
        documents = retriever.retriever.invoke(query)
        documents = retriever._convert_bytes_to_docs(documents)

        if not documents:
            return f"No SOP content found for query '{query}'."

        return _format_sop_results(documents)

    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error processing SOP query '%s': %s", query, exc)
        return f"Error retrieving SOP content for '{query}': {exc}. Please try again."


__all__ = [
    "LitSenseObject",
    "PyLitSense",
    "literature_search_pubmed",
    "protocol_search_sop",
    "getProteinsforDrugs",
    "getMechanismofActionforDrugs",
    "getPathwaysforDrugs",
]
