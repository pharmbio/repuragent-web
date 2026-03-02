# Scope (single integrated Data Agent step)

Load the uploaded combo dataset, clean/harmonize keys, compute a composite ranking, add a multi-source novelty signal (PubMed + ClinicalTrials.gov), and apply an explicit diversification/exclusion rule to avoid repeatedly testing the same compounds across many pairs. Save all outputs + an API query log for traceability.

---

# Inputs used

**Excel:**
`/home/repuragent/app/persistence/data/7b87215a-cefd-46b5-be7b-ab162989611b/45cfee29-9ac7-4fe1-8be2-6f6dd5ae0bc4/combo_two_results_annotated_EA_20260223_071249.xlsx` (Sheet1)

**Rows/cols:** 3790 × 12

* `type = single` (892)
* `type = combo` (2898)

Combo pairs were already unique by** **`(name_1,name_2)` ignoring order (no A/B vs B/A duplicates detected)

---

# What I did (methods)

## 1) Harmonization

Built a canonical** **`pair_key` for combos:

```
sorted([name_1, name_2]) (casefolded)
```

to ensure uniqueness.

---

## 2) Complementarity proxy (target/MoA-based)

Because we do not have a full host-interactome module mapping in the file, I computed a target-set complementarity proxy:

* Parse** **`target_1` and** **`target_2` into gene/protein sets (split on** **`; | ,`)
* `complementarity = 1 - (|overlap| / |union|)` (0..1)

In this dataset, complementarity is often 1.0 because many pairs have disjoint annotated targets.

---

## 3) Novelty scoring (flexible, multi-source; no single-API dependency)

I implemented novelty evidence using two independent external sources:

### PubMed (NCBI E-utilities)

Count of papers mentioning both drugs in title/abstract

**Query pattern:**

```
("DrugA"[Title/Abstract] AND "DrugB"[Title/Abstract])
```

---

### ClinicalTrials.gov API v2

Presence of registered studies mentioning both drugs + COVID terms

**Query pattern:**

```
(DrugA) AND (DrugB) AND (COVID-19 OR SARS-CoV-2)
```

ClinicalTrials.gov v2 does not directly return a total count in the minimal call, so I recorded:

* 0 if no studies returned
* 1 if at least one study returned
* 2 if** **`nextPageToken` present (indicating more pages; “>1” approximate)

**Practical constraint:** to keep runtime reasonable, I queried novelty evidence for the top 200 combos by raw experimental score, then merged these novelty metrics back into the full combo table; non-queried pairs get median-imputed novelty scores (and -1 sentinel in the raw count columns).

**Novelty score used:**

```
novelty_score = 1 / (1 + log(1 + pubmed_pair_count))
```

then downweighted if trials exist:

```
novelty_score *= 1 / (1 + ctgov_pair_count_approx)
```

---

## 4) Composite ranking score

Computed:

```
score_z = z-score of experimental score among combos (higher assumed better)
```

```
composite_score = 0.6*score_z + 0.2*novelty_score + 0.2*complementarity
```

---

## 5) Exclusion/diversification to avoid repeating compounds

Created a greedy diversified selection of the top list:

* Traverse combos in descending** **`composite_score`
* Keep a combo only if each drug has been used < 3 times already (`max_per_drug = 3`)
* Output the first 100 passing this rule

This directly enforces your requirement: exclude/remove combinations from the top ranked list to avoid testing the same compounds many times in different combinations.

---

# Deliverables written (with paths)

## Filtered/harmonized combo dataset

`/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/45cfee29-9ac7-4fe1-8be2-6f6dd5ae0bc4/filtered_input.csv`

---

## Full ranked list with component scores

`/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/45cfee29-9ac7-4fe1-8be2-6f6dd5ae0bc4/ranked_results.csv`

---

## Novelty evidence table (top 200 queried)

`/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/45cfee29-9ac7-4fe1-8be2-6f6dd5ae0bc4/novelty_evidence.csv`

---

## Raw API query log (all PubMed + ClinicalTrials queries/errors)

`/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/45cfee29-9ac7-4fe1-8be2-6f6dd5ae0bc4/api_queries_log.csv`

---

## Diversified top-100 list (max 3 appearances per drug)

`/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/45cfee29-9ac7-4fe1-8be2-6f6dd5ae0bc4/ranked_selected_diversified_top100.csv`

---

# Notable observations / limitations (for next workflow step)

The very top raw-score combos are heavily dominated by SGI-1027 combinations; the diversification step is therefore important to prevent over-testing that same compound.

PubMed “both drugs in title/abstract” is a strict criterion and may undercount prior work (e.g., supplementary material, different naming, or general “combination therapy” without both names in abstract).

ClinicalTrials.gov counts are approximate (0/≥1/“more than one page”) due to API v2 pagination behavior.

Patent/DrugBank/ChEMBL combo-specific evidence was not added yet (DrugBank is not openly queryable; patents require additional sources). If you want, I can extend novelty evidence with:

* synonym normalization (RxNorm-like, MeSH, fuzzy matching)
* ChEMBL mechanism/target overlap checks
* broader PubMed query strategies (including synonyms and “co-administered” terms)

If you want a stricter exclusion policy (e.g., max 1 appearance per drug or ban certain MoA classes), tell me the rule and I’ll regenerate the diversified ranked list.
