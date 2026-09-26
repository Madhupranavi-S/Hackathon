# Business Entity Resolution: pipeline overview

**Task:** for every Source 1 business record, find all Source 2 / Source 3 records that
describe the same real business.

**Official metric:** F0.5 computed **per Source 1 entity, then averaged** over all entities,
singletons included (empty prediction for a true singleton = 1.0, any match = 0.0).

**Data size:** train 2.2M Source 1 + 10.3M Source 2/3 records; test 1.7M + 10.0M.
Test also contains **France**, which never appears in training.

**Final ranking also rewards small candidate sets** (`candidate_pairs.tsv` is reviewed).

---

## Results so far

| Version | Validation (official metric) | Leaderboard |
|---|---|---|
| v1: first model, pair-tuned threshold | 0.9610 | 0.952 |
| v2: + second-stage model | 0.9637 | 0.952 |
| probe: v2 with France threshold 0.95 | n/a | 0.953 |
| v3: French fixes + rarity features + candidate filter + per-entity decisions | see `work/finalize.txt` | predicted in `work/predict_lb.txt` |

What we learned: validation was first measured with a pair-level F0.5 (0.976), which
overstates the official per-entity score (0.964). The remaining gap to the leaderboard is
mostly France: the probe showed French probabilities were over-confident.

---

## How the pipeline works

```
raw TSV files
 | preprocess.py          normalize names/addresses once (parallel) -> Parquet
 v
 | generate_candidates.py BLOCKING: ~28 candidates per Source 1 record (97.7% of true matches kept)
 v
 | run_pipeline.py        FIRST MODEL: 35 features per pair -> LightGBM -> probability p1
 v
 | choose_prune.py        CANDIDATE FILTER: keep pairs with p1 >= tau (chosen on validation
 |                        so the official score loses <= 0.0005)  -> candidate_pairs.tsv
 v
 | run_stage2.py          SECOND MODEL: runs only on the filtered pairs, adds the first
 |                        model's confidence about each entity's other candidates
 v
 | finalize.py            DECISIONS per entity, maximising expected official F0.5
 v                        -> output_final/matching_results.tsv
 | predict_lb.py          predicted leaderboard score before uploading
```

### 1. Normalization (`common.py`, `preprocess.py`)
- Any script (Telugu, Hindi, Tamil, accented French) is transliterated to Latin (`anyascii`).
- Core name: legal suffixes (Pvt Ltd, LLC, SARL, SAS and transliterated forms), generic
  words (Holdings, Services) and the record's own country name are removed.
- Digits used as letters are repaired (c0nsultants -> consultants).
- Addresses: abbreviations unified to short forms (Road -> rd, Rue -> r, Saint/Street -> st),
  US/Indian state names -> codes, French regions/departments removed, "Door No / Plot No /
  # / N° / bis / ter" removed, ordinals (eleventh, 11th) -> 11, leading zeros removed,
  French filler words (de, du, des, le, les) removed, native-script state names dropped.

### 2. Blocking (`blocking_eval.py` = experiments + shared code, `generate_candidates.py`)
- Each record becomes a sparse TF-IDF vector of tokens: name words, glued name
  (digitaltech), address words, house numbers, **combinations** (adjacent address words
  like `nehru_nagar`, name word x house number like `tech#53`) and **sound keys**
  (consonant skeletons: `southern` ~ `sdrn`, `white` ~ `vhait`).
- Tokens in more than 5,000 records are ignored (keeps search fast at this scale).
- Search runs **within each country** (true matches never cross countries).
- **Forward** top-20 per Source 1 record + **reverse** top-3 per Source 2/3 record
  (Source 1 is deduplicated, so a pool record's correct Source 1 is usually its #1).
- Blocking recall: India 96.9%, US 98.3%.

### 3. First model (`matcher_features.py`, `run_pipeline.py`)
Features per pair: blocking score and ranks, context gaps (how much better than the
alternatives), name/address string similarities (rapidfuzz), **rarity-weighted name and
address overlap** (sharing "baobab" counts, sharing "club" or "roubaix" barely),
house-number agreement/conflict, address-word overlap, sound-key overlap, lengths.
LightGBM on 300k Source 1 entities, two configurations compared automatically.

### 4. Candidate filter (`choose_prune.py`)
The first model acts as a filter: pairs with p1 below tau are dropped. tau is the largest
value whose loss on the official metric (valA) is at most 0.0005. The filtered set is
exactly what the final model scores, and is written to `candidate_pairs.tsv`.

### 5. Second model (`run_stage2.py`)
Scores only the filtered pairs. Adds evidence from the entity's other candidates: the
first model's confidence about them and how similar this record is to the confident ones.
Trained on 300k "fresh" entities the first model never saw (so its probabilities are
realistic).

### 6. Decisions (`finalize.py`)
For each entity, accept the top-k candidates with k chosen to maximise the entity's
**expected** F0.5 (k = 0 means "predict singleton"). This fits the official per-entity
metric better than one global threshold; both are compared on validation and the better
one is used. France (unseen) probabilities are sharpened with p -> p^5, calibrated on the
leaderboard probe. The one-to-one rule (each Source 2/3 record matches at most one
Source 1 entity, confirmed on all 7.6M training pairs) is always applied.

### Honest validation
Training Source 1 entities are split by entity: training data, **valA** (early stopping,
thresholds, every choice) and **valB** (never used for a decision: the estimate to trust).
All validation scores use the official per-entity metric (checked against the example on
the challenge page).

---

## Files

| File | Purpose |
|---|---|
| `common.py` | paths, reading/writing, text normalization, folds, scoring |
| `preprocess.py` | normalize all six source files in parallel, save Parquet |
| `explore.py` | data report (coverage, what true matches share, examples) |
| `blocking_eval.py` | blocking experiments with recall measurement + shared token/matrix code |
| `generate_candidates.py` | blocking for every record, saves candidate pairs |
| `matcher_features.py` | feature computation for millions of pairs |
| `run_pipeline.py` | first model: features, training, test prediction, v1 outputs |
| `choose_prune.py` | chooses the candidate-filter cutoff |
| `run_stage2.py` | second model (on filtered candidates), v2 outputs |
| `finalize.py` | per-entity decisions for the official metric, `output_final/`, checks |
| `predict_lb.py` | predicted leaderboard score of `output_final/` |
| `run_all.py` | one command: preprocess -> candidates -> first model -> stage 2 -> diagnose |
| `run_final.py` | one command: filter cutoff -> stage 2 -> finalize -> official validator |
| `diagnose.py` | metric definitions, confidence per country, French examples, probes |
| `make_probe.py` | probe submissions with a different threshold for unseen countries |
| `requirements.txt` | Python packages |

---

## How to reproduce (Windows, from the project folder)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Put the challenge data in `dataset/train/` and `dataset/test/`, copy the organisers'
`utils/validate_submission.py` into `utils/`, then:

| Command | What it does | Time (12 cores, 16 GB) |
|---|---|---|
| `python -u run_all.py` | preprocess, blocking, first model, stage 2, diagnose | ~2.5 h |
| `python -u run_final.py` | candidate filter, stage 2 on filtered set, final decisions, validation | ~30 min |
| `python -u predict_lb.py` | predicted leaderboard score | ~5 min |

Every script can be re-run safely: finished parts are skipped. Logs and summaries are in
`work/` (`pipeline.log`, `report.md`, `stage2.log`, `report_stage2.md`, `prune.txt`,
`finalize.txt`, `predict_lb.txt`, `diagnose.txt`). Previous outputs are moved to
`backups/<date-time>/` by `run_all.py`.

**Final outputs:** `output_final/matching_results.tsv` (upload this) and
`output_final/candidate_pairs.tsv`.

---

## Constraints check
- Models: LightGBM (MIT license), no pretrained model, far below 8B parameters.
- Country is treated as an open set of labels: nothing is hard-coded or filtered to
  US/India; France goes through the same pipeline (plus a calibration correction).
- Every test Source 1 entity gets exactly one row; matches are always a subset of
  candidates (checked by `finalize.py` and the organisers' validator).

## Possible next steps
- Contrastive encoder (multilingual-e5, MIT) as an extra retriever/feature for very
  different names and cross-script names.
- Better France handling with a few hundred hand-labelled French pairs.
