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
| **v3 (`output_final_baseline`)**: French fixes, rarity features, candidate filter (3.8 candidates per entity instead of 28), per-entity decisions | 0.9646-0.9661 | **0.956** |
| v4: v3 + neural cross-encoder | pending (`work/finalize.txt`) | predicted in `work/predict_lb.txt` |

**Score by country (v3):** the leaderboard is a per-entity average, so it splits by each
country's share of test entities. India and US are measured on validation (real labels);
France follows from the leaderboard score:

| Country | Share of test entities | Score | Points lost vs. perfect |
|---|---|---|---|
| India | 0.468 | 0.955 (validation) | 0.021 |
| US | 0.383 | 0.971 (validation) | 0.011 |
| France | 0.150 | ~0.915 (= (0.956 - 0.8188) / 0.150) | 0.013 |

India/US validation and the leaderboard agree exactly, so validation is trustworthy for
those countries. France went from ~0.89 (v2) to ~0.915 (v3) after the French fixes.

**What we learned along the way**
- The official metric is per entity. A pair-level F0.5 (0.976) overstated it (0.964).
  Under a per-entity average, missed matches cost much more: finding 3 of 4 true matches
  scores only 0.94 for that entity.
- The model's own confidence overshoots France by ~0.063 (estimated 0.978, real 0.915);
  `predict_lb.py` corrects new France estimates by this measured offset.
- Error analysis on validation (`error_analysis.py`, v3): the 0.034 gap splits evenly into
  false merges (+0.011 if all removed), blocking misses (+0.009), rejected true matches
  (+0.009) and matches cut by the candidate filter (+0.007).
  - 61% of false merges are **decoys**: records that belong to no entity but closely
    imitate one (`YNL Cyber` vs `VYNL Cyber`, `35/8-Ba-1` vs `35/8-Ba-8`).
  - 38% of false merges involve a record that belongs to another entity outside the
    validation sample; on the full test set that entity competes for the record, so
    validation is somewhat pessimistic on precision.
  - 24-43% of each error type involve a Source 2/3 record with an **empty address**,
    where the name is the only evidence.

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
 | neural_ce.py           (optional) NEURAL CROSS-ENCODER scores every filtered pair;
 |                        its score becomes extra evidence for stage 2
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

### 6. Neural cross-encoder (`neural_ce.py`, optional)
A multilingual transformer (`intfloat/multilingual-e5-small`, MIT, 118M parameters) reads
both RAW records together (Ditto-style) and outputs a match probability. Being pretrained
on ~100 languages, it relates scripts, transliterations and French address words that the
rule-based features only approximate. Trained on the first model's training entities
(matches + hardest blocking negatives), it scores only the filtered candidates of the
fresh / valA / valB / test entities, which it never saw. Its score, rank and gap within
the entity are added to stage 2; stage 2's validation decides how much it helps.
Training: 1.2M pairs (540k matches, 660k hardest non-matches), 1 epoch, batch 64,
learning rate 3e-5 with warm-up and linear decay, mixed precision. On validation hard
pairs it reached 98.8% accuracy (logloss 0.032). On an RTX 3050 Laptop GPU: training
~1.5 h (~190 pairs/s), scoring ~7.8M filtered pairs ~3.5 h (~600 pairs/s).

### 7. Decisions (`finalize.py`)
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
| `run_stage2.py` | second model (on filtered candidates, with cross-encoder features if available), v2 outputs |
| `neural_ce.py` | neural cross-encoder: training on the GPU, scoring all filtered pairs |
| `run_neural.py` | one command: filter cutoff -> cross-encoder -> stage 2 -> finalize -> validator -> predicted score |
| `finalize.py` | per-entity decisions for the official metric, `output_final/`, checks |
| `predict_lb.py` | predicted leaderboard score of `output_final/` (France calibrated on the 0.956 result) |
| `run_all.py` | one command: preprocess -> candidates -> first model -> stage 2 -> diagnose |
| `run_final.py` | one command: filter cutoff -> stage 2 -> finalize -> official validator |
| `diagnose.py` | metric definitions, confidence per country, French examples, probes |
| `error_analysis.py` | where the score is lost on valB: oracle gains per error type + examples |
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
| `python -u run_neural.py` | (optional) cross-encoder + stage 2 + final package + predicted score | ~5 h on an RTX 3050 |
| `python -u error_analysis.py` | where the validation score is lost, with examples | ~5 min |

The neural track needs PyTorch with CUDA. With an NVIDIA driver supporting CUDA 12.x:
```
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir --resume-retries 20
pip install transformers
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last command must print `True` and the GPU name. `--resume-retries` lets the 2.6 GB
download continue after connection drops.

Every script can be re-run safely: finished parts are skipped. Logs and summaries are in
`work/` (`pipeline.log`, `report.md`, `stage2.log`, `report_stage2.md`, `prune.txt`,
`finalize.txt`, `predict_lb.txt`, `diagnose.txt`). Previous outputs are moved to
`backups/<date-time>/` by `run_all.py`.

**Final outputs:** `output_final/matching_results.tsv` (upload this) and
`output_final/candidate_pairs.tsv`.

---

## Submission workflow

Submissions are limited, so nothing is uploaded without an offline estimate first:
1. `finalize.py` compares setups on valA and reports the untouched valB score.
2. `predict_lb.py` predicts the leaderboard: India/US from valB, France from the model's
   estimate corrected by the measured offset. Its method reproduces the real 0.956 of v3.
3. A new version is submitted only if its predicted range lies clearly above the best
   score so far. `output_final_baseline/` (0.956) is kept as the fallback.
4. The organisers' validator (`utils/validate_submission.py`) must print PASS.

---

## Constraints check
- Models: LightGBM (MIT license) and, optionally, multilingual-e5-small (MIT license,
  118M parameters), far below the 8B limit.
- Country is treated as an open set of labels: nothing is hard-coded or filtered to
  US/India; France goes through the same pipeline (plus a calibration correction).
- Every test Source 1 entity gets exactly one row; matches are always a subset of
  candidates (checked by `finalize.py` and the organisers' validator).

## Possible next steps
Ordered by expected gain, based on the error analysis:
- **Competition features:** score every training pair with the first model, so each
  Source 2/3 record's competing entities can be compared (as on the test set). Targets
  records claimed by several lookalike entities, and makes validation realistic.
- **Name-only fallback retriever** for records with an empty address (character n-grams,
  typo-tolerant): targets part of the 3,951 blocking misses on valB.
- **Fellegi-Sunter probabilistic linkage fitted with EM on France** (no labels needed):
  a France-specific calibration and feature, replacing the p^5 correction.
- **Contrastive bi-encoder** as an extra blocking retriever (recall ceiling 97.7%).
