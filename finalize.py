"""FINAL STEP: choose matches per entity to maximise the OFFICIAL metric, then write and
check the submission package.

    python -u finalize.py

Official metric: F0.5 computed per Source 1 entity, then averaged over ALL entities.
A singleton scores 1.0 for an empty prediction and 0.0 for any match.

Compared on valA (choices) and valB (untouched estimate), for each available model
(first model, stage 2):
  threshold   one global probability threshold (what we used so far), tuned for the
              official metric instead of the pair-level one
  expected_f  per entity, accept the top-k candidates, with k chosen to maximise the
              entity's EXPECTED F0.5 given the model's probabilities (k = 0 means
              "predict singleton"). Uses E[F] ~ 1.25*E[TP] / (k + 0.25*E[T]).
The best (model, rule) on valA is applied to the test set.

Unseen countries (France): the leaderboard probe showed their probabilities are too high
(pairs scored 0.77-0.95 were correct less than ~76% of the time), so they are sharpened
with p -> p**UNSEEN_POWER before deciding (0.95 -> 0.77, 0.90 -> 0.59).

Writes output_final/ (matching_results.tsv + candidate_pairs.tsv), checks it, and saves
a summary to work/finalize.txt. Works on whatever models are currently in work/.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA_DIR, read_tsv, up_to_date
from preprocess import PREP_DIR
from run_pipeline import FEAT_DIR, MODEL_DIR, PRED_DIR, WORK, countries_of
from run_stage2 import PRED2_DIR, S2_DIR, write_candidates

OUT = DATA_DIR.parent / "output_final"
UNSEEN_POWER = 5.0
THRESH_GRID = np.round(np.arange(0.10, 0.97, 0.01), 2)
MISS_GRID = (0.0, 0.02, 0.05, 0.10)     # share of true matches blocking never retrieves
LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


# ---------------------------------------------------------------- scoring (official metric)
def macro_f05(pred, truth, entities):
    """pred, truth: DataFrames (s1, m). Per-entity F0.5 averaged over `entities`."""
    ents = pd.Index(entities)
    n_pred = pred.groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    n_true = truth.groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    tp = pred.merge(truth, on=["s1", "m"]).groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    both_empty = (n_pred == 0) & (n_true == 0)
    denom = n_pred + 0.25 * n_true
    f = np.where(both_empty, 1.0, np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-9), 0.0))
    return float(f.mean())


# ---------------------------------------------------------------- decision rules
def one_to_one(d):
    """Each S2/S3 record keeps only its most probable S1 entity."""
    return d.sort_values("prob", ascending=False).drop_duplicates("cand_id")


def rule_threshold(d, t):
    a = d[d["prob"] >= t]
    return a.rename(columns={"s1_id": "s1", "cand_id": "m"})[["s1", "m"]]


def rule_expected_f(d, miss):
    """Per entity, keep the top-k candidates maximising 1.25*cumP_k / (k + 0.25*E[T]);
    k=0 (empty) has expected F = P(no true match) ~ prod(1-p) * (no match missed by blocking)."""
    d = d.sort_values(["s1_id", "prob"], ascending=[True, False])
    p = np.clip(d["prob"].to_numpy(np.float64), 1e-6, 1 - 1e-6)
    code, uniq = pd.factorize(d["s1_id"])
    new = np.r_[True, code[1:] != code[:-1]]
    start = np.maximum.accumulate(np.where(new, np.arange(len(p)), 0))
    k = np.arange(len(p)) - start + 1                     # 1-based position within entity
    csum = np.cumsum(p)
    cum_p = csum - np.r_[0.0, csum][start]                # expected TP of the top-k
    total_p = np.bincount(code, weights=p)
    exp_t = total_p * (1 + miss)                          # expected number of true matches
    f_k = 1.25 * cum_p / (k + 0.25 * exp_t[code])
    log_empty = np.bincount(code, weights=np.log1p(-p))
    f_empty = np.exp(log_empty) * np.exp(-miss * total_p)
    best_k = np.zeros(len(uniq), dtype=np.int64)
    best_f = f_empty.copy()
    # best k per entity = argmax over positions
    order = np.lexsort((-f_k, code))
    first = np.r_[True, code[order][1:] != code[order][:-1]]
    top_rows = order[first]
    better = f_k[top_rows] > best_f[code[top_rows]]
    best_k[code[top_rows][better]] = k[top_rows][better]
    keep = k <= best_k[code]
    a = d[keep]
    return a.rename(columns={"s1_id": "s1", "cand_id": "m"})[["s1", "m"]]


def adjust(d, seen):
    """Sharpen probabilities of countries never seen in training."""
    unseen = ~d["country"].isin(seen)
    if unseen.any():
        d = d.copy()
        d.loc[unseen, "prob"] = d.loc[unseen, "prob"] ** UNSEEN_POWER
    return d


# ---------------------------------------------------------------- models on validation
def available_models():
    models = {}
    b1 = json.loads((MODEL_DIR / "best.json").read_text())
    models["first model"] = dict(
        booster=lgb.Booster(model_file=str(MODEL_DIR / f"{b1['experiment']}.txt")),
        cols=b1["feature_cols"], val={v: FEAT_DIR / f"{v}.parquet" for v in ("valA", "valB")},
        test_dir=PRED_DIR, model_files=[MODEL_DIR / "best.json"])
    p2 = MODEL_DIR / "best_stage2.json"
    if p2.exists() and (MODEL_DIR / "stage2.txt").exists():
        b2 = json.loads(p2.read_text())
        models["stage 2"] = dict(
            booster=lgb.Booster(model_file=str(MODEL_DIR / "stage2.txt")),
            cols=b2["feature_cols"], val={v: S2_DIR / f"{v}_s2.parquet" for v in ("valA", "valB")},
            test_dir=PRED2_DIR, model_files=[MODEL_DIR / "best_stage2.json", MODEL_DIR / "stage2.txt"])
    # with the cascade active, the first model is the candidate filter; only stage 2 may decide
    from run_stage2 import prune_tau
    if prune_tau() > 0:
        models.pop("first model", None)
    # a model is only usable if its test predictions exist for every test country
    usable = {}
    for k, v in models.items():
        preds = [v["test_dir"] / f"test_{c}.parquet" for c in countries_of("test")]
        if all(up_to_date([p], v["model_files"]) for p in preds):
            usable[k] = v
        else:
            print(f"  (skipping '{k}': its test predictions are missing or older than the model)")
    return usable


def main():
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    seen = set(countries_of("train"))
    models = available_models()
    if not models:
        sys.exit("no model with complete test predictions found in work/")
    log(f"models with complete test predictions: {list(models)}")

    results = []
    for name, m in models.items():
        val = {}
        for v in ("valA", "valB"):
            f = pd.read_parquet(m["val"][v])
            prob = m["booster"].predict(f[m["cols"]].to_numpy(np.float32))
            d = f[["s1_id", "cand_id", "country"]].assign(prob=prob)
            t = truth[truth["part"] == v][["s1", "m"]]
            ents = sorted(set(f["s1_id"]) | set(t["s1"]))
            val[v] = (one_to_one(d), t, ents)
        (dA, tA, eA), (dB, tB, eB) = val["valA"], val["valB"]

        best_t = max(THRESH_GRID, key=lambda t: macro_f05(rule_threshold(dA, t), tA, eA))
        results.append(dict(model=name, rule="threshold", param=float(best_t),
                            valA=macro_f05(rule_threshold(dA, best_t), tA, eA),
                            valB=macro_f05(rule_threshold(dB, best_t), tB, eB)))
        best_m = max(MISS_GRID, key=lambda x: macro_f05(rule_expected_f(dA, x), tA, eA))
        results.append(dict(model=name, rule="expected_f", param=float(best_m),
                            valA=macro_f05(rule_expected_f(dA, best_m), tA, eA),
                            valB=macro_f05(rule_expected_f(dB, best_m), tB, eB)))

    log("\n=== official metric (per-entity F0.5, singletons included) ===")
    log(f"  {'model':12} {'rule':11} {'param':>6}   {'valA':>7}   {'valB (untouched)':>16}")
    for r in results:
        log(f"  {r['model']:12} {r['rule']:11} {r['param']:6.2f}   {r['valA']:.4f}   {r['valB']:.4f}")
    best = max(results, key=lambda r: r["valA"])
    log(f"\nCHOSEN on valA: {best['model']} + {best['rule']} (param {best['param']:.2f}); "
        f"expected leaderboard for India/US-like data: {best['valB']:.4f}")
    log(f"unseen countries: probabilities sharpened with p**{UNSEEN_POWER:g}")
    (WORK / "final_choice.json").write_text(json.dumps(
        {k: best[k] for k in ("model", "rule", "param", "valB")}, indent=2))

    # ---- apply to test
    m = models[best["model"]]
    OUT.mkdir(exist_ok=True)
    path, tmp = OUT / "matching_results.tsv", OUT / "matching_results.tmp"
    log("\n=== test ===")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for c in countries_of("test"):
            s1_ids = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"],
                                     filters=[("country_key", "=", c)])["entity_id"]
            d = pq.read_table(m["test_dir"] / f"test_{c}.parquet").to_pandas().assign(country=c)
            d = one_to_one(adjust(d, seen))
            a = rule_threshold(d, best["param"]) if best["rule"] == "threshold" \
                else rule_expected_f(d, best["param"])
            lists = a.groupby("s1")["m"].agg(lambda x: ",".join(sorted(x)))
            col = lists.reindex(s1_ids).fillna("").to_numpy(object)
            fm.write("".join(f"{x}\t{y}\n" for x, y in zip(s1_ids.to_numpy(object), col)))
            log(f"  {c:8} {'(unseen)' if c not in seen else '':8} S1={len(s1_ids):,}  "
                f"predicted singletons={(col == '').mean():.1%}  matched pairs={len(a):,} "
                f"({len(a) / len(s1_ids):.2f} per S1)")
    tmp.replace(path)
    log("  (train ground truth: 5.6% singletons, 3.46 matches per S1)")

    write_candidates(OUT / "candidate_pairs.tsv")
    check()
    (WORK / "finalize.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


def check():
    s1 = set(pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"])
    res = read_tsv(OUT / "matching_results.tsv")
    cand = read_tsv(OUT / "candidate_pairs.tsv")
    problems = []
    for name, df in (("matching_results", res), ("candidate_pairs", cand)):
        if df["source1_entity_id"].duplicated().any() or len(df) != len(s1) \
                or set(df["source1_entity_id"]) != s1:
            problems.append(f"{name}: rows do not match the test Source 1 ids exactly once")
    allowed = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
    bad = sum(len(set(ids := lst.split(","))) != len(ids)
              or any(x not in set(allowed.get(sid, "").split(",")) for x in ids)
              for sid, lst in zip(res["source1_entity_id"], res["matched_entity_ids"]) if lst)
    if bad:
        problems.append(f"{bad} rows with duplicate ids or ids not in candidate_pairs.tsv")
    log("\nFINAL PACKAGE CHECK: " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
    log(f"-> upload {OUT / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
