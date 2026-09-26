"""Choose the candidate filter cutoff tau (first-model probability) for the cascade

    blocking  ->  first model keeps pairs with p1 >= tau  (= candidate_pairs.tsv)
              ->  stage 2 scores only those pairs          (= matching_results.tsv)

The challenge ranks smaller candidate sets higher, so we want tau as LARGE as possible
while losing (almost) nothing on the official metric. For each tau on valA we remove
pairs with p1 < tau, apply the per-entity expected-F rule to the current stage-2
probabilities, and keep the largest tau whose loss is <= MAX_LOSS. Test-set candidate
counts per entity are reported for every tau. Writes work/prune.json.

    python -u choose_prune.py
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from finalize import macro_f05, one_to_one, rule_expected_f
from run_pipeline import FEAT_DIR, MODEL_DIR, PRED_DIR, WORK, countries_of
from run_stage2 import S2_DIR

TAUS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
MAX_LOSS = 0.0005
MISS = 0.02


def main():
    b2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
    if b2.get("prune", 0) > 0:
        raise SystemExit("the current stage-2 model was already trained on a filtered set; "
                         "choose_prune needs the unfiltered one (rerun run_stage2.py with prune 0)")
    booster = lgb.Booster(model_file=str(MODEL_DIR / "stage2.txt"))
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    lines = []

    def log(m=""):
        print(m, flush=True)
        lines.append(m)

    val = {}
    for v in ("valA", "valB"):
        f = pd.read_parquet(S2_DIR / f"{v}_s2.parquet")
        prob = booster.predict(f[b2["feature_cols"]].to_numpy(np.float32))
        d = f[["s1_id", "cand_id", "country", "p1"]].assign(prob=prob)
        t = truth[truth["part"] == v][["s1", "m"]]
        val[v] = (d, t, sorted(set(f["s1_id"]) | set(t["s1"])))

    test_p1 = {c: pq.read_table(PRED_DIR / f"test_{c}.parquet", columns=["s1_id", "prob"])
               for c in countries_of("test")}
    n_test_s1 = {c: pc_unique(t) for c, t in test_p1.items()}

    log(f"{'tau':>6} {'valA':>8} {'loss':>8} {'valB':>8}  {'val cand/S1':>11}  "
        f"{'true pairs kept':>15}  test cand/S1 per country")
    rows = []
    for tau in TAUS:
        res = {}
        for v, (d, t, ents) in val.items():
            dp = d[d["p1"] >= tau]
            res[v] = macro_f05(rule_expected_f(one_to_one(dp), MISS), t, ents)
            if v == "valA":
                size = len(dp) / len(ents)
                kept = dp.merge(t, left_on=["s1_id", "cand_id"], right_on=["s1", "m"]).shape[0] / \
                    d.merge(t, left_on=["s1_id", "cand_id"], right_on=["s1", "m"]).shape[0]
        tsize = {c: int((np.asarray(tb.column("prob")) >= tau).sum()) / n_test_s1[c]
                 for c, tb in test_p1.items()}
        rows.append((tau, res["valA"], res["valB"], size, kept, tsize))
    base = rows[0][1]
    for tau, a, b, size, kept, tsize in rows:
        log(f"{tau:6.3f} {a:8.4f} {base - a:8.4f} {b:8.4f}  {size:11.2f}  {kept:15.2%}  "
            + "  ".join(f"{c}={s:.2f}" for c, s in tsize.items()))
    ok = [r for r in rows if base - r[1] <= MAX_LOSS]
    chosen = max(ok, key=lambda r: r[0])
    log(f"\nCHOSEN tau={chosen[0]} (largest with valA loss <= {MAX_LOSS}): "
        f"~{np.mean(list(chosen[5].values())):.1f} candidates per test entity "
        f"instead of ~{np.mean(list(rows[0][5].values())):.1f}")
    (WORK / "prune.json").write_text(json.dumps({"tau": chosen[0]}, indent=2))
    (WORK / "prune.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pc_unique(table):
    import pyarrow.compute as pc
    return pc.count_distinct(table.column("s1_id")).as_py()


if __name__ == "__main__":
    main()
