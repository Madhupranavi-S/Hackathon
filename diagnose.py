"""Why is the leaderboard (0.952) lower than validation (0.976)? Three checks, ~5 minutes.

    python -u diagnose.py

1. Scores valB under several F0.5 definitions a leaderboard might use. If one of them
   lands near 0.952, the gap is just a different formula.
2. Compares how CONFIDENT the model is on valB vs each test country. A country where many
   accepted matches sit just above the threshold is one where the model is unsure.
3. Prints French predicted matches (borderline and confident) to eyeball for mistakes.
Writes work/diagnose.txt. Read-only otherwise.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import SEED
from preprocess import PREP_DIR
from run_pipeline import FEAT_DIR, MODEL_DIR, WORK, countries_of, keep_best_per_cand
from run_stage2 import PRED2_DIR, S2_DIR
from run_pipeline import PRED_DIR

OUT = []


def log(msg=""):
    print(msg, flush=True)
    OUT.append(msg)


def f05(p, r):
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def metric_variants(pred, truth):
    """pred, truth: DataFrames (s1, m). Returns {name: F0.5}."""
    P = pred.groupby("s1")["m"].apply(set)
    T = truth.groupby("s1")["m"].apply(set)
    ents = sorted(set(truth["s1"]) | set(pred["s1"]) | set(ENTITIES))
    tp = n_pred = n_true = 0
    ctp = cpred = ctrue = 0
    macro_all, macro_nonempty, exact = [], [], 0
    for e in ents:
        p, t = P.get(e, set()), T.get(e, set())
        i = len(p & t)
        tp, n_pred, n_true = tp + i, n_pred + len(p), n_true + len(t)
        # cluster pairwise: all pairs inside the cluster {S1} U matches
        ctp += (i + 1) * i // 2
        cpred += (len(p) + 1) * len(p) // 2
        ctrue += (len(t) + 1) * len(t) // 2
        if not t and not p:
            fe = 1.0
        elif not t or not p:
            fe = 0.0
        else:
            fe = f05(i / len(p), i / len(t))
        macro_all.append(fe)
        if t:
            macro_nonempty.append(fe)
        exact += p == t
    return {
        "pairs, micro (what we optimise)": f05(tp / max(n_pred, 1), tp / max(n_true, 1)),
        "cluster pairwise (incl. S2-S3 pairs)": f05(ctp / max(cpred, 1), ctp / max(ctrue, 1)),
        "per-entity average, all entities": float(np.mean(macro_all)),
        "per-entity average, entities with matches": float(np.mean(macro_nonempty)),
        "exact set match rate (accuracy)": exact / len(ents),
    }


PROBES = (0.50, 0.93)
SWEEP = (0.40, 0.50, 0.60, 0.70, 0.77, 0.85, 0.90, 0.93, 0.96)


def accepted(d, thr_map_or_value):
    thr = (d["country"].map(thr_map_or_value).fillna(thr_map_or_value["_default"]).to_numpy()
           if isinstance(thr_map_or_value, dict) else thr_map_or_value)
    a = d[d["prob"].to_numpy() >= thr]
    return a.rename(columns={"s1_id": "s1", "cand_id": "m"})[["s1", "m"]]


def write_probe(t, one2one, pred_dir):
    from common import DATA_DIR
    out = DATA_DIR.parent / "output_probe"
    out.mkdir(exist_ok=True)
    path = out / f"matching_results_t{int(round(t * 100))}.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for c in countries_of("test"):
            s1_ids = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"],
                                     filters=[("country_key", "=", c)])["entity_id"]
            pr = pq.read_table(pred_dir / f"test_{c}.parquet", filters=[("prob", ">=", t)]).to_pandas()
            if one2one:
                pr = pr.sort_values("prob", ascending=False).drop_duplicates("cand_id")
            m = pr.groupby("s1_id")["cand_id"].agg(lambda x: ",".join(sorted(x)))
            col = m.reindex(s1_ids).fillna("").to_numpy(object)
            fm.write("".join(f"{a}\t{b}\n" for a, b in zip(s1_ids.to_numpy(object), col)))
    return path


def main():
    global ENTITIES, TRAIN_COUNTRIES
    TRAIN_COUNTRIES = set(countries_of("train"))
    b2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
    # test-side sections use the model that actually produced the submission
    live = b2 if b2.get("use_stage2") else json.loads((MODEL_DIR / "best.json").read_text())
    live_dir = PRED2_DIR if b2.get("use_stage2") else PRED_DIR
    booster = lgb.Booster(model_file=str(MODEL_DIR / "stage2.txt"))
    vB = pd.read_parquet(S2_DIR / "valB_s2.parquet")
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    tB = truth[truth["part"] == "valB"].rename(columns={"m": "m"})
    ENTITIES = sorted(set(vB["s1_id"]) | set(tB["s1"]))
    probB = booster.predict(vB[b2["feature_cols"]].to_numpy(np.float32))
    thr_map = b2["thresholds"]

    d = keep_best_per_cand(vB, probB) if b2["one_to_one"] else \
        vB[["s1_id", "cand_id", "country", "label"]].assign(prob=probB)
    thr = d["country"].map(thr_map).fillna(thr_map["_default"]).to_numpy()
    acc = d[d["prob"].to_numpy() >= thr]

    b1 = json.loads((MODEL_DIR / "best.json").read_text())
    boost1 = lgb.Booster(model_file=str(MODEL_DIR / f"{b1['experiment']}.txt"))
    v1 = pd.read_parquet(FEAT_DIR / "valB.parquet")
    prob1 = boost1.predict(v1[b1["feature_cols"]].to_numpy(np.float32))
    d1 = keep_best_per_cand(v1, prob1) if b1["one_to_one"] else \
        v1[["s1_id", "cand_id", "country", "label"]].assign(prob=prob1)
    m1 = metric_variants(accepted(d1, b1["thresholds"]), tB[["s1", "m"]])
    m2 = metric_variants(accepted(d, thr_map), tB[["s1", "m"]])
    log("=== 1. valB under different F0.5 definitions: v1 vs v2 (leaderboard: both 0.952) ===")
    log(f"  {'definition':45} {'v1':>7} {'v2':>7} {'v2-v1':>7}")
    for name in m1:
        log(f"  {name:45} {m1[name]:7.4f} {m2[name]:7.4f} {m2[name] - m1[name]:+7.4f}")

    log("\n=== 1b. v2 on valB: predicted score per threshold, for each definition ===")
    names = list(m2)[:4]
    short = ["pairs", "cluster", "ent_all", "ent_match"]
    log("  thr   " + "  ".join(f"{n:>9}" for n in short))
    for t in SWEEP:
        mv = metric_variants(accepted(d, t), tB[["s1", "m"]])
        log(f"  {t:.2f}  " + "  ".join(f"{mv[n]:9.4f}" for n in names))

    log("\n=== 2. how confident are accepted matches? (share of accepted with prob < 0.90) ===")
    for c in sorted(acc["country"].unique()):
        a = acc[acc["country"] == c]
        log(f"  valB {c:8} accepted={len(a):>9,}  unsure(<0.90)={(a['prob'] < 0.9).mean():6.1%}  "
            f"mean prob={a['prob'].mean():.3f}")
    tl = live["thresholds"]
    for c in countries_of("test"):
        t = tl.get(c, tl["_default"])
        pr = pq.read_table(live_dir / f"test_{c}.parquet", filters=[("prob", ">=", t)]).to_pandas()
        if live["one_to_one"]:
            pr = pr.sort_values("prob", ascending=False).drop_duplicates("cand_id")
        log(f"  test {c:8} accepted={len(pr):>9,}  unsure(<0.90)={(pr['prob'] < 0.9).mean():6.1%}  "
            f"mean prob={pr['prob'].mean():.3f}")
        if c not in TRAIN_COUNTRIES:     # a country never seen in training
            unseen = (c, pr, t)

    log("\n=== 3. French predicted matches (unseen country) ===")
    c, pr, t = unseen
    rng = np.random.default_rng(SEED)
    border = pr[pr["prob"] < t + 0.08]
    sure = pr[pr["prob"] >= 0.98]
    pick = pd.concat([border.sample(min(15, len(border)), random_state=SEED).assign(kind="borderline"),
                      sure.sample(min(10, len(sure)), random_state=SEED).assign(kind="confident")])
    ids = list(pick["s1_id"]) + list(pick["cand_id"])
    raw = pd.concat([pd.read_parquet(PREP_DIR / f"test_source{k}.parquet",
                                     columns=["entity_id", "business_name", "business_address"],
                                     filters=[("entity_id", "in", ids)]) for k in (1, 2, 3)])
    raw = raw.drop_duplicates("entity_id").set_index("entity_id")
    for kind, g in pick.groupby("kind", sort=False):
        log(f"  --- {kind}")
        for s, m, p in zip(g["s1_id"], g["cand_id"], g["prob"]):
            a, b = raw.loc[s], raw.loc[m]
            log(f"  p={p:.2f}  {a['business_name']} | {a['business_address']}")
            log(f"          = {b['business_name']} | {b['business_address']}")
    log("\n=== 4. probe submissions (same model, different threshold) ===")
    for t in PROBES:
        log(f"  wrote {write_probe(t, live['one_to_one'], live_dir)}")
    (WORK / "diagnose.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"\nsaved to {WORK / 'diagnose.txt'}")


if __name__ == "__main__":
    main()
