"""WHERE is the score lost? Oracle error analysis on valB with the official metric.

    python -u error_analysis.py

For the model and decision rule currently in use (work/final_choice.json if it exists,
otherwise stage 2 or the first model with a tuned threshold), it measures how much the
official per-entity F0.5 would rise if ONE kind of error were fixed perfectly:
  false merges removed               -> precision problem
  rejected true matches accepted     -> matcher problem (the pair was scored, but rejected)
  filtered-out true matches restored -> candidate-filter problem
  blocking misses added              -> blocking problem (never became a candidate)
and prints real examples of each, plus which entities lose the most score.
Writes work/error_analysis.txt. Read-only otherwise.
"""
import json

import numpy as np
import pandas as pd

from common import SEED
from finalize import (available_models, macro_f05, one_to_one, rule_expected_f,
                      rule_threshold, THRESH_GRID)
from preprocess import PREP_DIR
from run_pipeline import FEAT_DIR, WORK
from run_stage2 import S2_DIR, prune_tau

LINES = []


def log(m=""):
    print(m, flush=True)
    LINES.append(m)


def per_entity(pred, truth, ents):
    ents = pd.Index(ents)
    n_pred = pred.groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    n_true = truth.groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    tp = pred.merge(truth, on=["s1", "m"]).groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    f = np.where((n_pred == 0) & (n_true == 0), 1.0,
                 1.25 * tp / np.maximum(n_pred + 0.25 * n_true, 1e-9))
    return pd.DataFrame({"f": f, "n_pred": n_pred, "n_true": n_true, "tp": tp}, index=ents)


def raw(ids):
    ids = list(set(ids))
    parts = [pd.read_parquet(PREP_DIR / f"train_source{k}.parquet",
                             columns=["entity_id", "business_name", "business_address"],
                             filters=[("entity_id", "in", ids)]) for k in (1, 2, 3)]
    return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id")


def show(title, pairs, n=12, prob=None):
    log(f"\n--- {title} (random {min(n, len(pairs))} of {len(pairs):,})")
    if not len(pairs):
        return
    smp = pairs.sample(min(n, len(pairs)), random_state=SEED)
    r = raw(list(smp["s1"]) + list(smp["m"]))
    for _, row in smp.iterrows():
        a, b = r.loc[row["s1"]], r.loc[row["m"]]
        p = f"p={row['prob']:.2f}  " if "prob" in row and pd.notna(row.get("prob")) else ""
        log(f"  {p}{a['business_name']} | {a['business_address']}")
        log(f"      {'=' if row.get('true', True) else 'x'} {b['business_name']} | {b['business_address']}")


def main():
    models = available_models()
    choice_file = WORK / "final_choice.json"
    if choice_file.exists():
        choice = json.loads(choice_file.read_text())
    else:
        name = "stage 2" if "stage 2" in models else "first model"
        choice = {"model": name, "rule": "threshold", "param": None}
    m = models[choice["model"]]
    tau = prune_tau()
    log(f"model: {choice['model']}  rule: {choice['rule']}  candidate filter tau={tau}")

    truth_all = pd.read_parquet(FEAT_DIR / "truth.parquet")
    out = {}
    for v in ("valA", "valB"):
        f = pd.read_parquet(m["val"][v])
        prob = m["booster"].predict(f[m["cols"]].to_numpy(np.float32))
        d = one_to_one(f[["s1_id", "cand_id", "country"]].assign(prob=prob))
        t = truth_all[truth_all["part"] == v][["s1", "m", "country"]]
        full = pd.read_parquet(FEAT_DIR / f"{v}.parquet", columns=["s1_id", "cand_id"])
        ents = sorted(set(full["s1_id"]) | set(t["s1"]))
        out[v] = (d, t, full, ents)
    if choice["param"] is None:   # tune a threshold on valA for the official metric
        dA, tA, _, eA = out["valA"]
        choice["param"] = float(max(THRESH_GRID, key=lambda x: macro_f05(rule_threshold(dA, x),
                                                                           tA[["s1", "m"]], eA)))
    d, t, full, ents = out["valB"]
    decide = (lambda dd: rule_threshold(dd, choice["param"])) if choice["rule"] == "threshold" \
        else (lambda dd: rule_expected_f(dd, choice["param"]))
    pred = decide(d)
    tt = t[["s1", "m"]]
    base = macro_f05(pred, tt, ents)

    # classify every true pair and every predicted pair
    scored = set(zip(d["s1_id"], d["cand_id"]))
    in_full = set(zip(full["s1_id"], full["cand_id"]))
    predicted = set(zip(pred["s1"], pred["m"]))
    tset = set(zip(tt["s1"], tt["m"]))
    fp = pred[[p not in tset for p in zip(pred["s1"], pred["m"])]]
    missed = tt[[p not in predicted for p in zip(tt["s1"], tt["m"])]].copy()
    missed["where"] = ["rejected" if p in scored else ("filtered" if p in in_full else "blocking")
                       for p in zip(missed["s1"], missed["m"])]

    def score_with(p):
        return macro_f05(p, tt, ents)

    no_fp = pred[[p in tset for p in zip(pred["s1"], pred["m"])]]
    gains = {
        "remove all false merges (precision)": score_with(no_fp) - base,
        "accept all rejected true matches (matcher)": score_with(
            pd.concat([pred, missed.loc[missed["where"] == "rejected", ["s1", "m"]]])) - base,
        "restore true matches cut by the filter": score_with(
            pd.concat([pred, missed.loc[missed["where"] == "filtered", ["s1", "m"]]])) - base,
        "add true matches blocking never found": score_with(
            pd.concat([pred, missed.loc[missed["where"] == "blocking", ["s1", "m"]]])) - base,
    }
    log(f"\n=== valB official score now: {base:.4f}  (perfect = 1.0, gap {1 - base:.4f}) ===")
    log("gain if ONE kind of error were fixed perfectly:")
    for k, g in sorted(gains.items(), key=lambda kv: -kv[1]):
        log(f"  {k:45} +{g:.4f}")
    log(f"\ncounts: {len(pred):,} predicted pairs, {len(fp):,} false merges; {len(tt):,} true pairs, "
        + ", ".join(f"{(missed['where'] == w).sum():,} missed by {w}" for w in ("rejected", "filtered", "blocking")))

    # which entities lose the score
    pe = per_entity(pred, tt, ents)
    pe["kind"] = np.select(
        [(pe["n_true"] == 0) & (pe["n_pred"] > 0), (pe["n_true"] > 0) & (pe["n_pred"] == 0),
         (pe["n_true"] > 0) & (pe["tp"] < pe["n_pred"]), (pe["n_true"] > 0) & (pe["tp"] < pe["n_true"])],
        ["singleton wrongly matched", "matches exist, predicted none",
         "has a false merge", "some matches missed"], "perfect")
    loss = (1 - pe["f"]).groupby(pe["kind"]).sum() / len(pe)
    cnt = pe["kind"].value_counts()
    log("\nscore lost per kind of entity (sums to the gap):")
    for k in loss.sort_values(ascending=False).index:
        if k != "perfect":
            log(f"  {k:32} -{loss[k]:.4f}   ({cnt[k]:,} entities, {cnt[k] / len(pe):.1%})")
    log("\nmissed matches: how many per affected entity")
    per = missed.groupby("s1").size()
    log("  " + "  ".join(f"{k} missed: {v:,}" for k, v in per.clip(upper=4).value_counts().sort_index().items())
        + "   (4 = 4 or more)")

    # probabilities of the rejected true matches: close calls or confident misses?
    rej = missed[missed["where"] == "rejected"].merge(
        d.rename(columns={"s1_id": "s1", "cand_id": "m"})[["s1", "m", "prob"]], on=["s1", "m"])
    if len(rej):
        bins = pd.cut(rej["prob"], [0, 0.05, 0.2, 0.5, 0.8, 1.0], include_lowest=True)
        log("\nrejected true matches by model probability:")
        for b, n in bins.value_counts().sort_index().items():
            log(f"  p in {b}: {n:,}")

    # ---- are validation false merges an artifact of the SAMPLE? (competing entity absent)
    from run_pipeline import load_gt
    gt_all = load_gt()                                  # ALL training ground truth
    owner = dict(zip(gt_all["m"], gt_all["s1"]))        # pool record -> its true S1 (at most one)
    in_sample = set(ents)
    kinds = []
    for s1_id, m_id in zip(fp["s1"], fp["m"]):
        o = owner.get(m_id)
        kinds.append("belongs to NO entity (distractor)" if o is None else
                     "belongs to another entity IN the sample" if o in in_sample else
                     "belongs to another entity OUTSIDE the sample")
    log("\nfalse merges: who does the pool record really belong to?")
    for k, n in pd.Series(kinds).value_counts().items():
        log(f"  {k:45} {n:6,}  ({n / max(len(fp), 1):.1%})")
    log("  (OUTSIDE the sample: on the real test set that entity competes for the record,")
    log("   so the one-to-one rule can often prevent the merge; validation overstates these)")

    # ---- empty or very short addresses among the errors
    addr = pd.concat([pd.read_parquet(PREP_DIR / f"train_source{k}.parquet", columns=["entity_id", "addr_n"],
                                      filters=[("entity_id", "in", list(set(fp["m"]) | set(missed["m"])))])
                      for k in (2, 3)]).drop_duplicates("entity_id").set_index("entity_id")["addr_n"]
    def empty_share(ids):
        a = addr.reindex(ids).fillna("")
        return (a.str.len() == 0).mean() if len(a) else 0.0
    log("\nshare of errors where the Source 2/3 record has an EMPTY address:")
    log(f"  false merges            {empty_share(fp['m']):.1%}")
    for w in ("rejected", "filtered", "blocking"):
        log(f"  missed by {w:13} {empty_share(missed.loc[missed['where'] == w, 'm']):.1%}")

    show("REJECTED true matches (matcher said no)", rej.assign(true=True))
    show("BLOCKING misses (never became a candidate)", missed[missed["where"] == "blocking"].assign(true=True))
    show("FALSE MERGES (predicted, but wrong)",
         fp.merge(d.rename(columns={"s1_id": "s1", "cand_id": "m"})[["s1", "m", "prob"]],
                  on=["s1", "m"]).assign(true=False))
    (WORK / "error_analysis.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nsaved to {WORK / 'error_analysis.txt'}")


if __name__ == "__main__":
    main()