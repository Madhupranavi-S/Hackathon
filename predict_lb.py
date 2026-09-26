"""Predict the leaderboard score of output_final/ BEFORE uploading it.

    python -u predict_lb.py

Leaderboard = per-entity F0.5 averaged over all test entities, so it splits by country,
weighted by each country's share of test Source 1 entities:
  India / US : MEASURED on valB with real labels (official metric, per country).
  France     : no labels. ESTIMATED from the model's own expected per-entity F0.5 on the
               French test predictions, corrected by how far that estimator is off on
               India/US valB (where we can check it against the truth).
Also checks for a test shift: the same estimator on India/US TEST predictions vs valB.
Writes work/predict_lb.txt.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA_DIR, read_tsv
from finalize import (FEAT_DIR, MODEL_DIR, adjust, available_models, one_to_one,
                      rule_expected_f, rule_threshold)
from preprocess import PREP_DIR
from run_pipeline import WORK, countries_of

MISS = 0.02
# known leaderboard result + the OLD model's official valB score (from diagnose.txt, v2)
LB_ANCHORS = {"probe (old v2, France at 0.95)": (0.953, 0.9637)}
# Measured on the leaderboard: output_final_baseline scored 0.956. With India/US measured
# on valB (weighted contribution 0.8188), France actually scored (0.956-0.8188)/0.150 = 0.915,
# while this script's raw model estimate for France was 0.9782. The difference is the
# France calibration offset, applied to any new model's raw France estimate.
FRANCE_ACTUAL, FRANCE_RAW_THEN = (0.956 - 0.8188) / 0.150, 0.9782
LINES = []


def log(m=""):
    print(m, flush=True)
    LINES.append(m)


def decide(d, choice):
    return rule_threshold(d, choice["param"]) if choice["rule"] == "threshold" \
        else rule_expected_f(d, choice["param"])


def per_entity_actual(pred, truth, ents):
    ents = pd.Index(ents)
    n_pred = pred.groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    n_true = truth.groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    tp = pred.merge(truth, on=["s1", "m"]).groupby("s1").size().reindex(ents, fill_value=0).to_numpy()
    f = np.where((n_pred == 0) & (n_true == 0), 1.0,
                 1.25 * tp / np.maximum(n_pred + 0.25 * n_true, 1e-9))
    return pd.Series(f, index=ents)


def per_entity_expected(d, pred, ents):
    """Model-based expected F0.5 per entity for the chosen predictions `pred`."""
    ents = pd.Index(ents)
    p = np.clip(d["prob"].to_numpy(np.float64), 1e-6, 1 - 1e-6)
    g = pd.DataFrame({"s1": d["s1_id"].to_numpy(), "m": d["cand_id"].to_numpy(), "p": p})
    total = g.groupby("s1")["p"].sum().reindex(ents, fill_value=0.0)
    log_empty = np.log1p(-g["p"]).groupby(g["s1"]).sum().reindex(ents, fill_value=0.0)
    chosen = g.merge(pred, on=["s1", "m"])
    k = chosen.groupby("s1").size().reindex(ents, fill_value=0)
    tp = chosen.groupby("s1")["p"].sum().reindex(ents, fill_value=0.0)
    exp_t = total * (1 + MISS)
    f = np.where(k == 0, np.exp(log_empty - MISS * total),
                 1.25 * tp / np.maximum(k + 0.25 * exp_t, 1e-9))
    return pd.Series(f, index=ents)


def main():
    choice = json.loads((WORK / "final_choice.json").read_text())
    m = available_models()[choice["model"]]
    seen = set(countries_of("train"))
    log(f"final setup: {choice['model']} + {choice['rule']} ({choice['param']})")

    # ---- India/US on valB: actual vs estimator
    f = pd.read_parquet(m["val"]["valB"])
    prob = m["booster"].predict(f[m["cols"]].to_numpy(np.float32))
    d = one_to_one(f[["s1_id", "cand_id", "country"]].assign(prob=prob))
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    truth = truth[truth["part"] == "valB"]
    pred = decide(d, choice)
    val, bias = {}, {}
    log("\n=== valB (real labels), official metric per country ===")
    for c in sorted(seen):
        ents = sorted(set(f.loc[f["country"] == c, "s1_id"]) | set(truth.loc[truth["country"] == c, "s1"]))
        dc = d[d["country"] == c]
        pc = pred[pred["s1"].isin(set(ents))]
        act = per_entity_actual(pc, truth[truth["country"] == c][["s1", "m"]], ents).mean()
        est = per_entity_expected(dc, pc, ents).mean()
        val[c], bias[c] = act, act - est
        log(f"  {c:8} actual={act:.4f}   estimator={est:.4f}   estimator error={act - est:+.4f}")
    mean_bias = float(np.mean(list(bias.values())))

    # ---- test: weights, estimator per country, shift check, France estimate
    s1 = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id", "country_key"])
    weights = s1["country_key"].value_counts(normalize=True).to_dict()
    final = read_tsv(DATA_DIR.parent / "output_final" / "matching_results.tsv")
    final_sets = dict(zip(final["source1_entity_id"], final["matched_entity_ids"]))
    log("\n=== test ===")
    est_test, parts = {}, {}
    for c in countries_of("test"):
        dt = pq.read_table(m["test_dir"] / f"test_{c}.parquet").to_pandas().assign(country=c)
        dt = one_to_one(adjust(dt, seen))
        ents = s1.loc[s1["country_key"] == c, "entity_id"].tolist()
        pt = decide(dt, choice)
        e = per_entity_expected(dt, pt, ents).mean()
        est_test[c] = float(np.clip(e + (bias[c] if c in seen else mean_bias), 0.0, 1.0))
        if c in seen:
            parts[c] = val[c]
            log(f"  {c:8} weight={weights[c]:.3f}  valB actual={val[c]:.4f}  "
                f"test estimate={est_test[c]:.4f}  (shift {est_test[c] - val[c]:+.4f})")
        else:
            calibrated = float(np.clip(e + (FRANCE_ACTUAL - FRANCE_RAW_THEN), 0.0, 1.0))
            parts[c] = calibrated
            log(f"  {c:8} weight={weights[c]:.3f}  UNSEEN: calibrated estimate={calibrated:.4f}  "
                f"(raw model estimate {e:.4f}; the leaderboard showed this estimator overshoots "
                f"France by {FRANCE_RAW_THEN - FRANCE_ACTUAL:.3f})")
            log(f"           (uncalibrated estimate using India/US error only: {est_test[c]:.4f})")
            probe = DATA_DIR.parent / "output_probe" / "matching_results_unseen95.tsv"
            if probe.exists():
                old = read_tsv(probe)
                old = old[old["source1_entity_id"].isin(set(ents))]
                same = np.mean([final_sets.get(a, "") == b for a, b in
                                zip(old["source1_entity_id"], old["matched_entity_ids"])])
                log(f"           French entities with the SAME prediction as the 0.953 probe: {same:.1%}")

    shift = float(np.mean([est_test[c] - val[c] for c in seen]))
    lb_val = sum(weights[c] * parts[c] for c in parts)
    lb_shift = sum(weights[c] * (parts[c] + (shift if c in seen else 0.0)) for c in parts)
    unseen_w = sum(w for c, w in weights.items() if c not in seen)
    log("\n=== predicted leaderboard ===")
    log(f"  India/US as measured on valB           : {lb_val:.4f}")
    log(f"  India/US adjusted for test shift ({shift:+.4f}): {lb_shift:.4f}")
    lo, hi = min(lb_val, lb_shift) - unseen_w * 0.015, max(lb_val, lb_shift) + unseen_w * 0.015
    log(f"  likely range (France +/- 0.015)          : {lo:.4f} - {hi:.4f}")
    log("  (check: the same method applied to output_final_baseline gives 0.956, its real score)")
    for name, (lb, old_val) in LB_ANCHORS.items():
        log(f"  anchor: {name} scored {lb} with India/US ~{old_val} on valB -> France then ~ "
            f"{(lb - (1 - unseen_w) * old_val) / unseen_w:.3f} (+/-0.004 from leaderboard rounding); "
            "the new France estimate above should be at least this")
    (WORK / "predict_lb.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()