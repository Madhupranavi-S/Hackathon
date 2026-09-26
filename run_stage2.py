"""STAGE 2 in one command: a second model that uses the FIRST model's confidence about
the other candidates of the same Source 1 entity.

    python -u run_stage2.py                  # everything (resumes where it stopped)
    python -u run_stage2.py --redo s2train   # redo a stage and everything after it

Why: the first model judges each pair alone. But if it is very sure about two other
records of an entity, and this record has the same address, that is strong evidence.
Stage 2 features (per pair S1 s, candidate c, first-model probability p1):
  p1, its rank among s's candidates, the best / sum / count of s's OTHER confident
  candidates, and "support": how similar c is (address, name) to s's most confident
  other candidates, weighted by their p1. Plus cheap blocking signals.

Honesty: stage 2 must learn from REALISTIC p1 values, so it trains on 'fresh' Source 1
entities that the first model never saw. valA picks settings, valB is the untouched score.
Stage 2 is only used if it beats the first model on valA; otherwise the run stops.

Outputs: work/stage2/*, work/pred2/test_<country>.parquet,
         output_v2/matching_results.tsv + candidate_pairs.tsv (v1 in output/ is untouched),
         work/stage2.log and work/report_stage2.md
"""
import argparse
import json
import logging
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import DATA_DIR, OUTPUT_DIR, SEED, assign_folds, read_tsv, up_to_date
from generate_candidates import CAND_DIR
from matcher_features import build_features, group_stats, idf_matrices, load_records, make_pool
from preprocess import PREP_DIR
from run_pipeline import (FEAT_DIR, MODEL_DIR, PRED_DIR, WORK, best_threshold, countries_of,
                          evaluate, keep_best_per_cand, load_gt)

S2_DIR, PRED2_DIR = WORK / "stage2", WORK / "pred2"
CE_DIR, CE_MODEL = WORK / "ce", MODEL_DIR / "cross_encoder"   # neural cross-encoder (optional)
OUT2 = DATA_DIR.parent / "output_v2"
STAGES = ["fresh", "s2features", "s2train", "s2predict", "write", "check"]
CHUNK = 2_000_000
CONF = 0.5          # a candidate counts as "confident" above this first-model probability
TOP_SUPPORT = 3     # compare each candidate with up to this many confident siblings
CHEAP = ["score", "fwd_rank", "rev_rank", "rev_top1", "gap_s1", "gap_cand", "s1_n", "cand_n", "is_s3"]
KEEP = ["s1_id", "cand_id", "country", "label", "p1"] + CHEAP
S2_PARAMS = dict(learning_rate=0.05, num_leaves=127, min_child_samples=200, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0)

def prune_tau():
    """Candidate filter cutoff from choose_prune.py (0 = no filter)."""
    p = WORK / "prune.json"
    return float(json.loads(p.read_text())["tau"]) if p.exists() else 0.0


def ce_files():
    return [CE_DIR / f"{p}.parquet" for p in ("fresh", "valA", "valB")] + \
           [CE_DIR / f"test_{c}.parquet" for c in countries_of("test")]


def ce_available():
    """Use cross-encoder scores only if they exist for EVERY part (train, val and test), so
    stage 2 always sees the same features in training and on the test set."""
    return all(f.exists() for f in ce_files())


LOG = logging.getLogger("er2")
REPORT = []


def setup_logging():
    WORK.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(WORK / "stage2.log", encoding="utf-8")):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    LOG.setLevel(logging.INFO)


def log(msg="", report=False):
    LOG.info(msg)
    if report:
        REPORT.append(msg)


def stage1():
    best = json.loads((MODEL_DIR / "best.json").read_text())
    booster = lgb.Booster(model_file=str(MODEL_DIR / f"{best['experiment']}.txt"))
    return best, booster


# ================================================================ stage-2 features
def s2_features(d, poolrec):
    """d: one country, COMPLETE S1 groups, columns s1_id, cand_id, p1 (+ CHEAP).
    Returns stage-2 features aligned with d's rows."""
    n = len(d)
    code = pd.factorize(d["s1_id"])[0]
    ng = int(code.max()) + 1 if n else 0
    p = d["p1"].to_numpy(np.float32)
    order = np.lexsort((-p, code))                      # by S1, then p1 descending
    code_s, p_s = code[order], p[order]
    new = np.r_[True, code_s[1:] != code_s[:-1]] if n else np.array([], bool)
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0)) if n else new.astype(int)
    r_s = np.arange(n) - start
    rank = np.empty(n, dtype=np.int64)
    rank[order] = r_s

    top1 = np.full(ng, np.nan, np.float32)
    top2 = np.full(ng, np.nan, np.float32)
    top1[code_s[r_s == 0]] = p_s[r_s == 0]
    top2[code_s[r_s == 1]] = p_s[r_s == 1]
    other_max = np.where(rank == 0, top2[code], top1[code])
    conf = (p >= CONF).astype(np.float32)
    other_sum = np.bincount(code, weights=p, minlength=ng)[code] - p
    other_conf = np.bincount(code, weights=conf, minlength=ng)[code] - conf

    f = pd.DataFrame(index=d.index)
    for c in CHEAP:
        f[c] = d[c].to_numpy(np.float32)
    f["p1"] = p
    f["p1_rank"] = rank.astype(np.float32)
    f["other_max"] = other_max
    f["p1_minus_other"] = p - np.nan_to_num(other_max, nan=0.0)
    f["other_sum"] = other_sum.astype(np.float32)
    f["other_conf"] = other_conf.astype(np.float32)

    if "ce" in d.columns:     # neural cross-encoder score and its context within the entity
        ce = d["ce"].to_numpy(np.float32)
        g = pd.Series(ce).groupby(code)
        f["ce"] = ce
        f["ce_gap"] = g.transform("max").to_numpy(np.float32) - ce
        f["ce_rank"] = g.rank(ascending=False, method="min").to_numpy(np.float32)

    # support: similarity to the most confident OTHER candidates of the same S1
    mask = (r_s <= TOP_SUPPORT) & (p_s >= CONF)
    confs = pd.DataFrame({"code": code_s[mask], "rc": r_s[mask], "row_c": order[mask],
                          "p_c": p_s[mask]})
    x = pd.DataFrame({"code": code, "i": np.arange(n)}).merge(confs, on="code")
    x = x[x["row_c"] != x["i"]].sort_values(["i", "rc"]).groupby("i").head(TOP_SUPPORT)
    for name in ("sup_addr", "sup_name", "sup_w", "sup_both"):
        f[name] = np.nan
    if len(x):
        jb = poolrec.index.get_indexer(d["cand_id"])
        addr = poolrec["addr_n"].to_numpy(object)
        core = poolrec["name_core"].to_numpy(object)
        ja, jc = jb[x["i"].to_numpy()], jb[x["row_c"].to_numpy()]
        sa = cpdist(addr[ja], addr[jc], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        sn = cpdist(core[ja], core[jc], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        agg = pd.DataFrame({"i": x["i"].to_numpy(), "sup_addr": sa, "sup_name": sn,
                            "sup_w": x["p_c"].to_numpy() * sa / 100,
                            "sup_both": np.minimum(sa, sn)}).groupby("i").max()
        for name in agg.columns:
            vals = np.full(n, np.nan, np.float32)
            vals[agg.index.to_numpy()] = agg[name].to_numpy()
            f[name] = vals
    return f.astype(np.float32)


def s2_features_grouped(d, poolrec):
    """Run s2_features in chunks that never split an S1 group."""
    d = d.sort_values("s1_id", kind="stable")
    code = pd.factorize(d["s1_id"])[0]
    cuts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    out, s = [], 0
    while s < len(d):
        k = np.searchsorted(cuts, s + CHUNK)
        e = int(cuts[k]) if k < len(cuts) else len(d)
        out.append(s2_features(d.iloc[s:e], poolrec))
        s = e
    return pd.concat(out).loc[d.index], d


# ================================================================ stage: fresh
def stage_fresh(args, pool):
    """Stage-1 features + p1 for fresh entities; p1 for valA/valB."""
    best, booster = stage1()
    cols = best["feature_cols"]
    used = set(pd.read_parquet(FEAT_DIR / "train.parquet", columns=["s1_id"])["s1_id"])
    ids = pd.read_parquet(PREP_DIR / "train_source1.parquet", columns=["entity_id"])["entity_id"]
    folds = assign_folds(ids)
    pool_ids = np.array(sorted(k for k, v in folds.items() if v != "val" and k not in used),
                        dtype=object)
    fresh = set(np.random.default_rng(SEED + 1).choice(
        pool_ids, min(args.fresh_s1, len(pool_ids)), replace=False))
    if not fresh:
        raise RuntimeError("no fresh Source 1 entities left: lower --train-s1 in run_pipeline.py")
    log(f"fresh entities (never seen by the first model): {len(fresh):,}", report=True)

    gt = load_gt()
    gt = gt[gt["s1"].isin(fresh)]
    parts = []
    for country in countries_of("train"):
        t0 = time.time()
        s1rec, poolrec = load_records("train", country)
        s1stats, candstats = group_stats("train", country)
        idf = idf_matrices(s1rec, poolrec, args.threads, pool)
        here = [x for x in s1rec.index if x in fresh]
        if not here:
            continue
        cand = pq.read_table(CAND_DIR / f"train_{country}.parquet",
                             filters=[("s1_id", "in", here)]).to_pandas()
        lab = gt.assign(label=1).rename(columns={"s1": "s1_id", "m": "cand_id"})
        cand = cand.merge(lab, on=["s1_id", "cand_id"], how="left")
        cand["label"] = cand["label"].fillna(0).astype(np.int8)
        F = pd.concat([build_features(cand.iloc[s:s + CHUNK], s1rec, poolrec, s1stats,
                                      candstats, pool, idf) for s in range(0, len(cand), CHUNK)])
        F["p1"] = booster.predict(F[cols].to_numpy(np.float32), num_threads=args.threads)
        F["s1_id"] = cand["s1_id"].to_numpy(object)
        F["cand_id"] = cand["cand_id"].to_numpy(object)
        F["country"] = country
        F["label"] = cand["label"].to_numpy()
        parts.append(F[KEEP])
        log(f"  {country}: {len(here):,} fresh S1, {len(cand):,} pairs in {time.time() - t0:.0f}s")
    S2_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(S2_DIR / "fresh_p1.parquet", index=False)
    for v in ("valA", "valB"):
        F = pd.read_parquet(FEAT_DIR / f"{v}.parquet")
        F["p1"] = booster.predict(F[cols].to_numpy(np.float32), num_threads=args.threads)
        F[KEEP].to_parquet(S2_DIR / f"{v}_p1.parquet", index=False)
    log("first-model probabilities saved")


# ================================================================ stage: s2features
def stage_s2features(args):
    tau = prune_tau()
    use_ce = ce_available()
    log(f"neural cross-encoder features: {'yes' if use_ce else 'no'}", report=True)
    log(f"candidate filter: first-model probability >= {tau}", report=True)
    for part in ("fresh", "valA", "valB"):
        d_all = pd.read_parquet(S2_DIR / f"{part}_p1.parquet")
        n0 = len(d_all)
        d_all = d_all[d_all["p1"] >= tau]
        log(f"  {part}: {len(d_all):,} of {n0:,} pairs kept after the filter", report=True)
        if use_ce:
            d_all = d_all.merge(pd.read_parquet(CE_DIR / f"{part}.parquet"),
                                on=["s1_id", "cand_id"], how="left")
        out = []
        for country in sorted(d_all["country"].unique()):
            t0 = time.time()
            _, poolrec = load_records("train", country)
            F, d = s2_features_grouped(d_all[d_all["country"] == country], poolrec)
            F[["s1_id", "cand_id", "country", "label"]] = d[["s1_id", "cand_id", "country", "label"]]
            out.append(F)
            log(f"  {part}/{country}: {len(d):,} pairs in {time.time() - t0:.0f}s")
        pd.concat(out, ignore_index=True).to_parquet(S2_DIR / f"{part}_s2.parquet", index=False)


# ================================================================ stage: s2train
def stage_s2train(args):
    best1, _ = stage1()
    tr = pd.read_parquet(S2_DIR / "fresh_s2.parquet")
    vA = pd.read_parquet(S2_DIR / "valA_s2.parquet")
    vB = pd.read_parquet(S2_DIR / "valB_s2.parquet")
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    tA, tB = truth[truth["part"] == "valA"], truth[truth["part"] == "valB"]
    meta = ["s1_id", "cand_id", "country", "label"]
    cols = [c for c in tr.columns if c not in meta]
    log(f"stage-2 train rows={len(tr):,}  features={len(cols)}", report=True)

    t0 = time.time()
    model = lgb.LGBMClassifier(n_estimators=3000, random_state=SEED, n_jobs=args.threads,
                               verbose=-1, **S2_PARAMS)
    model.fit(tr[cols], tr["label"], eval_set=[(vA[cols], vA["label"])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "stage2.txt"))
    pA, pB = model.predict_proba(vA[cols])[:, 1], model.predict_proba(vB[cols])[:, 1]
    log(f"stage-2 model: {model.best_iteration_} trees in {time.time() - t0:.0f}s", report=True)

    results = []
    for one2one in (True, False):
        dA = keep_best_per_cand(vA, pA) if one2one else vA[meta].assign(prob=pA)
        dB = keep_best_per_cand(vB, pB) if one2one else vB[meta].assign(prob=pB)
        g = best_threshold(dA, len(tA))
        per_c = {c: best_threshold(dA[dA["country"] == c], int((tA["country"] == c).sum()))
                 for c in sorted(dA["country"].unique())}
        for mode, thr in (("global", {"_default": g}), ("per_country", {**per_c, "_default": g})):
            rA, rB = evaluate(dA, tA, thr), evaluate(dB, tB, thr)
            results.append(dict(one_to_one=one2one, mode=mode, thresholds=thr, valA=rA["all"][0],
                                valB=rB["all"], valB_by_country={k: v for k, v in rB.items() if k != "all"}))
            log(f"   one_to_one={one2one!s:5} thresholds={mode:11}  valA F0.5={rA['all'][0]:.4f}"
                f"   valB F0.5={rB['all'][0]:.4f} (P={rB['all'][1]:.4f} R={rB['all'][2]:.4f})",
                report=True)
    imp = sorted(zip(model.booster_.feature_importance("gain"), cols), reverse=True)[:10]
    log("   top features (gain): " + ", ".join(n for _, n in imp), report=True)

    best2 = max(results, key=lambda r: r["valA"])
    best2["feature_cols"] = cols
    best2["prune"] = prune_tau()
    use = best2["valA"] > best1["valA"] + 0.0005 or best2["prune"] > 0   # the cascade needs stage 2
    best2["use_stage2"] = bool(use)
    (MODEL_DIR / "best_stage2.json").write_text(json.dumps(best2, indent=2))
    log("", report=True)
    log(f"FIRST MODEL : valA F0.5={best1['valA']:.4f}   valB F0.5={best1['valB'][0]:.4f}", report=True)
    log(f"STAGE 2     : valA F0.5={best2['valA']:.4f}   valB F0.5={best2['valB'][0]:.4f}  "
        f"(P={best2['valB'][1]:.4f} R={best2['valB'][2]:.4f})", report=True)
    for c, (fc, pc_, rc) in best2["valB_by_country"].items():
        log(f"   {c:8} F0.5={fc:.4f}  precision={pc_:.4f}  recall={rc:.4f}", report=True)
    log("DECISION: " + ("use stage 2" if use else "stage 2 does not help; keep output/ (v1)"),
        report=True)


# ================================================================ stage: s2predict
def stage_s2predict(args):
    tau = prune_tau()
    best2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
    use_ce = "ce" in best2["feature_cols"]
    if use_ce and not ce_available():
        raise RuntimeError("stage 2 was trained with cross-encoder features but test scores are missing")
    best2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
    booster = lgb.Booster(model_file=str(MODEL_DIR / "stage2.txt"))
    PRED2_DIR.mkdir(parents=True, exist_ok=True)
    for country in countries_of("test"):
        dst = PRED2_DIR / f"test_{country}.parquet"
        if "s2predict" not in args.redo and up_to_date(
                [dst], [MODEL_DIR / "stage2.txt", MODEL_DIR / "best_stage2.json",
                        PRED_DIR / f"test_{country}.parquet", WORK / "prune.json",
                        CE_DIR / f"test_{country}.parquet"]):
            log(f"  skip {dst.name} (up to date)")
            continue
        t0 = time.time()
        _, poolrec = load_records("test", country)
        s1stats, candstats = group_stats("test", country)
        cand = pq.read_table(CAND_DIR / f"test_{country}.parquet")
        pred = pq.read_table(PRED_DIR / f"test_{country}.parquet")
        ce_test = pd.read_parquet(CE_DIR / f"test_{country}.parquet") if use_ce else None
        if cand.num_rows != pred.num_rows or not pc.all(pc.equal(
                cand.column("cand_id"), pred.column("cand_id"))).as_py():
            raise RuntimeError(f"{country}: candidate and prediction files are not aligned")
        ids = cand.column("s1_id").combine_chunks()
        changed = pc.not_equal(ids.slice(1), ids.slice(0, len(ids) - 1)).to_numpy(zero_copy_only=False)
        cuts = np.flatnonzero(np.r_[True, changed])        # start of each S1 group
        out, s = [], 0
        while s < cand.num_rows:
            k = np.searchsorted(cuts, s + CHUNK)
            e = int(cuts[k]) if k < len(cuts) else cand.num_rows
            d = cand.slice(s, e - s).to_pandas()
            d["p1"] = pred.column("prob").slice(s, e - s).to_numpy()
            d = d[d["p1"] >= tau].reset_index(drop=True)
            if not len(d):
                s = e
                continue
            if use_ce:
                d = d.merge(ce_test, on=["s1_id", "cand_id"], how="left")
            score = d["score"].to_numpy(np.float32)
            d["rev_top1"] = (d["rev_rank"] == 0).astype(np.float32)
            d["gap_s1"] = s1stats["s1_max"].reindex(d["s1_id"]).to_numpy(np.float32) - score
            d["gap_cand"] = candstats["cand_max"].reindex(d["cand_id"]).to_numpy(np.float32) - score
            d["s1_n"] = s1stats["s1_n"].reindex(d["s1_id"]).to_numpy(np.float32)
            d["cand_n"] = candstats["cand_n"].reindex(d["cand_id"]).to_numpy(np.float32)
            d["is_s3"] = d["cand_id"].str.startswith("S3-").to_numpy(np.float32)
            F = s2_features(d, poolrec)[best2["feature_cols"]]
            prob = booster.predict(F.to_numpy(np.float32), num_threads=args.threads)
            out.append(pd.DataFrame({"s1_id": d["s1_id"].to_numpy(object),
                                     "cand_id": d["cand_id"].to_numpy(object),
                                     "prob": prob.astype(np.float32)}))
            log(f"    {country}: {e:,} / {cand.num_rows:,} pairs ({time.time() - t0:.0f}s)")
            s = e
        pd.concat(out, ignore_index=True).to_parquet(dst, index=False)
        log(f"  {country}: stage-2 predictions saved in {time.time() - t0:.0f}s")


# ================================================================ stage: write & check
def stage_write(args):
    best2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
    thresholds = best2["thresholds"]
    OUT2.mkdir(parents=True, exist_ok=True)
    m_path = OUT2 / "matching_results.tsv"
    m_tmp = m_path.with_suffix(".tmp")
    log("", report=True)
    log("TEST PREDICTIONS (stage 2)", report=True)
    with open(m_tmp, "w", encoding="utf-8", newline="\n") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for country in countries_of("test"):
            thr = thresholds.get(country, thresholds["_default"])
            s1_ids = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"],
                                     filters=[("country_key", "=", country)])["entity_id"]
            pr = pq.read_table(PRED2_DIR / f"test_{country}.parquet",
                               filters=[("prob", ">=", thr)]).to_pandas()
            if best2["one_to_one"]:
                pr = pr.sort_values("prob", ascending=False).drop_duplicates("cand_id")
            matches = pr.groupby("s1_id")["cand_id"].agg(lambda x: ",".join(sorted(x)))
            m_col = matches.reindex(s1_ids).fillna("").to_numpy(object)
            fm.write("".join(f"{a}\t{b}\n" for a, b in zip(s1_ids.to_numpy(object), m_col)))
            n = len(s1_ids)
            log(f"  {country:8} S1={n:,}  threshold={thr:.2f}  with matches="
                f"{(m_col != '').mean():.1%}  matched pairs={len(pr):,} ({len(pr) / n:.2f} per S1)",
                report=True)
    m_tmp.replace(m_path)
    write_candidates(OUT2 / "candidate_pairs.tsv")
    log(f"wrote {OUT2}", report=True)


def write_candidates(path):
    """candidate_pairs.tsv = exactly the pairs the final model scores: the blocking output,
    or, with the cascade active, the blocking output filtered by first-model probability."""
    tau = prune_tau()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in countries_of("test"):
            s1_ids = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"],
                                     filters=[("country_key", "=", country)])["entity_id"]
            if tau > 0:
                ct = pq.read_table(PRED_DIR / f"test_{country}.parquet", columns=["s1_id", "cand_id"],
                                   filters=[("prob", ">=", tau)]).to_pandas()
            else:
                ct = pq.read_table(CAND_DIR / f"test_{country}.parquet",
                                   columns=["s1_id", "cand_id"]).to_pandas()
            cands = ct.groupby("s1_id", sort=False)["cand_id"].agg(",".join)
            n_pairs = len(ct)
            del ct
            c_col = cands.reindex(s1_ids).fillna("").to_numpy(object)
            fc.write("".join(f"{a}\t{b}\n" for a, b in zip(s1_ids.to_numpy(object), c_col)))
            log(f"  candidate_pairs: {country}: {n_pairs / len(s1_ids):.2f} candidates per entity",
                report=True)
    tmp.replace(path)


def stage_check(args):
    problems = []
    s1 = set(pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"])
    res = read_tsv(OUT2 / "matching_results.tsv")
    cand = read_tsv(OUT2 / "candidate_pairs.tsv")
    for name, df in (("matching_results", res), ("candidate_pairs", cand)):
        if df["source1_entity_id"].duplicated().any():
            problems.append(f"{name}: duplicate Source 1 rows")
        if len(df) != len(s1) or set(df["source1_entity_id"]) != s1:
            problems.append(f"{name}: rows do not match the test Source 1 ids exactly")
    cand_map = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
    bad = dup = 0
    for sid, lst in zip(res["source1_entity_id"], res["matched_entity_ids"]):
        if lst:
            ids = lst.split(",")
            dup += len(ids) != len(set(ids))
            allowed = set(cand_map.get(sid, "").split(","))
            bad += any(x not in allowed for x in ids)
    if dup:
        problems.append(f"{dup} rows with duplicate ids")
    if bad:
        problems.append(f"{bad} rows with matches not in candidate_pairs.tsv")
    log("", report=True)
    log("SUBMISSION CHECK (output_v2): " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)),
        report=True)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-s1", type=int, default=300_000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--redo", nargs="*", default=[], choices=STAGES)
    args = ap.parse_args()
    if args.redo:
        args.redo = STAGES[min(STAGES.index(s) for s in args.redo):]
    if not (MODEL_DIR / "best.json").exists():
        sys.exit("run run_pipeline.py first (stage 1 model not found)")
    feats = [FEAT_DIR / f"{p}.parquet" for p in ("train", "valA", "valB")]
    if not up_to_date([MODEL_DIR / "best.json"], feats):
        sys.exit("the first model is older than its features: run  python -u run_pipeline.py  first")

    setup_logging()
    REPORT.append(f"# Stage 2 report ({time.strftime('%Y-%m-%d %H:%M')})\n")
    p1_files = [S2_DIR / f"{p}_p1.parquet" for p in ("fresh", "valA", "valB")]
    s2_files = [S2_DIR / f"{p}_s2.parquet" for p in ("fresh", "valA", "valB")]
    done = {   # evaluated when each stage is reached
        "fresh": lambda: up_to_date(p1_files, [MODEL_DIR / "best.json", FEAT_DIR / "valA.parquet",
                                               FEAT_DIR / "valB.parquet"]),
        "s2features": lambda: up_to_date(s2_files, p1_files + [WORK / "prune.json"] + ce_files()),
        "s2train": lambda: up_to_date([MODEL_DIR / "best_stage2.json", MODEL_DIR / "stage2.txt"], s2_files),
        "s2predict": lambda: False,
        "write": lambda: up_to_date([OUT2 / "matching_results.tsv", OUT2 / "candidate_pairs.tsv"],
                                    sorted(PRED2_DIR.glob("test_*.parquet")) + [WORK / "prune.json"]),
        "check": lambda: False,
    }
    t_all = time.time()
    with make_pool(args.threads) as pool:
        for stage in STAGES:
            if stage == "s2predict" or stage == "write":
                best2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
                if not best2["use_stage2"]:
                    log("stage 2 did not beat the first model: stopping. Keep using output/ (v1).",
                        report=True)
                    break
            if done[stage]() and stage not in args.redo:
                log(f"== {stage}: already done, skipping")
                if stage == "s2train":
                    b = json.loads((MODEL_DIR / "best_stage2.json").read_text())
                    log(f"STAGE 2: valB F0.5={b['valB'][0]:.4f}  use_stage2={b['use_stage2']}",
                        report=True)
                continue
            t0 = time.time()
            log(f"== {stage}: starting")
            {"fresh": lambda: stage_fresh(args, pool), "s2features": lambda: stage_s2features(args),
             "s2train": lambda: stage_s2train(args), "s2predict": lambda: stage_s2predict(args),
             "write": lambda: stage_write(args), "check": lambda: stage_check(args)}[stage]()
            log(f"== {stage}: finished in {(time.time() - t0) / 60:.1f} min")
            (WORK / "report_stage2.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min. Summary: {WORK / 'report_stage2.md'}")
    (WORK / "report_stage2.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":   # required on Windows for multiprocessing
    main()
