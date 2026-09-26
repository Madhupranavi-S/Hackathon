"""STEPS 6 + 7 in one command: features -> train & compare models -> predict test ->
write submission files -> check them -> report.

    python -u run_pipeline.py                      # everything (resumes where it stopped)
    python -u run_pipeline.py --redo train write   # redo selected stages (and what follows)

Stages and what they leave behind (a stage is skipped if its output exists):
  features  work/features/{train,valA,valB}.parquet   features + labels for a sample
  train     work/models/*.txt, work/models/best.json  every experiment, best one chosen
  predict   work/pred/test_<country>.parquet          a probability for every test pair
  write     output/matching_results.tsv, output/candidate_pairs.tsv
  check     verifies every submission rule
A full log goes to work/pipeline.log and a summary to work/report.md.

Validation is honest: training uses non-validation Source 1 entities; the validation
entities are split into valA (early stopping + threshold tuning + choosing the best
setup) and valB (never used for any choice; its score is the one to trust).
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
import pyarrow as pa
import pyarrow.parquet as pq

from common import DATA_DIR, OUTPUT_DIR, SEED, assign_folds, read_tsv, up_to_date
from generate_candidates import CAND_DIR
from matcher_features import build_features, group_stats, idf_matrices, load_records, make_pool
from preprocess import PREP_DIR

WORK = DATA_DIR.parent / "work"
FEAT_DIR, MODEL_DIR, PRED_DIR = WORK / "features", WORK / "models", WORK / "pred"
STAGES = ["features", "train", "predict", "write", "check"]
META = ["s1_id", "cand_id", "country", "label"]
CHUNK = 2_000_000
EXPERIMENTS = {
    "lgb_fast": dict(learning_rate=0.10, num_leaves=127, min_child_samples=100,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
    "lgb_big": dict(learning_rate=0.05, num_leaves=255, min_child_samples=200,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0),
}
THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

LOG = logging.getLogger("er")
REPORT = []


def setup_logging():
    WORK.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(WORK / "pipeline.log", encoding="utf-8")):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    LOG.setLevel(logging.INFO)


def log(msg="", report=False):
    LOG.info(msg)
    if report:
        REPORT.append(msg)


def write_report():
    (WORK / "report.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


def countries_of(split):
    return sorted(pd.read_parquet(PREP_DIR / f"{split}_source1.parquet",
                                  columns=["country_key"])["country_key"].unique())


def load_gt():
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    return gt.loc[gt["m"].str.len() > 0, ["source1_entity_id", "m"]].rename(
        columns={"source1_entity_id": "s1"}).reset_index(drop=True)


# ================================================================ stage 1: features
def stage_features(args, pool):
    parts = ["train", "valA", "valB"]
    ids = pd.read_parquet(PREP_DIR / "train_source1.parquet", columns=["entity_id"])["entity_id"]
    folds = assign_folds(ids)
    rng = np.random.default_rng(SEED)
    val = np.array(sorted(k for k, v in folds.items() if v == "val"), dtype=object)
    rest = np.array(sorted(k for k, v in folds.items() if v != "val"), dtype=object)
    tr = rng.choice(rest, min(args.train_s1, len(rest)), replace=False)
    va = rng.choice(val, min(args.val_s1, len(val)), replace=False)
    part_of = {**{x: "train" for x in tr}, **{x: "valA" for x in va[: len(va) // 2]},
               **{x: "valB" for x in va[len(va) // 2:]}}
    log(f"sample: train={len(tr):,}  valA={len(va) // 2:,}  valB={len(va) - len(va) // 2:,} "
        "Source 1 entities", report=True)

    gt = load_gt()
    gt = gt[gt["s1"].isin(part_of.keys())]
    out, truths = {p: [] for p in parts}, []
    for country in countries_of("train"):
        t0 = time.time()
        s1rec, poolrec = load_records("train", country)
        s1stats, candstats = group_stats("train", country)
        idf = idf_matrices(s1rec, poolrec, args.threads, pool)
        here = [x for x in s1rec.index if x in part_of]
        cand = pq.read_table(CAND_DIR / f"train_{country}.parquet",
                             filters=[("s1_id", "in", here)]).to_pandas()
        g = gt[gt["s1"].isin(set(here))]
        truths.append(g.assign(country=country))
        lab = g.assign(label=1).rename(columns={"s1": "s1_id", "m": "cand_id"})
        cand = cand.merge(lab, on=["s1_id", "cand_id"], how="left")
        cand["label"] = cand["label"].fillna(0).astype(np.int8)
        log(f"  {country}: {len(here):,} S1 entities, {len(cand):,} candidate pairs, "
            f"{cand['label'].mean():.1%} positive")
        feats = [build_features(cand.iloc[s:s + CHUNK], s1rec, poolrec, s1stats, candstats, pool, idf)
                 for s in range(0, len(cand), CHUNK)]
        F = pd.concat(feats)
        F["s1_id"] = cand["s1_id"].to_numpy(object)
        F["cand_id"] = cand["cand_id"].to_numpy(object)
        F["country"] = country
        F["label"] = cand["label"].to_numpy()
        part = cand["s1_id"].map(part_of).to_numpy(object)
        for p in parts:
            out[p].append(F[part == p])
        del s1rec, poolrec, cand, F, feats
        log(f"  {country}: features done in {time.time() - t0:.0f}s")

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    truth = pd.concat(truths, ignore_index=True)
    truth["part"] = truth["s1"].map(part_of)
    truth.to_parquet(FEAT_DIR / "truth.parquet", index=False)
    for p in parts:
        pd.concat(out[p], ignore_index=True).to_parquet(FEAT_DIR / f"{p}.parquet", index=False)
    log(f"features saved to {FEAT_DIR}")


# ================================================================ stage 2: train
def keep_best_per_cand(df, prob):
    """One-to-one rule: each S2/S3 record keeps only its highest-probability S1 record."""
    d = df[["s1_id", "cand_id", "country", "label"]].assign(prob=prob)
    return d.sort_values("prob", ascending=False).drop_duplicates("cand_id")


def f05(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return (1.25 * p * r / (0.25 * p + r) if p + r else 0.0), p, r


def best_threshold(d, n_true):
    """Scan thresholds using cumulative counts (fast): d has prob and label."""
    probs = np.sort(d["prob"].to_numpy())[::-1]
    labels = d["label"].to_numpy()[np.argsort(-d["prob"].to_numpy(), kind="stable")]
    cum_tp = np.cumsum(labels)
    best = (-1.0, 0.5)
    for t in THRESH_GRID:
        n = int(np.searchsorted(-probs, -t, side="right"))
        f = f05(cum_tp[n - 1] if n else 0, n, n_true)[0]
        if f > best[0]:
            best = (f, float(t))
    return best[1]


def evaluate(d, truth, thresholds):
    """d: rows after (optional) one-to-one, with prob/label/country. Micro F0.5 overall
    and per country; truth includes pairs that blocking missed."""
    thr = d["country"].map(thresholds).fillna(thresholds["_default"]).to_numpy()
    pred = d["prob"].to_numpy() >= thr
    res = {"all": f05(int(d["label"].to_numpy()[pred].sum()), int(pred.sum()), len(truth))}
    for c in sorted(d["country"].unique()):
        m = d["country"].to_numpy() == c
        res[c] = f05(int(d["label"].to_numpy()[pred & m].sum()), int((pred & m).sum()),
                     int((truth["country"] == c).sum()))
    return res


def stage_train(args):
    tr = pd.read_parquet(FEAT_DIR / "train.parquet")
    vA = pd.read_parquet(FEAT_DIR / "valA.parquet")
    vB = pd.read_parquet(FEAT_DIR / "valB.parquet")
    truth = pd.read_parquet(FEAT_DIR / "truth.parquet")
    tA, tB = truth[truth["part"] == "valA"], truth[truth["part"] == "valB"]
    cols = [c for c in tr.columns if c not in META]
    log(f"train rows={len(tr):,} ({tr['label'].mean():.1%} positive)  valA={len(vA):,}  "
        f"valB={len(vB):,}  features={len(cols)}", report=True)
    for name, v, t in (("valA", vA, tA), ("valB", vB, tB)):
        log(f"blocking recall on {name}: {v['label'].sum() / len(t):.4f}  "
            f"(ceiling for recall)", report=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.experiments:
        t0 = time.time()
        model = lgb.LGBMClassifier(n_estimators=3000, random_state=SEED, n_jobs=args.threads,
                                   verbose=-1, **EXPERIMENTS[name])
        model.fit(tr[cols], tr["label"], eval_set=[(vA[cols], vA["label"])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
        model.booster_.save_model(str(MODEL_DIR / f"{name}.txt"))
        pA = model.predict_proba(vA[cols])[:, 1]
        pB = model.predict_proba(vB[cols])[:, 1]
        log(f"experiment {name}: {model.best_iteration_} trees in {time.time() - t0:.0f}s",
            report=True)
        for one2one in (True, False):
            dA = keep_best_per_cand(vA, pA) if one2one else vA[META].assign(prob=pA)
            dB = keep_best_per_cand(vB, pB) if one2one else vB[META].assign(prob=pB)
            g = best_threshold(dA, len(tA))
            per_c = {c: best_threshold(dA[dA["country"] == c], int((tA["country"] == c).sum()))
                     for c in sorted(dA["country"].unique())}
            for mode, thresholds in (("global", {"_default": g}),
                                     ("per_country", {**per_c, "_default": g})):
                rA, rB = evaluate(dA, tA, thresholds), evaluate(dB, tB, thresholds)
                results.append(dict(experiment=name, one_to_one=one2one, mode=mode,
                                    thresholds=thresholds, valA=rA["all"][0], valB=rB["all"],
                                    valB_by_country={k: v for k, v in rB.items() if k != "all"}))
                log(f"   one_to_one={one2one!s:5} thresholds={mode:11}  valA F0.5={rA['all'][0]:.4f}"
                    f"   valB F0.5={rB['all'][0]:.4f} (P={rB['all'][1]:.4f} R={rB['all'][2]:.4f})",
                    report=True)
        imp = sorted(zip(model.booster_.feature_importance("gain"), cols), reverse=True)[:12]
        log("   top features (gain): " + ", ".join(n for _, n in imp), report=True)

    best = max(results, key=lambda r: r["valA"])     # chosen on valA only
    best["feature_cols"] = cols
    (MODEL_DIR / "best.json").write_text(json.dumps(best, indent=2))
    log("", report=True)
    log(f"CHOSEN: {best['experiment']}, one_to_one={best['one_to_one']}, "
        f"thresholds={best['mode']} {best['thresholds']}", report=True)
    f, p, r = best["valB"]
    log(f"EXPECTED SCORE (valB, untouched): F0.5={f:.4f}  precision={p:.4f}  recall={r:.4f}",
        report=True)
    for c, (fc, pc_, rc) in best["valB_by_country"].items():
        log(f"   {c:8} F0.5={fc:.4f}  precision={pc_:.4f}  recall={rc:.4f}", report=True)


# ================================================================ stage 3: predict
def stage_predict(args, pool):
    best = json.loads((MODEL_DIR / "best.json").read_text())
    booster = lgb.Booster(model_file=str(MODEL_DIR / f"{best['experiment']}.txt"))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for country in countries_of("test"):
        dst = PRED_DIR / f"test_{country}.parquet"
        if "predict" not in args.redo and up_to_date(
                [dst], [MODEL_DIR / "best.json", CAND_DIR / f"test_{country}.parquet"]):
            log(f"  skip {dst.name} (up to date)")
            continue
        t0 = time.time()
        s1rec, poolrec = load_records("test", country)
        s1stats, candstats = group_stats("test", country)
        idf = idf_matrices(s1rec, poolrec, args.threads, pool)
        pf = pq.ParquetFile(CAND_DIR / f"test_{country}.parquet")
        tmp = dst.with_suffix(".parquet.tmp")
        writer, done = None, 0
        for batch in pf.iter_batches(batch_size=CHUNK):
            cand = batch.to_pandas()
            F = build_features(cand, s1rec, poolrec, s1stats, candstats, pool, idf)[best["feature_cols"]]
            prob = booster.predict(F.to_numpy(np.float32), num_threads=args.threads)
            table = pa.table({"s1_id": batch.column("s1_id"), "cand_id": batch.column("cand_id"),
                              "prob": pa.array(prob.astype(np.float32))})
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            done += len(cand)
            log(f"    {country}: {done:,} / {pf.metadata.num_rows:,} pairs scored "
                f"({time.time() - t0:.0f}s)")
        if writer is not None:
            writer.close()
            tmp.replace(dst)
        log(f"  {country}: predictions saved in {time.time() - t0:.0f}s")


# ================================================================ stage 4: write
def stage_write(args):
    best = json.loads((MODEL_DIR / "best.json").read_text())
    thresholds = best["thresholds"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m_path, c_path = OUTPUT_DIR / "matching_results.tsv", OUTPUT_DIR / "candidate_pairs.tsv"
    m_tmp, c_tmp = m_path.with_suffix(".tmp"), c_path.with_suffix(".tmp")
    log("", report=True)
    log("TEST PREDICTIONS", report=True)
    with open(m_tmp, "w", encoding="utf-8", newline="\n") as fm, \
            open(c_tmp, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in countries_of("test"):
            thr = thresholds.get(country, thresholds["_default"])
            s1_ids = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"],
                                     filters=[("country_key", "=", country)])["entity_id"]
            ct = pq.read_table(CAND_DIR / f"test_{country}.parquet",
                               columns=["s1_id", "cand_id"]).to_pandas()
            cands = ct.groupby("s1_id", sort=False)["cand_id"].agg(",".join)
            del ct
            pr = pq.read_table(PRED_DIR / f"test_{country}.parquet",
                               filters=[("prob", ">=", thr)]).to_pandas()
            if best["one_to_one"]:
                pr = pr.sort_values("prob", ascending=False).drop_duplicates("cand_id")
            matches = pr.groupby("s1_id")["cand_id"].agg(lambda x: ",".join(sorted(x)))
            c_col = cands.reindex(s1_ids).fillna("").to_numpy(object)
            m_col = matches.reindex(s1_ids).fillna("").to_numpy(object)
            ids = s1_ids.to_numpy(object)
            fc.write("".join(f"{a}\t{b}\n" for a, b in zip(ids, c_col)))
            fm.write("".join(f"{a}\t{b}\n" for a, b in zip(ids, m_col)))
            n = len(ids)
            with_m = int((m_col != "").sum())
            log(f"  {country:8} S1={n:,}  threshold={thr:.2f}  with matches={with_m / n:.1%}  "
                f"matched pairs={len(pr):,} ({len(pr) / n:.2f} per S1)", report=True)
    m_tmp.replace(m_path)
    c_tmp.replace(c_path)
    log("  (train ground truth for comparison: 94.4% of S1 have matches, 3.46 per S1)", report=True)
    log(f"wrote {m_path} ({m_path.stat().st_size / 1e6:.0f} MB) and {c_path} "
        f"({c_path.stat().st_size / 1e6:.0f} MB)", report=True)


# ================================================================ stage 5: check
def stage_check(args):
    ok, problems = True, []
    s1 = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"]
    res = read_tsv(OUTPUT_DIR / "matching_results.tsv")
    cand = read_tsv(OUTPUT_DIR / "candidate_pairs.tsv")
    for name, df in (("matching_results", res), ("candidate_pairs", cand)):
        if df["source1_entity_id"].duplicated().any():
            problems.append(f"{name}: duplicate Source 1 rows")
        if len(df) != len(s1) or set(df["source1_entity_id"]) != set(s1):
            problems.append(f"{name}: rows do not match the test Source 1 ids exactly")
    cand_map = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
    bad_subset = dup = 0
    for sid, lst in zip(res["source1_entity_id"], res["matched_entity_ids"]):
        if not lst:
            continue
        ids = lst.split(",")
        dup += len(ids) != len(set(ids))
        allowed = set(cand_map.get(sid, "").split(","))
        bad_subset += any(x not in allowed for x in ids)
    if dup:
        problems.append(f"{dup} rows with duplicate ids in matched list")
    if bad_subset:
        problems.append(f"{bad_subset} rows with matches not in candidate_pairs.tsv")
    # spot-check that ids exist in the test S2/S3 files (a full check needs too much RAM)
    pool_ids = set()
    for k in (2, 3):
        pool_ids.update(pd.read_parquet(PREP_DIR / f"test_source{k}.parquet",
                                        columns=["entity_id"])["entity_id"])
    nonempty = res.loc[res["matched_entity_ids"] != "", "matched_entity_ids"]
    sample = nonempty.sample(min(20_000, len(nonempty)), random_state=SEED)
    unknown = sum(x not in pool_ids for lst in sample for x in lst.split(","))
    if unknown:
        problems.append(f"{unknown} unknown ids in a 20k-row sample of matching_results")
    ok = not problems
    log("", report=True)
    log("SUBMISSION CHECK: " + ("PASSED" if ok else "FAILED: " + "; ".join(problems)), report=True)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-s1", type=int, default=300_000)
    ap.add_argument("--val-s1", type=int, default=100_000)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--redo", nargs="*", default=[], choices=STAGES,
                    help="redo these stages (and every later stage)")
    args = ap.parse_args()
    if args.redo:   # redoing a stage invalidates everything after it
        args.redo = STAGES[min(STAGES.index(s) for s in args.redo):]

    setup_logging()
    REPORT.append(f"# Pipeline report ({time.strftime('%Y-%m-%d %H:%M')})\n")
    feat_files = [FEAT_DIR / f"{p}.parquet" for p in ("train", "valA", "valB", "truth")]
    upstream = sorted(CAND_DIR.glob("*.parquet")) + sorted(PREP_DIR.glob("*.parquet"))
    done = {   # evaluated when each stage is reached, so a redone stage invalidates later ones
        "features": lambda: up_to_date(feat_files, upstream),
        "train": lambda: up_to_date([MODEL_DIR / "best.json"], feat_files),
        "predict": lambda: False,   # decided per country inside the stage
        "write": lambda: up_to_date([OUTPUT_DIR / "matching_results.tsv", OUTPUT_DIR / "candidate_pairs.tsv"],
                                    sorted(PRED_DIR.glob("test_*.parquet")) + [MODEL_DIR / "best.json"]),
        "check": lambda: False,
    }
    t_all = time.time()
    with make_pool(args.threads) as pool:
        for stage in STAGES:
            is_done = done[stage]()
            if is_done and stage not in args.redo and stage != "train":
                log(f"== {stage}: already done, skipping")
                continue
            if stage == "train" and is_done and "train" not in args.redo:
                log("== train: already done, re-reading results")
                best = json.loads((MODEL_DIR / "best.json").read_text())
                log(f"CHOSEN: {best['experiment']}, one_to_one={best['one_to_one']}, "
                    f"thresholds={best['mode']}", report=True)
                log(f"EXPECTED SCORE (valB): F0.5={best['valB'][0]:.4f}", report=True)
                continue
            t0 = time.time()
            log(f"== {stage}: starting")
            {"features": lambda: stage_features(args, pool), "train": lambda: stage_train(args),
             "predict": lambda: stage_predict(args, pool), "write": lambda: stage_write(args),
             "check": lambda: stage_check(args)}[stage]()
            log(f"== {stage}: finished in {(time.time() - t0) / 60:.1f} min")
            write_report()
    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min. Summary: {WORK / 'report.md'}")
    write_report()


if __name__ == "__main__":   # required on Windows for multiprocessing
    main()
