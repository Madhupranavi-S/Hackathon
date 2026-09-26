"""STEP 5: generate candidate pairs for EVERY Source 1 record and save them.

    python -u generate_candidates.py               # train and test (about an hour)
    python -u generate_candidates.py --split test  # just one split
    python -u generate_candidates.py --force       # redo files that already exist

Settings locked in from Step 4: retriever combo_ph, forward top-20 + reverse top-3.
Per split and country it writes dataset/candidates/<split>_<country>.parquet with:
    s1_id, cand_id   the pair
    score            cosine similarity of the two records' token vectors
    fwd_rank         rank of cand among this S1 record's candidates (K_FWD = not in top-k)
    rev_rank         rank of this S1 record among cand's best S1 records (K_REV = not in top-k)
For train it also prints the blocking recall over ALL ground-truth pairs.
If interrupted, run it again: finished files are skipped.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking_eval import build_matrix, load_country, log, topk_pairs
from common import DATA_DIR, read_tsv
from preprocess import PREP_DIR

CAND_DIR = DATA_DIR / "candidates"
RETRIEVER = "combo_ph"
K_FWD, K_REV, MAX_DF = 20, 3, 5_000


def rowwise_dot(A, B, i, j, chunk=2_000_000):
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(
            A[i[s:s + chunk]].multiply(B[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def run_country(split, country, threads, gt):
    t_start = time.time()
    s1, pool = load_country(split, country)
    n1 = len(s1)
    log(f"\n=== {split} / {country}: S1={n1:,}  pool={len(pool):,}")

    name_n = np.concatenate([s1["name_n"].to_numpy(object), pool["name_n"].to_numpy(object)])
    core = np.concatenate([s1["name_core"].to_numpy(object), pool["name_core"].to_numpy(object)])
    addr = np.concatenate([s1["addr_n"].to_numpy(object), pool["addr_n"].to_numpy(object)])
    X = build_matrix(RETRIEVER, name_n, core, addr, MAX_DF, threads)
    del name_n, core, addr
    A, B = X[:n1], X[n1:]
    del X

    log(f"  forward search: {n1:,} S1 records -> top {K_FWD}")
    fwd = topk_pairs(A, B.T.tocsr(), K_FWD, threads, progress=True)
    fwd = fwd.rename(columns={"q": "i", "rank": "fwd_rank"})[["i", "j", "fwd_rank"]]
    log(f"  reverse search: {len(pool):,} pool records -> top {K_REV}")
    rev = topk_pairs(B, A.T.tocsr(), K_REV, threads, progress=True)
    rev = rev.rename(columns={"q": "j", "j": "i", "rank": "rev_rank"})[["i", "j", "rev_rank"]]

    cand = fwd.merge(rev, on=["i", "j"], how="outer")
    del fwd, rev
    cand["fwd_rank"] = cand["fwd_rank"].fillna(K_FWD).astype(np.int16)
    cand["rev_rank"] = cand["rev_rank"].fillna(K_REV).astype(np.int16)
    cand["score"] = rowwise_dot(A, B, cand["i"].to_numpy(), cand["j"].to_numpy())
    cand = cand.sort_values(["i", "score"], ascending=[True, False], ignore_index=True)

    s1_ids = pa.array(s1["entity_id"].to_numpy(object), type=pa.string())
    pool_ids = pa.array(pool["entity_id"].to_numpy(object), type=pa.string())
    table = pa.table({
        "s1_id": s1_ids.take(pa.array(cand["i"].to_numpy())),
        "cand_id": pool_ids.take(pa.array(cand["j"].to_numpy())),
        "score": cand["score"].to_numpy(),
        "fwd_rank": cand["fwd_rank"].to_numpy(),
        "rev_rank": cand["rev_rank"].to_numpy(),
    })
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    dst = CAND_DIR / f"{split}_{country}.parquet"
    tmp = dst.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dst)

    per_s1 = len(cand) / max(n1, 1)
    no_cand = 1 - cand["i"].nunique() / max(n1, 1)
    log(f"  saved {dst.name}: {len(cand):,} pairs  ({per_s1:.1f} per S1 record, "
        f"{100 * no_cand:.2f}% of S1 records have none)  in {time.time() - t_start:.0f}s")

    if gt is not None:   # train only: recall over every true pair of this country
        g = gt[gt["s1"].isin(set(s1["entity_id"]))]
        found = pd.DataFrame({"s1": table["s1_id"].to_numpy(zero_copy_only=False),
                              "m": table["cand_id"].to_numpy(zero_copy_only=False)})
        hit = g.merge(found, on=["s1", "m"], how="inner")
        log(f"  blocking recall over ALL {len(g):,} true pairs: {len(hit) / max(len(g), 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    log(f"settings: retriever={RETRIEVER}  forward top-{K_FWD}  reverse top-{K_REV}  "
        f"max_df={MAX_DF:,}  threads={args.threads}")
    for split in splits:
        gt = None
        if split == "train":
            gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
            gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
            gt = gt.loc[gt["m"].str.len() > 0, ["source1_entity_id", "m"]].rename(
                columns={"source1_entity_id": "s1"})
        countries = sorted(pd.read_parquet(PREP_DIR / f"{split}_source1.parquet",
                                           columns=["country_key"])["country_key"].unique())
        log(f"\n{split}: countries {countries}")
        for country in countries:   # open set: whatever labels the file contains
            if (CAND_DIR / f"{split}_{country}.parquet").exists() and not args.force:
                log(f"  skip {split}_{country}.parquet (exists; use --force to redo)")
                continue
            run_country(split, country, args.threads, gt)


if __name__ == "__main__":   # required on Windows for multiprocessing
    main()
