"""STEP 1: normalize every source file ONCE, in parallel, and save as Parquet.

    python preprocess.py                 # train and test
    python preprocess.py --split train   # just one split
    python preprocess.py --force         # redo files that already exist

Reads each TSV in chunks (so memory stays bounded even for 10M+ rows), spreads the
normalization over all CPU cores, and writes dataset/prepared/<split>_source<k>.parquet.
Later steps load these files in seconds instead of re-normalizing for 25 minutes.
"""
import argparse
import csv
import os
import re
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import DATA_DIR, core_name, norm_addr, norm_name

PREP_DIR = DATA_DIR / "prepared"
_DIGITS = re.compile(r"\d+")


def normalize_chunk(df):
    """Runs inside a worker process. Sets are stored as space-separated strings
    because Parquet can't store Python sets."""
    name_n = [norm_name(x) for x in df["business_name"]]
    addr_n = [norm_addr(x) for x in df["business_address"]]
    nums = [" ".join(sorted(set(_DIGITS.findall(a)))) for a in addr_n]
    country_key = df["country"].str.strip().str.lower().values
    return pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "business_name": df["business_name"].values,
        "business_address": df["business_address"].values,
        "country": df["country"].values,
        "country_key": country_key,
        "name_n": name_n,
        "name_core": [core_name(n, c) for n, c in zip(name_n, country_key)],
        "name_nonlatin": [any(ord(ch) > 0x24F for ch in x) for x in df["business_name"]],
        "addr_n": addr_n,
        "nums": nums,
        "postal": [" ".join(n for n in ns.split() if len(n) in (5, 6)) for ns in nums],
    })


def process_file(src, dst, pool, n_parts, chunksize):
    t0, rows, writer = time.time(), 0, None
    reader = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE, chunksize=chunksize)
    tmp = dst.with_suffix(".parquet.tmp")   # only renamed when complete
    try:
        for chunk in reader:
            chunk = chunk.fillna("")
            step = -(-len(chunk) // n_parts)   # ceil division
            parts = [chunk.iloc[s:s + step] for s in range(0, len(chunk), step)]
            out = pd.concat(pool.map(normalize_chunk, parts), ignore_index=True)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(out)
            print(f"    {rows:>12,} rows  {time.time() - t0:7.1f}s", flush=True)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(dst)
    print(f"  done {dst.name}: {rows:,} rows in {time.time() - t0:.1f}s", flush=True)


def load_prepared(split, columns=None):
    """Used by later steps: returns (source1, pool) where pool = Source 2 + Source 3."""
    s1 = pd.read_parquet(PREP_DIR / f"{split}_source1.parquet", columns=columns)
    pool = pd.concat([pd.read_parquet(PREP_DIR / f"{split}_source{k}.parquet", columns=columns)
                      for k in (2, 3)], ignore_index=True)
    return s1, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    splits = ["train", "test"] if args.split == "all" else [args.split]
    print(f"using {args.workers} worker processes", flush=True)
    with Pool(args.workers) as pool:
        for split in splits:
            for k in (1, 2, 3):
                src = DATA_DIR / split / f"{split}_source{k}.tsv"
                dst = PREP_DIR / f"{split}_source{k}.parquet"
                if dst.exists() and not args.force:
                    print(f"  skip {dst.name} (exists; use --force to redo)")
                    continue
                print(f"  {src.name}", flush=True)
                process_file(src, dst, pool, args.workers * 4, args.chunksize)


if __name__ == "__main__":   # required on Windows for multiprocessing
    main()
