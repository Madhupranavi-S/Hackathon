"""STEP 2: measure facts about the data that decide how blocking is built.

    python -u explore.py

Read-only: it just prints a report. Paste the whole output back.
"""
import time

import numpy as np
import pandas as pd

from common import DATA_DIR, SEED, read_tsv
from preprocess import PREP_DIR

LEAN = ["entity_id", "country_key", "name_core", "nums", "postal", "name_nonlatin"]
PAIR = ["entity_id", "country_key", "name_core", "addr_n", "nums", "postal"]
RAW = ["entity_id", "business_name", "business_address", "country"]
T0 = time.time()


def log(msg=""):
    print(msg, flush=True)


def path(split, k):
    return PREP_DIR / f"{split}_source{k}.parquet"


def load(split, k, cols):
    return pd.read_parquet(path(split, k), columns=cols, dtype_backend="pyarrow")


def raw_rows(split, ids):
    ids = list(ids)
    parts = [pd.read_parquet(path(split, k), columns=RAW, filters=[("entity_id", "in", ids)])
             for k in (1, 2, 3)]
    return pd.concat(parts).set_index("entity_id")


def pct(x):
    return f"{100 * x:5.1f}%"


def section_counts():
    log("=== 1. records per source and country ===")
    rows = []
    for split in ("train", "test"):
        for k in (1, 2, 3):
            vc = load(split, k, ["country"])["country"].value_counts()
            rows.append(pd.Series(vc, name=f"{split} S{k}"))
    log(pd.DataFrame(rows).fillna(0).astype(int).to_string())


def section_coverage(s1, pool):
    log("\n=== 2. field coverage in train (share of records that have it) ===")
    for name, df in (("S1", s1), ("S2+S3", pool)):
        g = df.assign(has_postal=df["postal"] != "", has_num=df["nums"] != "",
                      empty_name=df["name_core"] == "",
                      nonlatin_name=df["name_nonlatin"].astype(bool)).groupby("country_key")
        log(f"{name}:\n" + g[["has_postal", "has_num", "empty_name", "nonlatin_name"]].mean()
            .map(pct).to_string())


def section_ground_truth(s1, pool):
    log("\n=== 3. ground truth ===")
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    lists = gt["matched_entity_ids"].str.split(",")
    pairs = (gt.assign(m=lists).explode("m")[["source1_entity_id", "m"]])
    pairs = pairs[pairs["m"].str.len() > 0].rename(columns={"source1_entity_id": "s1"})
    n_match = lists.map(lambda l: sum(1 for x in l if x))
    log(f"GT rows: {len(gt):,}   pairs: {len(pairs):,}   S1 ids not in GT file: "
        f"{(~s1['entity_id'].isin(gt['source1_entity_id'])).sum():,}")
    dist = n_match.clip(upper=6).value_counts(normalize=True).sort_index()
    log("matches per S1 record (6 = 6+): " + "  ".join(f"{k}:{pct(v)}" for k, v in dist.items()))
    log(f"pairs pointing to S2: {pct(pairs['m'].str.startswith('S2-').mean())}   to S3: "
        f"{pct(pairs['m'].str.startswith('S3-').mean())}")
    per_rec = pairs["m"].value_counts()
    log(f"S2/S3 records matched to >1 S1: {(per_rec > 1).sum():,}")
    log(f"S2/S3 records matched to no S1 (pure distractors): "
        f"{pct(1 - len(per_rec) / len(pool))}")
    log(f"GT ids missing from the files: {(~pairs['m'].isin(pool['entity_id'])).sum():,}")
    return pairs, n_match.set_axis(gt["source1_entity_id"].values)


def _tokens(s):
    return set(s.split()) if isinstance(s, str) else set()


def _words(s):
    """Address words that carry information: no pure numbers, no 1-2 letter codes."""
    return {t for t in _tokens(s) if len(t) >= 3 and not t.isdigit()}


def _fetch(split, ids):
    """Load PAIR columns only for the given ids (keeps memory low)."""
    ids = list(set(ids))
    parts = [pd.read_parquet(path(split, k), columns=PAIR, filters=[("entity_id", "in", ids)])
             for k in (1, 2, 3)]
    return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id")


def section_pair_agreement(pairs, n=300_000):
    log(f"\n=== 4. what true matches have in common (random {n:,} GT pairs) ===")
    smp = pairs.sample(min(n, len(pairs)), random_state=SEED)
    rec = _fetch("train", list(smp["s1"]) + list(smp["m"]))
    a, b = rec.loc[smp["s1"].values], rec.loc[smp["m"].values]
    rows = []
    for ca, cb, na, nb, aa, ab, pa, pb, ua, ub in zip(
            a["country_key"], b["country_key"], a["name_core"], b["name_core"],
            a["addr_n"], b["addr_n"], a["postal"], b["postal"], a["nums"], b["nums"]):
        ta, tb = _tokens(na), _tokens(nb)
        wa, wb = _words(aa), _words(ab)
        sa, sb = _tokens(pa), _tokens(pb)
        xa, xb = _tokens(ua), _tokens(ub)
        name_tok, addr_word = bool(ta & tb), bool(wa & wb)
        rows.append((ca, ca != cb, na == nb, name_tok,
                     bool(sa & sb) if sa and sb else np.nan,
                     bool(xa & xb) if xa and xb else np.nan,
                     not wa or not wb, addr_word,
                     len(wa & wb) / len(wa | wb) if wa and wb else np.nan,
                     name_tok or addr_word, not name_tok and not addr_word))
    d = pd.DataFrame(rows, columns=[
        "country", "cross_country", "same_core_name", "share_name_token", "postal_agrees",
        "share_any_number", "either_addr_empty", "share_addr_word", "addr_word_jaccard",
        "share_name_or_addr_word", "share_nothing"])
    log(d.groupby("country").mean().map(pct).T.to_string())
    log("(postal_agrees / share_any_number / addr_word_jaccard: only where both sides have one)")
    miss = smp[d["share_nothing"].values].head(8)
    if len(miss):
        log("\nexamples of true pairs sharing NO name token and NO address word:")
        raw = raw_rows("train", list(miss["s1"]) + list(miss["m"]))
        for s1_id, m in zip(miss["s1"], miss["m"]):
            log(f"  {raw.loc[s1_id, 'business_name']} | {raw.loc[s1_id, 'business_address']}")
            log(f"    = {raw.loc[m, 'business_name']} | {raw.loc[m, 'business_address']}")


def section_examples(s1, pairs, n_match):
    log("\n=== 5. examples of matched groups (train) ===")
    rng = np.random.default_rng(SEED)
    multi = set(n_match[n_match >= 2].index)
    for c in sorted(s1["country_key"].unique()):
        ids = s1.loc[(s1["country_key"] == c) & s1["entity_id"].isin(multi), "entity_id"]
        for sid in rng.choice(ids.to_numpy(), size=min(3, len(ids)), replace=False):
            ms = pairs.loc[pairs["s1"] == sid, "m"].tolist()
            raw = raw_rows("train", [sid] + ms)
            log(f"--- {c}")
            for eid in [sid] + ms:
                r = raw.loc[eid]
                log(f"  {eid:12} | {r['business_name']} | {r['business_address']}")

    log("\n=== 6. test-only countries (not seen in training) ===")
    test_s1 = load("test", 1, ["entity_id", "country_key"])
    new = sorted(set(test_s1["country_key"]) - set(s1["country_key"]))
    log(f"new country labels in test: {new}")
    for c in new:
        ids = test_s1.loc[test_s1["country_key"] == c, "entity_id"].to_numpy()
        raw = raw_rows("test", rng.choice(ids, size=min(6, len(ids)), replace=False))
        for eid, r in raw.iterrows():
            log(f"  {eid:12} | {r['business_name']} | {r['business_address']}")


def main():
    try:
        import psutil
        log(f"RAM total: {psutil.virtual_memory().total / 2**30:.1f} GB")
    except ImportError:
        log("RAM total: (install psutil to show this, or check Task Manager)")
    section_counts()
    s1 = load("train", 1, LEAN)
    pool = pd.concat([load("train", k, LEAN) for k in (2, 3)], ignore_index=True)
    section_coverage(s1, pool)
    pairs, n_match = section_ground_truth(s1, pool)
    del pool
    section_pair_agreement(pairs)
    section_examples(s1, pairs, n_match)
    log(f"\nfinished in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
