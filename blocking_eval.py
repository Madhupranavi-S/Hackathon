"""STEP 4 (v2): scalable blocking, measured before we build anything on top of it.

    python -u blocking_eval.py                          # uni vs combo
    python -u blocking_eval.py --retrievers combo chars # add the char retriever
    python -u blocking_eval.py --max-df 20000           # try a different setting

Retrievers (each run separately per country; true matches never cross countries):
  uni:   IDF-weighted overlap of single words: name words, glued name, address words,
         house numbers. (The v1 'words' retriever, plus leading-zero fix.)
  combo: uni + COMBINATION tokens: adjacent address word pairs (nehru_nagar, 53_4),
         adjacent name word pairs (cozy_hair), name word x house number (tech#53).
         Each word may be common, but the combination is rare and specific.
  combo_ph: combo + SOUND keys of name words (consonant skeletons), for names spelled the
         way they sound after transliteration: 'southern shiva' ~ 'sdrn siva'.
  chars: character 3-4-grams of the core name (typos, transliteration). Optional.

Tokens seen in more than --max-df records are ignored; that keeps the search fast.
It samples validation-fold Source 1 records as queries, searches the FULL pool of their
country, and reports blocking recall. Nothing is written to disk.
"""
import argparse
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from common import DATA_DIR, SEED, assign_folds, read_tsv
from preprocess import PREP_DIR

COLS = ["entity_id", "country_key", "name_n", "name_core", "addr_n"]
KS = (5, 10, 20, 30, 50, 100)
N_FEATURES = 2 ** 23

# stateless, created at import time in every worker process
WORD_HV = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm=None,
                            binary=True, dtype=np.float32, tokenizer=str.split,
                            token_pattern=None, lowercase=False)
CHAR_HV = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm=None,
                            binary=True, dtype=np.float32, analyzer="char_wb",
                            ngram_range=(3, 4))


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- tokens
def _num(t):
    return t.lstrip("0") or "0"                    # 090 -> 90


def uni_tokens(name_n, core, addr):
    toks = set(core.split())
    toks.add(core.replace(" ", ""))                # 'digital tech' -> 'digitaltech'
    toks.add(name_n.replace(" ", ""))              # 'ams holdings' -> 'amsholdings'
    for t in addr.split():
        if t.isdigit():
            toks.add("#" + _num(t))
        elif len(t) >= 3:
            toks.add(t)
    toks.discard("")
    return toks


def combo_tokens(core, addr):
    c = core.split()
    a = [_num(t) if t.isdigit() else t for t in addr.split()]
    toks = {f"{x}_{y}" for x, y in zip(c, c[1:])}             # cozy_hair
    toks |= {f"{x}_{y}" for x, y in zip(a, a[1:])}            # nehru_nagar, 53_4
    nums = [t for t in a if t.isdigit()][:4]
    toks |= {f"{w}#{n}" for w in c[:4] for n in nums}         # tech#53
    return toks


_SOUND = [("ph", "f"), ("h", ""), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"),
          ("v", "w"), ("d", "t"), ("j", "g")]
_VOWELS = str.maketrans("", "", "aeiouy")
_REPEAT = re.compile(r"(.)\1+")


def sound_key(word):
    """Consonant skeleton: words that sound alike get the same key.
    'southern'/'sdrn' -> 'strn', 'white'/'vhait' -> 'wt', 'galaxy'/'gaileksi' -> 'glks'."""
    for a, b in _SOUND:
        word = word.replace(a, b)
    return _REPEAT.sub(r"\1", word.translate(_VOWELS))


def sound_tokens(core, addr):
    keys = [k for k in (sound_key(w) for w in core.split() if not w.isdigit()) if k]
    toks = {"~" + k for k in keys if len(k) >= 2}
    toks |= {f"~{x}_{y}" for x, y in zip(keys, keys[1:])}
    nums = [_num(t) for t in addr.split() if t.isdigit()][:4]
    toks |= {f"~{k}#{n}" for k in keys[:4] for n in nums}
    return toks


def _hash_chunk(job):
    """Runs in a worker process: text -> hashed sparse rows."""
    kind, name_n, core, addr = job
    if kind == "chars":
        return CHAR_HV.transform(core)
    if kind == "name_words":
        return WORD_HV.transform(core)
    if kind == "addr_words":
        return WORD_HV.transform(addr)
    if kind == "uni":
        docs = [" ".join(uni_tokens(x, y, z)) for x, y, z in zip(name_n, core, addr)]
    elif kind == "combo":
        docs = [" ".join(uni_tokens(x, y, z) | combo_tokens(y, z))
                for x, y, z in zip(name_n, core, addr)]
    else:  # combo_ph: combo + sound keys
        docs = [" ".join(uni_tokens(x, y, z) | combo_tokens(y, z) | sound_tokens(y, z))
                for x, y, z in zip(name_n, core, addr)]
    return WORD_HV.transform(docs)


def build_matrix(kind, name_n, core, addr, max_df, workers, chunk=200_000, pool=None):
    """Hashed TF-IDF: hash in parallel, drop tokens in <2 or >max_df records,
    weight by IDF, L2-normalise."""
    n = len(core)
    jobs = [(kind, name_n[s:s + chunk], core[s:s + chunk], addr[s:s + chunk])
            for s in range(0, n, chunk)]
    if pool is not None:              # reuse the caller's workers (no second set of processes)
        X = sparse.vstack(pool.map(_hash_chunk, jobs)).tocsr()
    else:
        with Pool(workers) as own:
            X = sparse.vstack(own.map(_hash_chunk, jobs)).tocsr()
    df = np.bincount(X.indices, minlength=X.shape[1])
    keep = (df >= 2) & (df <= max_df)
    idf = np.where(keep, np.log((1 + n) / (1 + df)) + 1, 0).astype(np.float32)
    X.data *= idf[X.indices]
    X.eliminate_zeros()
    normalize(X, copy=False)
    empty = (np.diff(X.indptr) == 0).mean()
    log(f"  {kind:5} features kept={int(keep.sum()):,}  nonzeros={X.nnz:,}  "
        f"records with NO usable token={100 * empty:.1f}%")
    return X


# ---------------------------------------------------------------- data & retrieval
def load_country(split, country):
    filt = [("country_key", "=", country)]
    s1 = pd.read_parquet(PREP_DIR / f"{split}_source1.parquet", columns=COLS, filters=filt)
    pool = pd.concat([pd.read_parquet(PREP_DIR / f"{split}_source{k}.parquet",
                                      columns=COLS, filters=filt) for k in (2, 3)],
                     ignore_index=True)
    return s1.reset_index(drop=True), pool


def topk_pairs(Q, P_T, k, threads, chunk=20_000, progress=False):
    """Top-k columns per query row -> DataFrame(q, j, rank, score)."""
    out, t0 = [], time.time()
    for s in range(0, Q.shape[0], chunk):
        if progress and s and s % (chunk * 25) == 0:
            log(f"      {s:>10,} / {Q.shape[0]:,} queries  {time.time() - t0:6.0f}s")
        R = sp_matmul_topn(Q[s:s + chunk], P_T, top_n=k, threshold=1e-9,
                           n_threads=threads, sort=True)
        counts = np.diff(R.indptr)
        q = np.repeat(np.arange(s, s + R.shape[0]), counts)
        rank = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.array([])
        out.append(pd.DataFrame({"q": q.astype(np.int32), "j": R.indices.astype(np.int32),
                                 "rank": rank.astype(np.int16), "score": R.data.astype(np.float32)}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- evaluation
def evaluate(country, args, gt, folds):
    t0 = time.time()
    s1, pool = load_country("train", country)
    n1 = len(s1)
    log(f"\n=== {country}: S1={n1:,}  pool={len(pool):,}  (loaded in {time.time() - t0:.0f}s)")

    val = s1.index[s1["entity_id"].map(folds).eq("val")].to_numpy()
    rng = np.random.default_rng(SEED)
    qrows = np.sort(rng.choice(val, size=min(args.queries, len(val)), replace=False))

    q_of = pd.Series(np.arange(len(qrows)), index=s1["entity_id"].values[qrows])
    j_of = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    g = gt[gt["s1"].isin(q_of.index)]
    truth = pd.DataFrame({"q": q_of[g["s1"]].values, "j": j_of.reindex(g["m"]).values})
    truth = truth.dropna().astype({"j": np.int64})
    log(f"  queries: {len(qrows):,}   true pairs: {len(truth):,}")

    name_n = np.concatenate([s1["name_n"].to_numpy(object), pool["name_n"].to_numpy(object)])
    core = np.concatenate([s1["name_core"].to_numpy(object), pool["name_core"].to_numpy(object)])
    addr = np.concatenate([s1["addr_n"].to_numpy(object), pool["addr_n"].to_numpy(object)])

    ranks, rev, rev_secs = {}, None, 0.0
    main = args.retrievers[-1]          # the last retriever listed is the one we evaluate in depth
    for kind in args.retrievers:
        t0 = time.time()
        X = build_matrix(kind, name_n, core, addr, args.max_df, args.threads)
        built = time.time() - t0
        t0 = time.time()
        found = topk_pairs(X[:n1][qrows], X[n1:].T.tocsr(), max(KS), args.threads)[["q", "j", "rank"]]
        secs = time.time() - t0
        if kind == main and args.rev_k:
            # REVERSE: each true pool record searches ALL S1 records of the country.
            # (The real pipeline will run this for every pool record.)
            t0 = time.time()
            J = truth["j"].unique()
            r = topk_pairs(X[n1:][J], X[:n1].T.tocsr(), max(args.rev_k), args.threads)[["q", "j", "rank"]]
            rev_secs = 1e5 * (time.time() - t0) / len(J)
            rev = pd.DataFrame({"s1row": r["j"].values, "j": J[r["q"].values],
                                "r_rev": r["rank"].values})
        del X
        ranks[kind] = found
        m = truth.merge(found, on=["q", "j"], how="left")["rank"]
        rec = "  ".join(f"@{k}:{(m < k).mean():.4f}" for k in KS)
        log(f"  {kind:5} recall {rec}")
        log(f"        built in {built:.0f}s, search {1e5 * secs / len(qrows):.0f}s per 100k queries")

    # union: main retriever (combo if present) at k1, every other retriever at k2
    others = [k for k in ranks if k != main]
    both = ranks[main].rename(columns={"rank": "r_" + main})
    for o in others:
        both = both.merge(ranks[o].rename(columns={"rank": "r_" + o}), on=["q", "j"], how="outer")
    t = truth.merge(both, on=["q", "j"], how="left")

    def hit(df, k1, k2):
        h = df["r_" + main] < k1
        for o in others:
            h |= df["r_" + o] < k2
        return h

    if others:
        log(f"  union ({main}@k1 + {'+'.join(others)}@k2):")
        settings = ((10, 5), (20, 5), (20, 10), (30, 10), (50, 10), (50, 20))
    else:
        log(f"  {main} alone:")
        settings = ((10, 0), (20, 0), (30, 0), (50, 0))
    for k1, k2 in settings:
        log(f"    k1={k1:3} k2={k2:3}  recall={hit(t, k1, k2).mean():.4f}   "
            f"candidates/query={hit(both, k1, k2).sum() / len(qrows):5.1f}")

    if rev is not None:
        t = t.assign(s1row=qrows[t["q"].values]).merge(rev, on=["s1row", "j"], how="left")
        per_s1 = len(pool) / n1
        log(f"  reverse search ({main}, each pool record -> its top S1 records): "
            f"{rev_secs:.0f}s per 100k pool records")
        m = t["r_rev"]
        log("    reverse alone: " + "  ".join(f"@{k}:{(m < k).mean():.4f}" for k in args.rev_k))
        log(f"  forward {main}@k1 + reverse@kr   (reverse adds at most kr x {per_s1:.1f} "
            f"candidates per S1 record)")
        for k1 in (10, 20, 30):
            f_size = (both["r_" + main] < k1).sum() / len(qrows)
            for kr in args.rev_k:
                h = (t["r_" + main] < k1) | (t["r_rev"] < kr)
                log(f"    k1={k1:3} kr={kr}  recall={h.mean():.4f}   candidates/query <= "
                    f"{f_size + kr * per_s1:5.1f}")
        t = t.assign(_rev_hit=t["r_rev"] < max(args.rev_k))
    else:
        t = t.assign(_rev_hit=False)

    miss = t[~(hit(t, 50, 20) | t["_rev_hit"])].sample(frac=1, random_state=SEED).head(args.show_misses)
    if len(miss):
        log("  examples still missed at the widest setting:")
        for q, j in zip(miss["q"], miss["j"]):
            a, b = s1.iloc[qrows[q]], pool.iloc[j]
            log(f"    {a['name_core']} | {a['addr_n']}")
            log(f"      = {b['name_core']} | {b['addr_n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="+", default=["combo", "combo_ph"],
                    choices=["uni", "combo", "combo_ph", "chars"])
    ap.add_argument("--queries", type=int, default=20_000, help="per country")
    ap.add_argument("--max-df", type=int, default=5_000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--rev-k", type=int, nargs="*", default=[1, 2, 3, 5],
                    help="reverse-search depths to report; pass nothing to skip")
    ap.add_argument("--show-misses", type=int, default=8)
    ap.add_argument("--countries", nargs="*", help="default: all countries in train")
    args = ap.parse_args()

    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    gt = gt.loc[gt["m"].str.len() > 0, ["source1_entity_id", "m"]].rename(
        columns={"source1_entity_id": "s1"})
    ids = pd.read_parquet(PREP_DIR / "train_source1.parquet", columns=["entity_id", "country_key"])
    folds = assign_folds(ids["entity_id"])
    countries = args.countries or sorted(ids["country_key"].unique())
    del ids
    args.rev_k = sorted(args.rev_k)
    log(f"settings: retrievers={args.retrievers}  queries={args.queries:,}/country  "
        f"max_df={args.max_df:,}  threads={args.threads}")
    for country in countries:
        evaluate(country, args, gt, folds)


if __name__ == "__main__":   # required on Windows for multiprocessing
    main()
