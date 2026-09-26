"""Features for candidate pairs, computed in bulk (millions of pairs at a time).

String similarities use rapidfuzz's multi-threaded cpdist; set-based features run in a
process pool. Everything is float32, NaN = "unknown" (LightGBM handles NaN natively).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from blocking_eval import build_matrix, sound_key
from generate_candidates import CAND_DIR, K_FWD, K_REV
from preprocess import PREP_DIR

REC_COLS = ["entity_id", "name_n", "name_core", "addr_n", "nums", "postal", "name_nonlatin"]
PY_FEATS = ["addr_word_jacc", "addr_word_shared", "num_jacc", "num_conflict",
            "postal_match", "sound_jacc", "glued_contains"]


# ---------------------------------------------------------------- inputs
def load_records(split, country):
    """Normalized records of one country, indexed by entity_id."""
    filt = [("country_key", "=", country)]
    s1 = pd.read_parquet(PREP_DIR / f"{split}_source1.parquet", columns=REC_COLS, filters=filt)
    pool = pd.concat([pd.read_parquet(PREP_DIR / f"{split}_source{k}.parquet",
                                      columns=REC_COLS, filters=filt) for k in (2, 3)],
                     ignore_index=True)
    return s1.set_index("entity_id"), pool.set_index("entity_id")


def group_stats(split, country):
    """Per-S1 and per-candidate statistics over the COMPLETE candidate file, so context
    features are the same whether we look at a sample (train) or everything (test)."""
    t = pq.read_table(CAND_DIR / f"{split}_{country}.parquet", columns=["s1_id", "cand_id", "score"])
    s = t.group_by("s1_id").aggregate([("score", "max"), ("score", "count")]).to_pandas()
    c = t.group_by("cand_id").aggregate([("score", "max"), ("score", "count")]).to_pandas()
    s = s.set_index("s1_id").rename(columns={"score_max": "s1_max", "score_count": "s1_n"})
    c = c.set_index("cand_id").rename(columns={"score_max": "cand_max", "score_count": "cand_n"})
    return s, c


def idf_matrices(s1rec, poolrec, workers, pool=None):
    """Word-level TF-IDF of core names and addresses for S1 + pool of one country.
    Rare words weigh a lot, common ones ('club', 'roubaix', 'street') almost nothing."""
    core = np.concatenate([s1rec["name_core"].to_numpy(object), poolrec["name_core"].to_numpy(object)])
    addr = np.concatenate([s1rec["addr_n"].to_numpy(object), poolrec["addr_n"].to_numpy(object)])
    n1 = len(s1rec)
    out = {}
    for kind in ("name_words", "addr_words"):
        X = build_matrix(kind, core, core, addr, 10 ** 12, workers, pool=pool)
        out[kind] = (X[:n1], X[n1:])
    return out


def _rowwise_dot(A, B, i, j, chunk=2_000_000):
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(A[i[s:s + chunk]].multiply(B[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


# ---------------------------------------------------------------- python-side features
def _py_chunk(job):
    a_addr, b_addr, a_nums, b_nums, a_post, b_post, a_core, b_core = job
    out = np.full((len(a_addr), len(PY_FEATS)), np.nan, dtype=np.float32)
    for r, (aa, ba, an, bn, ap, bp, ac, bc) in enumerate(
            zip(a_addr, b_addr, a_nums, b_nums, a_post, b_post, a_core, b_core)):
        wa = {t for t in aa.split() if len(t) >= 3 and not t.isdigit()}
        wb = {t for t in ba.split() if len(t) >= 3 and not t.isdigit()}
        if wa and wb:
            i = len(wa & wb)
            out[r, 0] = i / len(wa | wb)
            out[r, 1] = i
        na, nb = set(an.split()), set(bn.split())
        if na and nb:
            i = len(na & nb)
            out[r, 2] = i / len(na | nb)
            out[r, 3] = float(i == 0)            # both have numbers, none agree
        pa, pb = set(ap.split()), set(bp.split())
        if pa and pb:
            out[r, 4] = float(bool(pa & pb))
        ka = {sound_key(w) for w in ac.split()} - {""}
        kb = {sound_key(w) for w in bc.split()} - {""}
        if ka and kb:
            out[r, 5] = len(ka & kb) / len(ka | kb)
        ga, gb = ac.replace(" ", ""), bc.replace(" ", "")
        out[r, 6] = float(bool(ga) and bool(gb) and (ga in gb or gb in ga))
    return out


def _lengths(arr):
    return np.fromiter((len(x) for x in arr), dtype=np.float32, count=len(arr))


def _cp(scorer, a, b):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


# ---------------------------------------------------------------- main entry
def build_features(pairs, s1rec, poolrec, s1stats, candstats, pool, idf, chunk=100_000):
    """pairs: DataFrame with s1_id, cand_id, score, fwd_rank, rev_rank.
    Returns a float32 feature DataFrame aligned with pairs."""
    ia = s1rec.index.get_indexer(pairs["s1_id"])
    jb = poolrec.index.get_indexer(pairs["cand_id"])
    if (ia < 0).any() or (jb < 0).any():
        raise ValueError("candidate ids not found in prepared records")

    def col(df, name, idx):
        return df[name].to_numpy(object)[idx]

    a_n, b_n = col(s1rec, "name_n", ia), col(poolrec, "name_n", jb)
    a_c, b_c = col(s1rec, "name_core", ia), col(poolrec, "name_core", jb)
    a_a, b_a = col(s1rec, "addr_n", ia), col(poolrec, "addr_n", jb)

    f = pd.DataFrame(index=pairs.index)
    # blocking signals
    f["score"] = pairs["score"].to_numpy(np.float32)
    f["fwd_rank"] = pairs["fwd_rank"].to_numpy(np.float32)
    f["rev_rank"] = pairs["rev_rank"].to_numpy(np.float32)
    f["in_fwd"] = (f["fwd_rank"] < K_FWD).astype(np.float32)
    f["in_rev"] = (f["rev_rank"] < K_REV).astype(np.float32)
    f["rev_top1"] = (f["rev_rank"] == 0).astype(np.float32)
    # context: how this pair compares with the alternatives on each side
    s1_max = s1stats["s1_max"].reindex(pairs["s1_id"]).to_numpy(np.float32)
    cand_max = candstats["cand_max"].reindex(pairs["cand_id"]).to_numpy(np.float32)
    f["gap_s1"] = s1_max - f["score"].to_numpy()
    f["gap_cand"] = cand_max - f["score"].to_numpy()
    f["ratio_s1"] = f["score"].to_numpy() / np.maximum(s1_max, 1e-6)
    f["s1_n"] = s1stats["s1_n"].reindex(pairs["s1_id"]).to_numpy(np.float32)
    f["cand_n"] = candstats["cand_n"].reindex(pairs["cand_id"]).to_numpy(np.float32)
    f["is_s3"] = pairs["cand_id"].str.startswith("S3-").to_numpy(np.float32)
    # string similarities (multi-threaded C++)
    f["name_ratio"] = _cp(fuzz.ratio, a_n, b_n)
    f["name_tsort"] = _cp(fuzz.token_sort_ratio, a_n, b_n)
    f["core_tset"] = _cp(fuzz.token_set_ratio, a_c, b_c)
    f["core_partial"] = _cp(fuzz.partial_ratio, a_c, b_c)
    f["core_jw"] = _cp(JaroWinkler.normalized_similarity, a_c, b_c)
    f["addr_tset"] = _cp(fuzz.token_set_ratio, a_a, b_a)
    f["addr_tsort"] = _cp(fuzz.token_sort_ratio, a_a, b_a)
    f["addr_partial"] = _cp(fuzz.partial_ratio, a_a, b_a)
    # rarity-weighted overlap: sharing 'baobab' counts, sharing 'club' or 'roubaix' barely
    f["name_idf_cos"] = _rowwise_dot(*idf["name_words"], ia, jb)
    f["addr_idf_cos"] = _rowwise_dot(*idf["addr_words"], ia, jb)
    # lengths / emptiness / script
    la_c, lb_c = _lengths(a_c), _lengths(b_c)
    la_a, lb_a = _lengths(a_a), _lengths(b_a)
    f["core_len_min"] = np.minimum(la_c, lb_c)
    f["core_len_ratio"] = np.minimum(la_c, lb_c) / np.maximum(np.maximum(la_c, lb_c), 1)
    f["addr_len_a"], f["addr_len_b"] = la_a, lb_a
    f["addr_len_ratio"] = np.minimum(la_a, lb_a) / np.maximum(np.maximum(la_a, lb_a), 1)
    f["b_nonlatin"] = col(poolrec, "name_nonlatin", jb).astype(np.float32)
    # set-based features (process pool)
    arrays = (a_a, b_a, col(s1rec, "nums", ia), col(poolrec, "nums", jb),
              col(s1rec, "postal", ia), col(poolrec, "postal", jb), a_c, b_c)
    jobs = [tuple(x[s:s + chunk] for x in arrays) for s in range(0, len(pairs), chunk)]
    py = np.vstack(pool.map(_py_chunk, jobs)) if jobs else np.empty((0, len(PY_FEATS)))
    for k, name in enumerate(PY_FEATS):
        f[name] = py[:, k]
    return f.astype(np.float32)


def make_pool(workers):
    return Pool(workers)
