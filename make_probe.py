"""Write a probe submission from the CURRENT stage-2 predictions with a stricter threshold
for countries never seen in training (France). Everything else stays as in output_v2.

    python -u make_probe.py                 # unseen countries at 0.95 -> output_probe/
    python -u make_probe.py --unseen 0.90
    python -u make_probe.py --final         # same, as a complete checked package in output_final/

Run this BEFORE run_all.py (run_all regenerates the predictions it reads).
If the leaderboard score goes UP with this file, the unseen country's precision was the problem.
"""
import argparse
import json

import pandas as pd
import pyarrow.parquet as pq

from common import DATA_DIR
from preprocess import PREP_DIR
from run_pipeline import MODEL_DIR, countries_of
from run_stage2 import PRED2_DIR
from run_pipeline import PRED_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseen", type=float, default=0.95)
    ap.add_argument("--final", action="store_true",
                    help="write output_final/ (matching_results.tsv + candidate_pairs.tsv) and check it")
    args = ap.parse_args()
    b2 = json.loads((MODEL_DIR / "best_stage2.json").read_text())
    # test-side sections use the model that actually produced the submission
    live = b2 if b2.get("use_stage2") else json.loads((MODEL_DIR / "best.json").read_text())
    live_dir = PRED2_DIR if b2.get("use_stage2") else PRED_DIR
    thr_map, seen = live["thresholds"], set(countries_of("train"))
    out = DATA_DIR.parent / ("output_final" if args.final else "output_probe")
    out.mkdir(exist_ok=True)
    path = out / ("matching_results.tsv" if args.final
                  else f"matching_results_unseen{int(round(args.unseen * 100))}.tsv")
    model_name = "stage 2" if b2.get("use_stage2") else "first model"
    print(f"model: {model_name}   thresholds: seen countries {thr_map}, unseen {args.unseen}")
    with open(path, "w", encoding="utf-8", newline="\n") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for c in countries_of("test"):
            t = args.unseen if c not in seen else thr_map.get(c, thr_map["_default"])
            s1_ids = pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"],
                                     filters=[("country_key", "=", c)])["entity_id"]
            pr = pq.read_table(live_dir / f"test_{c}.parquet", filters=[("prob", ">=", t)]).to_pandas()
            if live["one_to_one"]:
                pr = pr.sort_values("prob", ascending=False).drop_duplicates("cand_id")
            m = pr.groupby("s1_id")["cand_id"].agg(lambda x: ",".join(sorted(x)))
            col = m.reindex(s1_ids).fillna("").to_numpy(object)
            fm.write("".join(f"{a}\t{b}\n" for a, b in zip(s1_ids.to_numpy(object), col)))
            print(f"  {c:8} threshold={t:.2f}  matched pairs={len(pr):,}  "
                  f"with matches={(col != '').mean():.1%}")
    print(f"wrote {path}")
    if args.final:
        finalize(out, b2)


def finalize(out, b2):
    """Copy the matching candidate file and check every submission rule."""
    import shutil
    from common import read_tsv
    src = DATA_DIR.parent / ("output_v2" if b2.get("use_stage2") else "output") / "candidate_pairs.tsv"
    shutil.copyfile(src, out / "candidate_pairs.tsv")
    s1 = set(pd.read_parquet(PREP_DIR / "test_source1.parquet", columns=["entity_id"])["entity_id"])
    res = read_tsv(out / "matching_results.tsv")
    cand = read_tsv(out / "candidate_pairs.tsv")
    problems = []
    for name, df in (("matching_results", res), ("candidate_pairs", cand)):
        if df["source1_entity_id"].duplicated().any() or len(df) != len(s1) \
                or set(df["source1_entity_id"]) != s1:
            problems.append(f"{name}: rows do not match the test Source 1 ids exactly once")
    allowed = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
    bad = sum(any(x not in set(allowed.get(sid, "").split(",")) for x in lst.split(","))
              or len(lst.split(",")) != len(set(lst.split(",")))
              for sid, lst in zip(res["source1_entity_id"], res["matched_entity_ids"]) if lst)
    if bad:
        problems.append(f"{bad} rows with duplicate ids or ids not in candidate_pairs.tsv")
    print("FINAL PACKAGE CHECK: " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
    print(f"-> upload {out / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
