"""Validate output files against every rule in the challenge before uploading.

    python check_submission.py
"""
import sys

from common import OUTPUT_DIR, load_split, read_tsv


def parse(path, col):
    df = read_tsv(path)
    return df, {r["source1_entity_id"]: [x for x in r[col].split(",") if x]
                for _, r in df.iterrows()}


def main():
    s1, pool = load_split("test")
    s1_ids, pool_ids = set(s1["entity_id"]), set(pool["entity_id"])
    errors = []

    res_df, res = parse(OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids")
    cand_df, cand = parse(OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")

    for name, df, d in (("matching_results", res_df, res), ("candidate_pairs", cand_df, cand)):
        if df["source1_entity_id"].duplicated().any():
            errors.append(f"{name}: duplicate Source 1 rows")
        if set(d) != s1_ids:
            errors.append(f"{name}: missing {len(s1_ids - set(d))} S1 ids, "
                          f"unknown {len(set(d) - s1_ids)} S1 ids")
        for sid, ids in d.items():
            if len(ids) != len(set(ids)):
                errors.append(f"{name}: duplicate ids in list for {sid}")
            bad = [x for x in ids if x not in pool_ids]
            if bad:
                errors.append(f"{name}: {sid} lists ids not in test S2/S3: {bad[:3]}")

    missing = sum(len(set(ids) - set(cand.get(sid, []))) for sid, ids in res.items())
    if missing:
        errors.append(f"{missing} matched ids are not in candidate_pairs.tsv")

    if errors:
        print("FAILED:\n  " + "\n  ".join(errors[:20]))
        sys.exit(1)
    print(f"OK: {len(res)} rows, {sum(map(len, res.values()))} matched pairs, "
          f"{sum(map(len, cand.values()))} candidate pairs")


if __name__ == "__main__":
    main()
