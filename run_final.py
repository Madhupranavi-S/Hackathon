"""FINAL command, run after run_all.py has finished:

    python -u run_final.py

1. choose_prune.py      picks the candidate-filter cutoff (skipped if work/prune.json exists)
2. run_stage2.py        retrains stage 2 on the filtered candidates, predicts test
3. finalize.py          official-metric decisions per entity -> output_final/ + checks
4. utils/validate_submission.py (the organisers' validator), if it can be found
Stops at the first failing step.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALIDATOR_PLACES = [ROOT / "utils", ROOT.parent / "utils", ROOT.parent / "student_resource" / "utils"]


def run(args):
    print(f"\n######## {' '.join(args)}  ({time.strftime('%H:%M')})", flush=True)
    code = subprocess.call([sys.executable, "-u"] + args, cwd=ROOT)
    if code != 0:
        sys.exit(f"STOPPED: {args[0]} failed (exit code {code})")


def main():
    t0 = time.time()
    if not (ROOT / "work" / "prune.json").exists():
        run([str(ROOT / "choose_prune.py")])
    else:
        print("work/prune.json exists: keeping the chosen cutoff", flush=True)
    run([str(ROOT / "run_stage2.py"), "--redo", "s2features"])
    run([str(ROOT / "finalize.py")])
    validator = next((p / "validate_submission.py" for p in VALIDATOR_PLACES
                      if (p / "validate_submission.py").exists()), None)
    if validator:
        run([str(validator), "--matching", str(ROOT / "output_final" / "matching_results.tsv"),
             "--candidate", str(ROOT / "output_final" / "candidate_pairs.tsv"),
             "--test-dir", str(ROOT / "dataset" / "test")])
    else:
        print("\nOfficial validator not found. Copy the 'utils' folder from student_resource into "
              "er_project, or run it yourself on output_final/.", flush=True)
    print(f"\nDONE in {(time.time() - t0) / 60:.0f} min. Read work/prune.txt, work/report_stage2.md "
          "and work/finalize.txt before uploading output_final/matching_results.tsv", flush=True)


if __name__ == "__main__":
    main()
