"""NEURAL TRACK in one command (after run_pipeline.py and run_stage2.py have run):

    python -u run_neural.py

1. choose_prune.py   candidate filter cutoff (skipped if work/prune.json exists)
2. neural_ce.py      train the cross-encoder on the GPU, score all filtered pairs
3. run_stage2.py     stage 2 retrained WITH the cross-encoder features
4. finalize.py       official-metric decisions -> output_final/ (+ organisers' validator)
5. predict_lb.py     predicted leaderboard score
Every step resumes by itself if interrupted: just run the same command again.
Extra options are passed to neural_ce.py, e.g.  python -u run_neural.py --batch 32
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALIDATOR_PLACES = [ROOT / "utils", ROOT.parent / "utils", ROOT.parent / "student_resource" / "utils"]


def run(args):
    print(f"\n######## {' '.join(Path(a).name if i == 0 else a for i, a in enumerate(args))}"
          f"  ({time.strftime('%H:%M')})", flush=True)
    code = subprocess.call([sys.executable, "-u"] + args, cwd=ROOT)
    if code != 0:
        sys.exit(f"STOPPED: {Path(args[0]).name} failed (exit code {code}). Fix it and run again.")


def main():
    t0 = time.time()
    if not (ROOT / "work" / "prune.json").exists():
        run([str(ROOT / "choose_prune.py")])
    run([str(ROOT / "neural_ce.py")] + sys.argv[1:])
    run([str(ROOT / "run_stage2.py")])
    run([str(ROOT / "finalize.py")])
    validator = next((p / "validate_submission.py" for p in VALIDATOR_PLACES
                      if (p / "validate_submission.py").exists()), None)
    if validator:
        run([str(validator), "--matching", str(ROOT / "output_final" / "matching_results.tsv"),
             "--candidate", str(ROOT / "output_final" / "candidate_pairs.tsv"),
             "--test-dir", str(ROOT / "dataset" / "test")])
    run([str(ROOT / "predict_lb.py")])
    print(f"\nDONE in {(time.time() - t0) / 60:.0f} min. Send work/finalize.txt, work/predict_lb.txt "
          "and work/report_stage2.md before uploading.", flush=True)


if __name__ == "__main__":
    main()
