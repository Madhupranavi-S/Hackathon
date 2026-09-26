"""Rerun the WHOLE pipeline after a normalization/feature change, in one command.

    python -u run_all.py

1. backs up output/, output_v2/ and the reports to backups/<date-time>/
2. preprocess.py --force          (new normalization)
3. generate_candidates.py --force (new candidates)
4. run_pipeline.py --redo features (new features, model, v1 submission)
5. run_stage2.py --redo fresh      (stage 2, v2 submission)
6. diagnose.py                     (metric table + French examples)
Stops at the first failing step. If it stops, fix the problem and run the step's own
command (shown in the log) to continue; each script resumes by itself.
Total time on a 12-core laptop: roughly 2.5 hours.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ["preprocess.py", "--force"],
    ["generate_candidates.py", "--force"],
    ["run_pipeline.py", "--redo", "features"],
    ["run_stage2.py", "--redo", "fresh"],
    ["diagnose.py"],
]


def main():
    stamp = time.strftime("%Y%m%d_%H%M")
    backup = ROOT / "backups" / stamp
    for name in ("output", "output_v2"):
        if (ROOT / name).exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ROOT / name), str(backup / name))
    for name in ("report.md", "report_stage2.md", "diagnose.txt"):
        if (ROOT / "work" / name).exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "work" / name, backup / name)
    shutil.rmtree(ROOT / "work" / "pred2", ignore_errors=True)   # stale after new candidates
    (ROOT / "work" / "prune.json").unlink(missing_ok=True)          # chosen again by run_final.py
    if backup.exists():
        print(f"previous outputs and reports backed up to {backup}", flush=True)

    t_all = time.time()
    for step in STEPS:
        cmd = [sys.executable, "-u", str(ROOT / step[0])] + step[1:]
        print(f"\n######## {' '.join(step)}  (started {time.strftime('%H:%M')})", flush=True)
        t0 = time.time()
        code = subprocess.call(cmd, cwd=ROOT)
        print(f"######## {step[0]} finished in {(time.time() - t0) / 60:.1f} min, exit code {code}",
              flush=True)
        if code != 0:
            print(f"\nSTOPPED: {step[0]} failed. Fix it, then continue with:\n"
                  f"   python -u {' '.join(step)}\nand the remaining steps after it.", flush=True)
            sys.exit(code)
    print(f"\nALL STEPS DONE in {(time.time() - t_all) / 60:.0f} min. Read work/report.md, "
          "work/report_stage2.md and work/diagnose.txt; upload output_v2/matching_results.tsv "
          "if stage 2 was used, else output/matching_results.tsv.", flush=True)


if __name__ == "__main__":
    main()
