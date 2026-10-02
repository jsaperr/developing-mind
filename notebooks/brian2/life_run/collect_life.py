"""Pulls the life run's compact files from the Modal Volume into this folder (the raw npz files stay on the Volume).

Usage (repo root, conda env):  python notebooks/brian2/life_run/collect_life.py
Prints which seeds are done; safe to re-run.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = list(range(54000, 54008))
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def modal(*args):
    return subprocess.run([sys.executable, "-m", "modal", "volume", *args], capture_output=True, text=True, env=ENV)


def main():
    ls = modal("ls", "developing-mind-raw", "life_run/compact")
    have = ls.stdout if ls.returncode == 0 else ""
    got = 0
    for s in SEEDS:
        name = f"life_seed{s}.json.gz"
        dst = HERE / name
        if dst.exists():
            got += 1; continue
        if name not in have:
            print(f"seed {s}: not done yet"); continue
        r = modal("get", "developing-mind-raw", f"life_run/compact/{name}", str(dst))
        print(f"seed {s}: {'pulled' if r.returncode == 0 else 'FAILED ' + r.stderr.strip()[-200:]}")
        got += r.returncode == 0
    print(f"== {got}/{len(SEEDS)} compact files here")


if __name__ == "__main__":
    main()
