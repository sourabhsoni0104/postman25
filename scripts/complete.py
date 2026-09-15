"""Reproduce the measured submission with one command and stage logs.

Run with .venv/bin/python -m scripts.complete. All stages fail fast.
GPU experiments are sequential to avoid competing for memory and compute.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

STAGES = {
    "correctness": ["tests.test_correctness", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--dtype", "float32"],
    "attention": ["scripts.instrument_attention", "--n_tokens", "2048"],
    "curve": ["scripts.run_curve", "--texts", "data/pride_and_prejudice.txt,data/frankenstein.txt",
              "--n_tokens", "3072", "--ctx", "2048", "--budgets", "128,256,512,1024,2048",
              "--depths", "0.1,0.5,0.9", "--n_trials", "2", "--chunk_size", "32"],
    "rope": ["scripts.rope_ablation", "--n_tokens", "3072", "--ctx", "2048", "--budget", "512",
             "--depths", "0.9,0.95,0.98", "--n_trials", "2"],
    "benchmark": ["scripts.benchmark", "--n_tokens", "4096", "--budgets", "128,512,2048", "--n_decode", "32"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    env = dict(os.environ, OMP_NUM_THREADS="4", MPLCONFIGDIR="/tmp/kvcache-mpl", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    for name in args.stages.split(","):
        command = [sys.executable, "-m", *STAGES[name], "--device", args.device]
        print("RUN", " ".join(command), flush=True)
        with open(f"results/logs/{name}.log", "w") as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise SystemExit(f"Stage {name} failed; see results/logs/{name}.log")

if __name__ == "__main__":
    main()
