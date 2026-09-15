"""Benchmark: time and peak memory of streaming N tokens through the model with
each policy at several budgets, relative to the full (unbounded) cache.
Reports the hardware it ran on. Only relative numbers are meaningful.

    python -m scripts.benchmark --n_tokens 8192 --budgets 256,1024
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from kvcache.policies import make_policy, POLICY_NAMES
from kvcache.streaming import StreamingRunner
from kvcache.utils import (build_model, hardware_report, Timer, reset_peak_memory,
                           peak_memory_bytes, save_json, DEFAULT_MODEL)


def bench_one(model, ids, policy_name, budget, chunk_size, n_decode):
    runner = StreamingRunner(model, make_policy(policy_name), budget=budget, chunk_size=chunk_size)
    reset_peak_memory()
    with Timer() as t_pre:
        runner.prefill(ids)
    with Timer() as t_dec:
        nxt = torch.tensor([[ids[0, -1].item()]], device=ids.device)
        for _ in range(n_decode):
            runner._step(nxt, last_only=True)
    return {
        "policy": policy_name, "budget": budget,
        "prefill_s": t_pre.seconds, "decode_s_per_token": t_dec.seconds / n_decode,
        "peak_mem_bytes": peak_memory_bytes(), "kv_bytes_final": runner.cache.memory_bytes(),
        "cache_T": runner.cache.T,
        "peak_kv_bytes": runner.peak_kv_bytes, "metadata_bytes": runner.cache.metadata_bytes(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n_tokens", type=int, default=8192)
    ap.add_argument("--n_decode", type=int, default=64)
    ap.add_argument("--budgets", default="256,1024")
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()

    hw = hardware_report()
    print("hardware:", hw)
    model, tok, _ = build_model(args.model, args.device, args.dtype)
    torch.manual_seed(0)
    ids = torch.randint(0, model.spec.vocab_size, (1, args.n_tokens), device=model.device)

    # warm-up
    StreamingRunner(model, chunk_size=args.chunk_size).prefill(ids[:, :256])

    rows = [bench_one(model, ids, "full", None, args.chunk_size, args.n_decode)]
    base = rows[0]
    for b in [int(x) for x in args.budgets.split(",")]:
        for p in POLICY_NAMES:
            rows.append(bench_one(model, ids, p, b, args.chunk_size, args.n_decode))
    weights_bytes = sum(p.numel() * p.element_size() for p in model.hf.parameters())
    for r in rows:
        r["rel_prefill_time"] = r["prefill_s"] / base["prefill_s"]
        r["rel_decode_time"] = r["decode_s_per_token"] / base["decode_s_per_token"]
        r["rel_kv_memory"] = r["kv_bytes_final"] / base["kv_bytes_final"]
        if torch.cuda.is_available():
            r["rel_peak_mem_excl_weights"] = ((r["peak_mem_bytes"] - weights_bytes) /
                                              max(1, base["peak_mem_bytes"] - weights_bytes))
        else:
            r["rel_peak_mem_excl_weights"] = None  # torch only tracks peak memory on CUDA
        pk = r["rel_peak_mem_excl_weights"]
        print(f"{r['policy']:10s} budget={str(r['budget']):6s} prefill x{r['rel_prefill_time']:.2f} "
              f"decode/tok x{r['rel_decode_time']:.2f} KV mem x{r['rel_kv_memory']:.3f} "
              f"peak(excl. weights) {'x%.2f' % pk if pk is not None else 'n/a (CUDA only)'}")

    save_json({"hardware": hw, "model": args.model, "device": str(model.device), "n_tokens": args.n_tokens, "n_decode": args.n_decode,
               "chunk_size": args.chunk_size, "dtype": str(model.dtype), "rows": rows},
              "results/benchmark.json")

    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{r['policy']}\n{r['budget'] or 'inf'}" for r in rows]
    axes[0].bar(labels, [r["rel_decode_time"] for r in rows]); axes[0].set_ylabel("decode time / token (relative to full)")
    axes[1].bar(labels, [r["rel_kv_memory"] for r in rows]); axes[1].set_ylabel("final KV memory (relative to full)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7); ax.axhline(1, ls="--", c="gray")
    fig.suptitle(f"N={args.n_tokens} tokens, {hw.get('gpu', hw['cpu'])}")
    fig.savefig("plots/benchmark.png", dpi=130, bbox_inches="tight")
    print("saved plots/benchmark.png")


if __name__ == "__main__":
    main()
