"""Correctness harness.

    python -m tests.test_correctness            # fast: random tiny Qwen2 (CPU, seconds)
    python -m tests.test_correctness --model Qwen/Qwen2.5-0.5B   # real model

Every check prints PASS/FAIL; the exit code is non-zero if anything fails.
The reference is HuggingFace's eager Qwen2 implementation.
"""
from __future__ import annotations
import argparse
import sys

import torch

from kvcache.cache import KVCache
from kvcache.policies import H2O, SlidingWindow, StreamingLLM
from kvcache.rope import apply_rope
from kvcache.streaming import StreamingRunner
from kvcache.utils import build_model, tiny_model

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    assert ok, f"{name}: {detail}"


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# --------------------------------------------------------------------------- tests
def test_rope_relative_property():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 1, 16)
    theta = 10000.0

    def score(pq, pk):
        qr = apply_rope(q, torch.tensor([pq]), theta)
        kr = apply_rope(k, torch.tensor([pk]), theta)
        return (qr * kr).sum(-1)

    d1 = maxdiff(score(10, 3), score(107, 100))
    d2 = maxdiff(score(10, 3), score(11, 3))  # different distance -> must differ
    check("rope: score depends only on q_pos - k_pos", d1 < 1e-4 and d2 > 1e-3, f"same-dist diff={d1:.2e}")


def test_manual_matches_hf(model, hf, ids, tol):
    ref = hf(ids).logits
    cache = model.new_cache("recompute")
    out = model.forward(ids, cache)
    d = maxdiff(out, ref)
    check(f"manual forward == HF eager (N={ids.shape[1]})", d < tol, f"max|diff|={d:.2e} tol={tol}")
    # 'stale' and 'absolute' must also equal HF when nothing has been evicted
    for mode in ("stale", "absolute"):
        d = maxdiff(model.forward(ids, model.new_cache(mode)), ref)
        check(f"rope_mode={mode} == HF when no eviction", d < tol, f"max|diff|={d:.2e}")


def test_chunked_equals_full(model, hf, ids, tol):
    ref = hf(ids).logits
    for cs in (1, 7, 64):
        r = StreamingRunner(model, chunk_size=cs)
        out = torch.cat(r.feed(ids), dim=1)
        d = maxdiff(out, ref)
        if model.dtype == torch.float32:
            check(f"chunked prefill (chunk={cs}) == full prefill", d < tol, f"max|diff|={d:.2e}")
        else:
            # Half-precision GEMV/GEMM kernels have different rounding even in HF.
            # Use an identically chunked HF cache reference to isolate our code.
            from transformers import DynamicCache
            hf_cache = DynamicCache()
            chunks = []
            for i in range(0, ids.shape[1], cs):
                chunks.append(hf(ids[:, i:i+cs], past_key_values=hf_cache, use_cache=True).logits)
            href = torch.cat(chunks, dim=1)
            matched = maxdiff(out, href)
            check(f"chunked prefill (chunk={cs}) == HF chunked", matched < tol,
                  f"matched diff={matched:.2e}; HF chunk/full drift={maxdiff(href, ref):.2e}; manual chunk/full={d:.2e}")


def test_h2o_scores_are_column_sums(model, ids):
    """cache.score must equal the attention mass each key received (per KV head)."""
    cache = model.new_cache("recompute")
    got = {}

    def cb(li, probs):
        Hkv = model.spec.n_kv_heads
        G = probs.shape[1] // Hkv
        got[li] = probs[0].view(Hkv, G, probs.shape[2], probs.shape[3]).sum((1, 2))

    model.forward(ids, cache, attn_callback=cb)
    ok = all(maxdiff(cache.layers[li].score, got[li]) < 1e-4 for li in got)
    check("h2o: accumulated score == column sums of attention probs", ok)


def test_policy_indices():
    lc = KVCache(1).layers[0]
    Hkv, T = 2, 20
    lc.k = torch.zeros(1, Hkv, T, 4); lc.v = torch.zeros_like(lc.k)
    lc.pos = torch.arange(T).expand(Hkv, T).clone()
    lc.score = torch.zeros(Hkv, T)
    lc.score[0, 5] = 10; lc.score[0, 9] = 9; lc.score[1, 2] = 10; lc.score[1, 12] = 9

    sw = SlidingWindow().keep_indices(lc, 8)
    check("sliding keeps last `budget` entries", sw[0].tolist() == list(range(12, 20)))

    st = StreamingLLM(4).keep_indices(lc, 8)
    check("streaming keeps 4 sinks + 4 most recent", st[0].tolist() == [0, 1, 2, 3, 16, 17, 18, 19])

    h = H2O(0.5).keep_indices(lc, 8)  # 4 recent + 4 heavy per head
    ok = (set([5, 9]) <= set(h[0].tolist())) and (set([2, 12]) <= set(h[1].tolist())) \
        and h[0].tolist()[-4:] == [16, 17, 18, 19] and h.shape == (2, 8) \
        and (h[0].diff() > 0).all() and (h[1].diff() > 0).all()
    check("h2o keeps per-head heavy hitters + recent, sorted", ok)

    lc.evict(h)
    check("evict(): tensors compacted to budget", lc.k.shape[2] == 8 and lc.pos.shape == (2, 8)
          and lc.pos[0].tolist() == h[0].tolist())


def test_budget_respected(model, ids):
    for pol in (SlidingWindow(), StreamingLLM(4), H2O(0.5)):
        r = StreamingRunner(model, pol, budget=24, chunk_size=8)
        r.feed(ids)
        ok = all(lc.T <= 24 for lc in r.cache.layers)
        check(f"budget respected after streaming ({pol.name})", ok, f"T={r.cache.T}")


def test_recompute_rope_after_eviction(model, ids):
    """After evicting middle tokens, the correct ('recompute') and buggy ('stale')
    modes must (a) agree with each other BEFORE eviction and (b) diverge AFTER.
    Also, 'recompute' and 'absolute' must produce identical attention when only the
    most recent tokens are kept (contiguous suffix => same relative distances)."""
    N = ids.shape[1]
    a, b = ids[:, : N // 2], ids[:, N // 2 :]
    outs = {}
    for mode in ("recompute", "stale"):
        cache = model.new_cache(mode)
        pre = model.forward(a, cache)
        for lc in cache.layers:
            lc.evict(StreamingLLM(4).keep_indices(lc, N // 4))
        post = model.forward(b, cache)
        outs[mode] = (pre, post)
    check("stale == recompute before any eviction", maxdiff(outs["stale"][0], outs["recompute"][0]) < 1e-4)
    check("stale != recompute after eviction (bug is real)", maxdiff(outs["stale"][1], outs["recompute"][1]) > 1e-3)

    # sliding window keeps a contiguous suffix -> 'absolute' and 'recompute' see identical distances
    outs = {}
    for mode in ("recompute", "absolute"):
        r = StreamingRunner(model, SlidingWindow(), budget=N // 4, rope_mode=mode, chunk_size=N // 2)
        r.feed(a)
        outs[mode] = r.feed(b)[0]
    diff = maxdiff(outs["recompute"], outs["absolute"])
    # Rotation shift identity is exact in real arithmetic; fp16 rotations round.
    tolerance = 1e-3 if model.dtype == torch.float32 else 5e-2
    check("recompute == absolute for contiguous-suffix eviction", diff < tolerance,
          f"max|diff|={diff:.2e}, tolerance={tolerance}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", help="'tiny' (random weights) or an HF model name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seq_len", type=int, default=96)
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.model == "tiny":
        model, hf = tiny_model(device=args.device or "cpu")
        tol = 1e-4
        ids = torch.randint(0, model.spec.vocab_size, (1, args.seq_len), device=model.device)
    else:
        model, tok, hf = build_model(args.model, args.device, args.dtype)
        tol = 1e-3 if model.dtype == torch.float32 else 5e-2  # fp16: HF eager also rounds
        text = "The quick brown fox jumps over the lazy dog. " * 40
        ids = tok(text, return_tensors="pt").input_ids[:, : args.seq_len].to(model.device)
    print(f"model={args.model} device={model.device} dtype={model.dtype} N={ids.shape[1]}\n")

    with torch.no_grad():
        test_rope_relative_property()
        test_manual_matches_hf(model, hf, ids, tol)
        test_chunked_equals_full(model, hf, ids, tol)
        test_h2o_scores_are_column_sums(model, ids)
        test_policy_indices()
        test_budget_respected(model, ids)
        test_recompute_rope_after_eviction(model, ids)

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    from kvcache.utils import save_json
    save_json({"model": args.model, "device": str(model.device), "dtype": str(model.dtype),
               "checks": [{"name": name, "passed": bool(ok)} for name, ok in RESULTS]},
              "results/correctness_" + ("tiny" if args.model == "tiny" else str(model.dtype).split(".")[-1]) + ".json")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
