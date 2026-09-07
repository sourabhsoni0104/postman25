# Attention-Aware KV Cache Compression

Postman AI/ML recruitment task 3. A small open-weight model (Qwen2.5-0.5B) is run
through a **hand-written forward pass** so that we fully own the KV cache: we
instrument attention, evict cache entries with three policies, handle RoPE
correctly after eviction, and measure what each policy costs in quality and memory.

## Layout

| path | what |
|---|---|
| `kvcache/model.py` | manual Qwen2 forward on HF weights; exposes attention probs (3.1) |
| `kvcache/rope.py` | hand-written RoPE, applied *at attention time* (3.4) |
| `kvcache/cache.py` | per-layer KV cache: pre-RoPE keys, per-head eviction, 3 RoPE modes (3.4) |
| `kvcache/policies.py` | sliding window / StreamingLLM / H2O (3.3) |
| `kvcache/streaming.py` | chunked prefill + decode loop with eviction between chunks |
| `tests/test_correctness.py` | **correctness harness** vs HF eager reference, prints PASS/FAIL |
| `scripts/instrument_attention.py` | attention statistics + plots (3.2) |
| `scripts/eval_ppl.py` | perplexity under a budget (3.5) |
| `scripts/eval_niah.py` | needle-in-a-haystack under a budget (3.5) |
| `scripts/rope_ablation.py` | recompute vs absolute vs stale positions (3.4) |
| `scripts/run_curve.py` | quality-vs-memory curve, all policies × budgets (3.6) |
| `scripts/benchmark.py` | **benchmark**: time + peak memory, reports hardware, relative numbers |
| `WRITEUP.md` | writeup (3.7) |
| `plots/`, `results/` | generated figures and JSON |

## Setup

```bash
pip install -r requirements.txt
python data/get_data.py          # public-domain long documents for perplexity
```

Runs on a free Colab T4 (fp16) in well under an hour total, or on a laptop CPU
(fp32, slower; reduce `--n_tokens`).

## Correctness harness

```bash
python -m tests.test_correctness                       # random tiny model, seconds, no download
python -m tests.test_correctness --model Qwen/Qwen2.5-0.5B
```

Checks: RoPE relative-position property; manual forward == HF eager logits;
chunked prefill == single prefill; H2O accumulated scores == attention column
sums; each policy keeps exactly the intended indices; budget is respected after
eviction; `stale` RoPE equals `recompute` before eviction and diverges after it.
Exit code is non-zero on any failure.

## Reproducing the results

```bash
# 3.2 attention instrumentation (~1 min on T4)
python -m scripts.instrument_attention --n_tokens 2048

# 3.5 single runs
python -m scripts.eval_ppl  --policy sliding   --budget 512
python -m scripts.eval_niah --policy streaming --budget 512 --ctx 2048

# 3.4 position handling ablation
python -m scripts.rope_ablation --policy streaming --budget 512 --ctx 2048

# 3.6 quality vs memory, all policies × budgets (~20-30 min on T4)
python -m scripts.run_curve --budgets 128,256,512,1024,2048 --n_tokens 6144 --ctx 2048

# benchmark (reports hardware; relative numbers only)
python -m scripts.benchmark --n_tokens 8192 --budgets 256,1024
```

`bash run_all.sh` runs everything in order. Every script accepts `--model tiny`
to smoke-test on a random 3-layer model without downloading anything.

## Design notes

* **Why a manual forward?** HF caches keys *after* RoPE and its `DynamicCache`
  assumes a contiguous history. To evict from the middle correctly we need
  pre-RoPE keys and control of position ids; re-implementing the 24-layer decoder
  (~100 lines) was simpler and more robust than monkey-patching.
* **Eviction granularity.** Tokens are fed in chunks (`--chunk_size`, default 32);
  eviction runs after every chunk during prefill and after every token during
  decode. With chunk size 1 this is exactly the per-token scheme of the papers.
* **Per-head eviction.** H2O keeps different tokens per KV head; the cache stores
  a `[Hkv, T]` index of original positions so each head can be tracked.
* **Batch size 1 only.**

## References

* Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), 2023.
* Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*, 2023.
* Mohtashami & Jaggi, *Landmark Attention* (passkey retrieval task), 2023.
