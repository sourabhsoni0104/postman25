# Measured results

Model: `Qwen/Qwen2.5-0.5B-Instruct`; revision `7ae557604adf67be50417f59c2c2f167def9a775`.
Device: mps:0; dtype: torch.float32; chunk size: 32.
Hardware: macOS-26.6.2-arm64-arm-64bit; PyTorch 2.8.0.

## Attention

| Input | Mean mass on first four tokens |
|---|---:|
| Natural text | 30.47% |
| Random IDs (including prefix) | 29.78% |
| Shuffled text | 23.01% |
| Changed first four tokens | 30.40% |
| Causal uniform control | 0.70% |

On natural text, 90% of attention mass uses 11.10% of available keys on average.
See [per-head heatmaps](../plots/sink_share_heatmap.png) and [concentration](../plots/attention_concentration.png).

## Quality and memory

PPL pools 2046 predictions from 2 documents; every run starts at query index 2048.
NIAH uses 2048 prompt tokens, depths [0.1, 0.5, 0.9], and 2 paired keys per depth.
Cells show perplexity / retrieval accuracy. Memory counts K and V only, in MiB.

| Budget | KV MiB | Sliding | Sink-aware | Heavy-hitter |
|---:|---:|---:|---:|---:|
| Full (3072 doc tokens) | 72.00 | 17.686 / 100% | same | same |
| 128 | 3.00 | 157.430 / 0% | 21.833 / 0% | 21.700 / 0% |
| 256 | 6.00 | 110.005 / 33% | 20.611 / 33% | 20.091 / 0% |
| 512 | 12.00 | 81.188 / 0% | 19.696 / 33% | 18.676 / 33% |
| 1024 | 24.00 | 68.586 / 17% | 18.941 / 67% | 18.063 / 33% |
| 2048 | 48.00 | 355.025 / 100% | 17.954 / 100% | 17.729 / 100% |

The full-cache memory differs between workloads: the document reference above uses
3072 tokens; retrieval uses 2048 prompt tokens plus generated tokens.
Every trial, generated answer, confidence interval, and per-layer/head needle-retention fraction is saved in curve.json.
Intervals are descriptive Wilson intervals over a small, correlated synthetic sample, not population guarantees.

## RoPE: needles verified present in every layer and KV head

| Mode | Post-eviction PPL | Retrieval | All needles retained before decoding |
|---|---:|---:|---|
| recompute | 16.704 | 100% | True |
| absolute | 16.711 | 100% | True |
| stale | 275.001 | 0% | True |

Full-cache retrieval on the same ablation trials: 100%.
These are measured outcomes; equal retrieval scores do not prove the stale implementation is correct.
The independent rotation-identity regression test verifies the positional correction algebraically.

## Benchmark

| Policy | Budget | Decode time relative to full | KV memory relative to full |
|---|---:|---:|---:|
| full | full | 1.000 | 1.000 |
| sliding | 128 | 0.441 | 0.031 |
| streaming | 128 | 0.430 | 0.031 |
| h2o | 128 | 0.499 | 0.031 |
| sliding | 512 | 0.444 | 0.124 |
| streaming | 512 | 0.514 | 0.124 |
| h2o | 512 | 0.559 | 0.124 |
| sliding | 2048 | 0.809 | 0.496 |
| streaming | 2048 | 0.593 | 0.496 |
| h2o | 2048 | 0.582 | 0.496 |

Single warmed-up timing run, synthetic repeated decode token; timings are descriptive.
GPU/CPU workspaces, weights, and allocator reservations are not KV memory.
Peak process allocation is unavailable on MPS/CPU and is recorded as null.
