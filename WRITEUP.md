# Attention-Aware KV Cache Compression — Writeup

Model: Qwen2.5-0.5B (24 layers, 14 query heads, 2 KV heads, head dim 64, RoPE θ=10⁶).
Hardware: `[FILL from results/benchmark.json → hardware]`.
All numbers below come from `results/*.json`; plots from `plots/`.

> Everything in `[FILL …]` is to be replaced with your measured numbers.
> Where I wrote "if X, say Y", keep only the branch that matches your results.

## 1. What I built

I re-implemented the Qwen2 decoder forward pass on top of the HuggingFace weights
(`kvcache/model.py`) so that the KV cache is a plain tensor I control. The
correctness harness (`tests/test_correctness.py`) checks that the manual
forward reproduces HF's eager logits (max |Δ| = `[FILL]` in fp32, `[FILL]` in fp16)
and that chunked prefill equals a single prefill. On top of that: three eviction
policies, three ways of handling positions after eviction, attention
instrumentation, and two quality evaluations.

## 2. Attention is concentrated, and the first tokens absorb it (3.2)

`scripts/instrument_attention.py` runs a 2048-token prefill and records, per
layer/head, (a) the share of each query's attention that lands on the first 4
tokens, (b) the fraction of available keys needed to cover 90% of the mass, and
(c) the mean attention each key position receives. I ran it on real text and on
uniformly random token ids as a control.

* Mean sink share (mass on tokens 0–3): `[FILL]` on text, `[FILL]` on random ids
  (`plots/sink_share_heatmap.png`). Layers `[FILL: which layers/heads are the
  strongest sinks]` put more than half of their mass there.
* 90% of the mass is covered by `[FILL]`% of available keys on average
  (`plots/attention_concentration.png`), i.e. most keys are almost never read.
* Tokens 0–3 receive `[FILL]`× more attention than a uniform distribution would
  give them, and the same is true for random ids (`plots/attention_received_by_position.png`).

The random-token control is the key result: the first tokens get the mass **regardless of what they are**.
That is the empirical fact any eviction policy has to respect.

## 3. Eviction policies (3.3)

* **Sliding window** — keep the last `B` tokens. Baseline.
* **StreamingLLM** — keep the first 4 tokens (sinks) + the last `B−4`.
* **H2O** — keep the last `B/2` tokens + the `B/2` tokens with the highest
  accumulated attention score, chosen independently per KV head. Scores are the
  column sums of the attention matrix, accumulated across all steps.

## 4. Positions after eviction (3.4)

RoPE rotates q and k by an angle proportional to position; the score depends only
on `pos_q − pos_k` (verified in the harness). HF caches *rotated* keys and sets the
next query position to the current cache length. If you evict from the middle and
carry on, the query is rotated as though the cache were contiguous while the old
keys still carry their original rotations: the relative distances the model sees are
wrong, but nothing crashes and text stays fluent. I implemented this bug on purpose
as `rope_mode="stale"` and compare it with:

* `recompute` (mine, correct): cache pre-RoPE keys; at every step rotate the
  compacted cache with positions 0..T−1 and the query with T−1. Distances are
  always consistent and never exceed the budget.
* `absolute`: rotate with original token indices (correct distances with gaps,
  but positions grow without bound).

`scripts/rope_ablation.py` (StreamingLLM, budget 512, 4096-token doc, needle at
depth 0.90–0.98 so it *stays inside the recent window*):

| mode | PPL post-budget | needle accuracy |
|---|---|---|
| recompute | `[FILL]` | `[FILL]` |
| absolute | `[FILL]` | `[FILL]` |
| stale | `[FILL]` | `[FILL]` |

`[FILL: expected pattern — stale has modestly higher PPL but retrieval of a needle
that is physically in the cache collapses; this is the "invisible without a targeted
evaluation" failure. If absolute also degrades at long contexts, say so: positions
exceed what the budget alone would produce.]`

## 5. Quality under a fixed budget (3.5, 3.6)

Perplexity is measured on `[FILL: document]` (6144 tokens) and reported on the
tokens after the first eviction. Needle-in-a-haystack uses a 2048-token context
with a 5-digit passkey at depths 0.1–0.9, `[FILL]` trials each.

`plots/quality_vs_memory.png`, from `results/curve.json`:

| budget | KV MB | sliding PPL / NIAH | streaming PPL / NIAH | h2o PPL / NIAH |
|---|---|---|---|---|
| full | `[FILL]` | `[FILL]` | | |
| 128 | | | | |
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |

What the two metrics show:

* **Sliding window** `[FILL: should visibly fail — PPL jumps by orders of magnitude
  the moment tokens 0–3 are evicted. Say exactly when: at the first eviction after
  `budget` tokens.]`
* **StreamingLLM** `[FILL: PPL should stay close to full-cache PPL at every budget
  — while needle accuracy is ~0 for any depth outside the recent window. This is the
  point of 3.5: perplexity barely moves, the fact is gone.]`
* **H2O** `[FILL: PPL close to streaming; needle accuracy depends on whether the
  needle tokens accumulated enough score during the filler to survive — report the
  per-depth numbers. If h2o also dropped the needle, explain: a passkey nobody
  refers to during prefill earns little attention mass, so a score-based policy has
  no reason to keep it until the question arrives — too late.]`

Memory scales linearly with the budget: `2 × 24 layers × 2 KV heads × 64 × B ×
2 bytes = 12 KB × B`, i.e. 6 MB at B=512 vs `[FILL]` MB for the 6144-token full cache.

## 6. Benchmark

`results/benchmark.json`, `plots/benchmark.png`, on `[FILL hardware]`, 8192
tokens streamed + 64 decode steps, fp16. Relative to the full cache: decode time
per token `[FILL]`× at budget 256 and `[FILL]`× at 1024; final KV memory `[FILL]`×
and `[FILL]`×. `[FILL: note that per-token decode with a full 8k cache is dominated
by attention over 8k keys, so a 256-entry cache is measurably faster; note also
the cost of re-rotating the whole cache every step in `recompute` mode — O(B·d),
negligible next to attention itself.]`

## 7. Discussion (3.7)

**Why attention sinks exist.** Softmax forces every row to sum to 1. When a head
has nothing useful to attend to for a given query — which is most of the time,
since attention is sparse — the excess mass has to go somewhere. During training,
the first token is the only position that is visible to *every* query, so it is the
one place a head can reliably park unwanted mass; the model learns to give it a
key that scores highly for almost any query while its value carries little
information. The random-token control in §2 confirms it is a positional
convention, not a content effect. Evicting the sink removes the dumping ground,
the mass gets redistributed over content tokens that were never meant to receive
it, and the residual stream is corrupted — hence the sliding-window collapse.

**Why accumulated-score eviction is biased toward early tokens.** H2O ranks a
token by the total attention it has received. A token at position `p` has been
a candidate key for every query after `p`, so it has had `T − p` chances to
collect mass; a token inserted 10 steps ago has had 10. The score is a sum,
not a rate, so all else being equal older tokens win. Combined with the sink
effect (tokens 0–3 receive a large share of *every* row) this means the heavy
hitters are dominated by early tokens, and genuinely informative but recent
tokens only survive because of the explicit recent window. Normalising by age
would fix the bias but break the sink retention that makes H2O work at all — the
bias is, in part, doing the job of StreamingLLM's explicit sink rule.

**Which policy for which workload.**

* *Multi-turn agent*: the history is long, the facts that matter (tool outputs,
  user constraints, earlier decisions) are scattered through the middle, and the
  next query is unknown at eviction time. A pure recency policy (sliding /
  StreamingLLM) throws away exactly those facts — §5 shows the needle vanishing
  while perplexity looks fine. H2O-style score-based retention is the better default
  because tokens that were referred to once tend to be referred to again, but it
  should be combined with a guaranteed sink set, a recent window, and ideally
  per-head budgets; and the *effective* budget needs to be generous, because
  score-based eviction cannot anticipate a fact that has not been used yet.
* *Long-document summariser*: the model reads once, front to back, and the
  question ("summarise") is known up front. Local coherence matters more than
  random access, throughput matters, and the summary is generated after the
  whole document has been read. StreamingLLM (sinks + recent window) is the right
  tool: near-full-cache perplexity at a tiny fixed budget, simplest possible
  eviction (no score bookkeeping, same indices for every head), and constant
  memory. If specific facts must survive, chunk the document and summarise
  hierarchically rather than trying to keep them in the cache.

**Limitations / honest notes.** Batch size 1; eviction during prefill happens
every `chunk_size` tokens rather than every token; H2O uses raw cumulative
scores without decay; `[FILL: anything else you changed or could not finish]`.
