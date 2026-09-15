"""Render measured results and extra diagnostic plots; never invent missing data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT / "results" / name).read_text())

def main():
    curve, attention, rope = [read(f"{n}.json") for n in ("curve", "attention_stats", "rope_ablation")]
    if not curve["complete"]:
        raise ValueError("budget sweep is incomplete")
    rows = curve["rows"]
    full = curve["full"]
    text = ["# Measured results", "", f"Model: `{curve['model']}`; revision `{curve['model_revision']}`.",
            f"Device: {curve['device']}; dtype: {curve['dtype']}; chunk size: {curve['chunk_size']}.",
            f"Hardware: {curve['hardware']['platform']}; PyTorch {curve['hardware']['torch']}.", "",
            "## Attention", "", "| Input | Mean mass on first four tokens |", "|---|---:|"]
    for label, key in [("Natural text", "text"), ("Random IDs (including prefix)", "random"),
                       ("Shuffled text", "shuffled"), ("Changed first four tokens", "changed_prefix")]:
        text.append(f"| {label} | {attention['sink_share_mean_'+key]:.2%} |")
    text += [f"| Causal uniform control | {attention['causal_uniform_sink_share']:.2%} |", "",
             f"On natural text, 90% of attention mass uses {attention['top90_fraction_mean_text']:.2%} of available keys on average.",
             "See [per-head heatmaps](../plots/sink_share_heatmap.png) and [concentration](../plots/attention_concentration.png).", "",
             "## Quality and memory", "",
             f"PPL pools {full['n_eval']} predictions from {len(full['documents'])} documents; every run starts at query index {curve['eval_start']}.",
             f"NIAH uses {curve['ctx']} prompt tokens, depths {curve['depths']}, and {curve['n_trials']} paired keys per depth.",
             "Cells show perplexity / retrieval accuracy. Memory counts K and V only, in MiB.", "",
             "| Budget | KV MiB | Sliding | Sink-aware | Heavy-hitter |", "|---:|---:|---:|---:|---:|",
             f"| Full ({curve['n_tokens']} doc tokens) | {full['kv_mb']:.2f} | {full['ppl_eval']:.3f} / {full['niah_acc']:.0%} | same | same |"]
    for i, row in enumerate(rows["sliding"]):
        values = [f"{rows[p][i]['ppl_eval']:.3f} / {rows[p][i]['niah_acc']:.0%}" for p in rows]
        text.append(f"| {row['budget']} | {row['kv_mb']:.2f} | " + " | ".join(values) + " |")
    text += ["", "The full-cache memory differs between workloads: the document reference above uses",
             f"{curve['n_tokens']} tokens; retrieval uses {curve['ctx']} prompt tokens plus generated tokens.",
             "Every trial, generated answer, confidence interval, and per-layer/head needle-retention fraction is saved in curve.json.",
             "Intervals are descriptive Wilson intervals over a small, correlated synthetic sample, not population guarantees.", "",
             "## RoPE: needles verified present in every layer and KV head", "",
             "| Mode | Post-eviction PPL | Retrieval | All needles retained before decoding |",
             "|---|---:|---:|---|"]
    for mode, r in rope["results"].items():
        text.append(f"| {mode} | {r['ppl_post_budget']:.3f} | {r['niah_acc']:.0%} | {r['all_needles_retained']} |")
    text += ["", f"Full-cache retrieval on the same ablation trials: {rope['full_niah']['accuracy']:.0%}.",
             "These are measured outcomes; equal retrieval scores do not prove the stale implementation is correct.",
             "The independent rotation-identity regression test verifies the positional correction algebraically.", ""]
    benchmark = read("benchmark.json")
    text += ["## Benchmark", "", "| Policy | Budget | Decode time relative to full | KV memory relative to full |",
             "|---|---:|---:|---:|"]
    for r in benchmark["rows"]:
        text.append(f"| {r['policy']} | {r['budget'] or 'full'} | {r['rel_decode_time']:.3f} | {r['rel_kv_memory']:.3f} |")
    text += ["", "Single warmed-up timing run, synthetic repeated decode token; timings are descriptive.",
             "GPU/CPU workspaces, weights, and allocator reservations are not KV memory.",
             "Peak process allocation is unavailable on MPS/CPU and is recorded as null.", ""]
    (ROOT / "results" / "REPORT.md").write_text("\n".join(text))

    # Loss over time makes the transition after first eviction visible.
    docs = list(full["documents"])
    fig, axes = plt.subplots(len(docs), 1, figsize=(11, 3.5*len(docs)), squeeze=False)
    for ax, doc in zip(axes[:, 0], docs):
        selected = [("full", full["documents"][doc])]
        selected += [(p, rows[p][0]["documents"][doc]) for p in rows]
        for label, r in selected:
            losses = np.array(r["nll_curve"])
            width = min(128, len(losses))
            smooth = np.convolve(losses, np.ones(width)/width, mode="valid")
            ax.plot(np.arange(width, len(losses)+1), np.exp(smooth), label=label)
        ax.axvline(rows["sliding"][0]["documents"][doc]["first_eviction_seen"], c="gray", ls="--", label="first eviction")
        ax.set(title=Path(doc).stem, xlabel="target token position", ylabel="rolling perplexity", yscale="log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "perplexity_over_time.png", dpi=140)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, policy in zip(axes, rows):
        matrix = [[r["niah_per_depth"][str(d)] for d in curve["depths"]] for r in rows[policy]]
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(curve["depths"])), labels=curve["depths"])
        ax.set_yticks(range(len(rows[policy])), labels=[r["budget"] for r in rows[policy]])
        ax.set(title=policy, xlabel="needle depth", ylabel="cache budget")
        for i, line in enumerate(matrix):
            for j, value in enumerate(line):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value < .5 else "black")
    fig.colorbar(im, ax=axes, label="retrieval accuracy", fraction=.03)
    fig.savefig(ROOT / "plots" / "retrieval_by_depth.png", dpi=140, bbox_inches="tight")
    print("Wrote results/REPORT.md and diagnostic plots")

if __name__ == "__main__":
    main()
