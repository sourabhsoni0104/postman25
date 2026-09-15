import pytest
import torch
from kvcache.policies import make_policy
from kvcache.streaming import StreamingRunner
from kvcache.rope import apply_rope
from kvcache.utils import ByteTokenizer
from scripts.eval_niah import build_prompt, needle_retention
from scripts.eval_ppl import run_ppl
from data.get_data import strip_gutenberg_header

@pytest.mark.parametrize("policy", ["sliding", "streaming", "h2o"])
@pytest.mark.parametrize("chunk", [1, 7, 32])
def test_budget_during_attention(model, ids, policy, chunk):
    runner = StreamingRunner(model, make_policy(policy), 24, chunk_size=chunk)
    sizes = []
    runner.prefill(ids, lambda li, probs: sizes.append(probs.shape[-1]))
    assert max(sizes) <= 24
    assert runner.n_evictions > 0
    assert runner.peak_kv_bytes <= 2*3*2*24*16*4
    if policy == "streaming":
        assert all(lc.pos[0, :4].tolist() == [0, 1, 2, 3] for lc in runner.cache.layers)
    if policy == "h2o":
        # Heavy hitters must survive chunks as large as the budget.
        oldest_recent = runner.cache.n_seen - 24
        assert all((lc.pos < oldest_recent).any() for lc in runner.cache.layers)

def test_layer_budgets(model, ids):
    runner = StreamingRunner(model, make_policy("h2o"), layer_budgets=[12, 24, 36], chunk_size=4)
    runner.prefill(ids)
    assert [lc.T for lc in runner.cache.layers] == [12, 24, 36]

def test_rope_rerotation_oracle():
    torch.manual_seed(2)
    raw = torch.randn(1, 2, 12, 16)
    old = torch.arange(12)
    keep = torch.tensor([0, 1, 7, 9, 11])
    new = torch.arange(5)
    rotated = apply_rope(raw, old, 10000)[:, :, keep]
    rerotated = apply_rope(rotated, new-old[keep], 10000)
    direct = apply_rope(raw[:, :, keep], new, 10000)
    torch.testing.assert_close(rerotated, direct, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(rotated, direct)

def test_nll_alignment_and_common_region(model, ids):
    with torch.no_grad():
        ref = model.hf(ids).logits[0, :-1].float()
        expected = torch.nn.functional.cross_entropy(ref, ids[0, 1:], reduction="none")
    runner = StreamingRunner(model, chunk_size=7)
    torch.testing.assert_close(runner.feed_nll(ids), expected, atol=1e-5, rtol=1e-5)
    a = run_ppl(model, ids, "sliding", 24, chunk_size=8, eval_start=48)
    b = run_ppl(model, ids, "full", None, chunk_size=8, eval_start=48)
    assert a["n_eval"] == b["n_eval"] == 47
    assert a["first_eviction_seen"] == 24

def test_prompt_length_and_retention(model):
    tok = ByteTokenizer()
    ids, start, end = build_prompt(tok, 512, .9, "12345", "cpu")
    assert ids.shape == (1, 512)
    runner = StreamingRunner(model, chunk_size=64)
    runner.prefill(ids)
    assert needle_retention(runner.cache, start, end)["complete_in_all_heads"]
    with pytest.raises(ValueError):
        build_prompt(tok, 10, .5, "12345", "cpu")

def test_invalid_configuration(model):
    with pytest.raises(ValueError):
        StreamingRunner(model, make_policy("streaming"), budget=4)
    with pytest.raises(ValueError):
        StreamingRunner(model, make_policy("h2o"), budget=0)

def test_gutenberg_footer():
    assert strip_gutenberg_header("header\n*** START OF BOOK ***\nbody\n*** END OF BOOK ***\nlicense") == "body"
