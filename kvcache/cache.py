"""A per-layer KV cache that supports (per-head) eviction and three ways of
handling RoPE after eviction.

rope_mode:
  "recompute" (correct)  : cache stores PRE-RoPE keys; at every step the whole
                           compacted cache is rotated with positions 0..T-1 and the
                           new queries get positions T-L..T-1. Relative distances are
                           always consistent and never exceed the cache budget.
  "absolute"  (also ok)  : cache stores PRE-RoPE keys; keys/queries rotated with
                           their ORIGINAL token index. Relative distances are the
                           true ones (with gaps). Positions grow unboundedly, so it
                           eventually leaves the model's trained position range.
  "stale"     (the bug)  : cache stores POST-RoPE keys, rotated with the position
                           = cache length at insertion time, and the query uses
                           position = current cache length. This is what you get if
                           you slice an HF DynamicCache and keep generating: after
                           any eviction the stored rotations no longer match.
"""
from __future__ import annotations
import torch

ROPE_MODES = ("recompute", "absolute", "stale")


class LayerKV:
    """Batch size 1 only (eviction with per-head index sets is much simpler that way).

    k, v : [1, Hkv, T, D]
    pos  : [Hkv, T] original token index of each cached entry (for analysis / 'absolute' mode)
    score: [Hkv, T] fp32 accumulated attention mass received by each entry (for H2O)
    """

    def __init__(self):
        self.k = None
        self.v = None
        self.pos = None
        self.score = None

    @property
    def T(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor, pos_new: torch.Tensor):
        Hkv, L = k_new.shape[1], k_new.shape[2]
        pos_new = pos_new.view(1, L).expand(Hkv, L).to(k_new.device)
        score_new = torch.zeros(Hkv, L, device=k_new.device, dtype=torch.float32)
        if self.k is None:
            self.k, self.v, self.pos, self.score = k_new, v_new, pos_new, score_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.pos = torch.cat([self.pos, pos_new], dim=1)
            self.score = torch.cat([self.score, score_new], dim=1)

    def add_scores(self, probs: torch.Tensor, n_kv_heads: int):
        """probs: [1, H, L, T] attention probabilities from this step.
        Accumulate per KV head (sum over the query heads sharing it and over queries)."""
        H, L, T = probs.shape[1], probs.shape[2], probs.shape[3]
        G = H // n_kv_heads
        s = probs[0].float().view(n_kv_heads, G, L, T).sum(dim=(1, 2))  # [Hkv, T]
        self.score += s

    def evict(self, keep_idx: torch.Tensor):
        """keep_idx: LongTensor [Hkv, K] of cache indices to keep, sorted ascending per head."""
        keep_idx = keep_idx.to(self.k.device)
        Hkv, K = keep_idx.shape
        D = self.k.shape[-1]
        g = keep_idx[None, :, :, None].expand(1, Hkv, K, D)
        self.k = self.k.gather(2, g)
        self.v = self.v.gather(2, g)
        self.pos = self.pos.gather(1, keep_idx)
        self.score = self.score.gather(1, keep_idx)

    def memory_bytes(self) -> int:
        if self.k is None:
            return 0
        return self.k.numel() * self.k.element_size() * 2


class KVCache:
    def __init__(self, n_layers: int, rope_mode: str = "recompute"):
        assert rope_mode in ROPE_MODES, rope_mode
        self.layers = [LayerKV() for _ in range(n_layers)]
        self.rope_mode = rope_mode
        self.n_seen = 0  # total tokens fed so far (original position counter)

    def reset(self):
        for l in self.layers:
            l.__init__()
        self.n_seen = 0

    @property
    def T(self) -> int:
        return self.layers[0].T

    def memory_bytes(self) -> int:
        return sum(l.memory_bytes() for l in self.layers)

    def metadata_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for lc in self.layers
                   for t in (lc.pos, lc.score) if t is not None)
