
from __future__ import annotations
import torch

ROPE_MODES = ("recompute", "absolute", "stale")


class LayerKV:

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
        H, L, T = probs.shape[1], probs.shape[2], probs.shape[3]
        G = H // n_kv_heads
        s = probs[0].float().view(n_kv_heads, G, L, T).sum(dim=(1, 2))  # [Hkv, T]
        self.score += s

    def evict(self, keep_idx: torch.Tensor):
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
        self.n_seen = 0 

    def reset(self):
        for l in self.layers:
            l.__init__()
        self.n_seen = 0

    @property
    def T(self) -> int:
        return self.layers[0].T

    def memory_bytes(self) -> int:
        return sum(l.memory_bytes() for l in self.layers)
