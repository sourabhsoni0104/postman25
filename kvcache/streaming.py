"""Bounded inference: reserve space BEFORE attention so cached T <= budget.

Chunks reserve their entire space before the first query. Use chunk_size=1 for
online eviction. Budgets include new keys, but exclude attention workspaces.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .policies import NoEviction


class StreamingRunner:
    def __init__(self, model, policy=None, budget=None, rope_mode="recompute", chunk_size=32,
                 layer_budgets=None):
        if chunk_size < 1 or (budget is not None and budget < 1):
            raise ValueError("chunk_size and budget must be positive")
        self.model, self.policy = model, policy or NoEviction()
        self.budget, self.rope_mode, self.chunk_size = budget, rope_mode, chunk_size
        self.layer_budgets = layer_budgets or [budget] * model.spec.n_layers
        if len(self.layer_budgets) != model.spec.n_layers:
            raise ValueError("one budget is required per layer")
        if any(b is not None and b < 1 for b in self.layer_budgets):
            raise ValueError("layer budgets must be positive")
        if not isinstance(self.policy, NoEviction):
            if any(b is None for b in self.layer_budgets):
                raise ValueError("an eviction policy requires a budget for every layer")
            if any(b <= getattr(self.policy, "n_sink", 0) for b in self.layer_budgets):
                raise ValueError("budget must leave space beyond the protected sinks")
        self.reset()

    def reset(self):
        self.cache = self.model.new_cache(self.rope_mode)
        self.n_evictions = 0
        self.first_eviction_seen = None
        self.peak_kv_bytes = 0
        self.peak_cache_tokens = 0

    def _chunk_size(self, requested=None):
        cs = self.chunk_size if requested is None else requested
        if cs < 1:
            raise ValueError("chunk_size must be positive")
        if not isinstance(self.policy, NoEviction):
            cs = min(cs, min(self.layer_budgets) - getattr(self.policy, "n_sink", 0))
        return cs

    def _reserve(self, incoming):
        changed = False
        if isinstance(self.policy, NoEviction):
            return
        for lc, b in zip(self.cache.layers, self.layer_budgets):
            if lc.T + incoming > b:
                lc.evict(self.policy.keep_indices(lc, b - incoming))
                changed = True
        if changed:
            self.n_evictions += 1
            if self.first_eviction_seen is None:
                self.first_eviction_seen = self.cache.n_seen

    def _step(self, ids, attn_callback=None, last_only=False):
        self._reserve(ids.shape[1])
        logits = self.model.forward(ids, self.cache, attn_callback, return_last_only=last_only)
        self.peak_kv_bytes = max(self.peak_kv_bytes, self.cache.memory_bytes())
        self.peak_cache_tokens = max(self.peak_cache_tokens, max(lc.T for lc in self.cache.layers))
        return logits

    @torch.no_grad()
    def feed(self, ids, attn_callback=None, chunk_size=None):
        """Diagnostic API retaining logits; use prefill/feed_nll for long inputs."""
        cs = self._chunk_size(chunk_size)
        return [self._step(ids[:, i:i + cs], attn_callback) for i in range(0, ids.shape[1], cs)]

    @torch.no_grad()
    def prefill(self, ids, attn_callback=None):
        if ids.shape[1] == 0:
            raise ValueError("prompt must not be empty")
        cs = self._chunk_size()
        for i in range(0, ids.shape[1], cs):
            logits = self._step(ids[:, i:i + cs], attn_callback, last_only=True)
        return logits

    @torch.no_grad()
    def feed_nll(self, ids, chunk_size=None):
        if ids.shape[1] < 2:
            raise ValueError("perplexity requires at least two tokens")
        cs = self._chunk_size(chunk_size)
        nll = []
        for i in range(0, ids.shape[1], cs):
            chunk = ids[:, i:i + cs]
            logits = self._step(chunk)
            targets = ids[0, i + 1:i + chunk.shape[1] + 1]
            if targets.numel():
                nll.append(F.cross_entropy(logits[0, :len(targets)].float(), targets, reduction="none").cpu())
        return torch.cat(nll)

    @torch.no_grad()
    def decode(self, logits, max_new_tokens=16):
        out = []
        for i in range(max_new_tokens):
            next_id = int(logits[0, -1].argmax())
            out.append(next_id)
            if i + 1 < max_new_tokens:
                logits = self._step(torch.tensor([[next_id]], device=self.model.device), last_only=True)
        return out

    def generate(self, prompt_ids, max_new_tokens=16):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        return self.decode(self.prefill(prompt_ids), max_new_tokens)
