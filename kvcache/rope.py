import torch


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    D = x.shape[-1]
    cos, sin = rope_cos_sin(positions, D, theta, x.dtype)
    if positions.dim() == 1:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[None], sin[None]
    return x * cos + rotate_half(x) * sin
