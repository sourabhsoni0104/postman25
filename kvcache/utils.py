from __future__ import annotations
import json
import os
import platform
import subprocess
import time

import torch

from .model import ManualQwen2, load_hf

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device_dtype(device: str | None = None, dtype: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if dtype is None:
        dtype = "float32"  # validated default; opt into lower precision explicitly
    return device, getattr(torch, dtype)


class _Encoded:
    def __init__(self, ids):
        self.input_ids = ids


class ByteTokenizer:
    """Fake tokenizer for the tiny random model (byte-level, vocab 512)."""

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([list(text.encode("utf-8"))], dtype=torch.long)
        return _Encoded(ids if return_tensors == "pt" else ids[0].tolist())

    def decode(self, ids):
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="ignore")


def build_model(model_name: str = DEFAULT_MODEL, device=None, dtype=None):
    """Returns (ManualQwen2, tokenizer, hf_model). model_name='tiny' gives a random
    3-layer model + byte tokenizer for smoke-testing scripts on any machine."""
    if model_name == "tiny":
        model, hf = tiny_model(device=device or "cpu")
        return model, ByteTokenizer(), hf
    device, dtype = pick_device_dtype(device, dtype)
    hf, tok = load_hf(model_name, device, dtype)
    return ManualQwen2(hf), tok, hf


def tiny_model(seed: int = 0, device: str = "cpu"):
    """Random-weight Qwen2 with the same architecture family; used by the test
    harness so it runs anywhere in seconds (no download)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=8192, tie_word_embeddings=True,
                      attn_implementation="eager")
    hf = Qwen2ForCausalLM(cfg).to(device).eval()
    return ManualQwen2(hf), hf


def hardware_report() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or "",
        "cpu_count": os.cpu_count(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "torch_threads": torch.get_num_threads(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
        info["cuda_version"] = torch.version.cuda
    return info


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.seconds = time.perf_counter() - self.t0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes() -> int:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return None  # No equivalent CPU/MPS peak allocator counter.


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    print(f"saved {path}")


def load_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Start known public-domain books at prose, excluding contents/front matter.
    starts = {"pride_and_prejudice.txt": "It is a truth universally acknowledged",
              "frankenstein.txt": "You will rejoice to hear",
              "moby_dick.txt": "Call me Ishmael."}
    marker = starts.get(os.path.basename(path))
    if marker is not None:
        if marker not in text:
            raise ValueError(f"expected prose start missing from {path}")
        text = text[text.index(marker):]
    return text
