"""Shared Modal pieces for Brian2 batches (2026-10-01): the image recipe, the raw-data Volume, and the compact
result format.

Data policy (Jasper, 2026-10-01): raw full-resolution results live on the Modal Volume `developing-mind-raw`
(1 TiB/month free, then $0.09/GiB-month); git and the laptop get a COMPACT file. Compact = everything the analyses
use, at resolutions that keep them exact:
  weight_blocks    weights averaged over 10 s blocks, shape (n_synapses, ceil(T/10)), rounded to 4 decimals.
                   Every analysis averages weights over windows aligned to multiples of 10 s (readout windows
                   W = 10/50 starting at multiples of W; holder windows [t-50, t) with t a multiple of 10), so
                   block means reproduce them (up to the 4-decimal rounding the full traces already had).
  change_flags     src.integration.interface.changing_per_second computed REMOTELY on the full 1 s trace (it needs
                   1 s resolution), as 0/1 per second.
  spike_rate_bins  per-neuron 1 s spike counts (small).
Anything needing the full 1 s weight trace (e.g. displacement peaks) fetches the raw file:
    python -m modal volume get developing-mind-raw <raw_path> <local_path>
`raw_path` is stored in every compact file.

load_compact() returns the same (d, wm, rates, changing) tuple as
notebooks/integration/set_worlds/run_rectified_memory.load, with wm expanded back to 1 s by repeating each block,
so existing analysis code runs unchanged.
"""
import gzip
import io
import json
from pathlib import Path

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    # versions match the local developing-mind conda env (see CLAUDE.md, Modal)
    .pip_install("brian2==2.9.0", "cython==3.2.8", "numpy==2.0.1", "setuptools==82.0.1")
    .add_local_python_source("src")
)
RAW_VOLUME = modal.Volume.from_name("developing-mind-raw", create_if_missing=True)
RAW_MOUNT = "/raw"
BLOCK_S = 10


def gz_bytes(obj) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(json.dumps(obj).encode("utf-8"))
    return buf.getvalue()


def compact(result: dict, raw_path: str, block_s: int = BLOCK_S) -> dict:
    """Full result (with a 1 s 'weight_trace') -> compact result. Runs inside the container."""
    import numpy as np
    from src.integration.interface import changing_per_second

    w = np.asarray(result["weight_trace"], dtype=float)                 # (n_syn, T)
    T = w.shape[1]
    nb = int(np.ceil(T / block_s))
    pad = np.full((w.shape[0], nb * block_s - T), np.nan)
    blocks = np.nanmean(np.concatenate([w, pad], axis=1).reshape(w.shape[0], nb, block_s), axis=2)
    si, sj = np.asarray(result["syn_i"]), np.asarray(result["syn_j"])
    wm = np.zeros((int(result["n_post"]), int(si.max()) + 1, T)); wm[sj, si] = w
    flags = changing_per_second(wm).astype(int)
    out = {k: v for k, v in result.items() if k != "weight_trace"}
    out.update(compact=True, block_s=block_s, n_t=T, weight_blocks=np.round(blocks, 4).tolist(),
               change_flags=flags.tolist(), raw_volume="developing-mind-raw", raw_path=raw_path)
    return out


def write_raw(raw_path: str, result: dict):
    """Write the full result to the raw Volume (call inside a function mounted with RAW_VOLUME at RAW_MOUNT)."""
    p = Path(RAW_MOUNT) / raw_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gz_bytes(result))
    RAW_VOLUME.commit()


def load_compact(path):
    """Compact file -> (d, wm, rates, changing), the run_rectified_memory.load tuple, with wm expanded to 1 s."""
    import numpy as np
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    blocks = np.asarray(d["weight_blocks"], dtype=float)
    w = np.repeat(blocks, d["block_s"], axis=1)[:, : d["n_t"]]
    si, sj = np.asarray(d["syn_i"]), np.asarray(d["syn_j"])
    n_pre = int(si.max()) + 1
    d.setdefault("context_sets", [list(range(b * 10, b * 10 + 10)) for b in range(n_pre // 10)])
    wm = np.zeros((int(d["n_post"]), n_pre, w.shape[1])); wm[sj, si] = w
    return d, wm, np.asarray(d["spike_rate_bins"], dtype=float), np.asarray(d["change_flags"], dtype=bool)
