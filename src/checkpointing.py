"""Atomic checkpoint writes and guards against silently reusing another run."""

import hashlib
import json
from pathlib import Path

import torch


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def require_run_contract(path, contract, existing_artifacts=()):
    """Require exact provenance before resuming or skipping a finished variant.

    Historical runs without this file remain valid plotting inputs, but they
    cannot be silently adopted as new training output directories.
    """
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f"Run contract mismatch: {path}; use a new output directory")
    elif any(Path(p).exists() for p in existing_artifacts):
        raise ValueError(
            f"Existing results lack a run contract: {path.parent}; use a new output directory"
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(contract, indent=2))
        temporary.replace(path)


def save_atomic(obj, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)
