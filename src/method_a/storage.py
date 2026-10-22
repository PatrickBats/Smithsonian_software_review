"""Method A data variants and paths. Historical run IDs stay unchanged on disk."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .configurations import EXPERIMENTS, SEED, resolve_variant, variant_id

_BASE_RHF = Path(os.environ.get("SMITHSONIAN_ROOT", "./external"))
_BASE_PROJECTS = Path(os.environ.get("SMITHSONIAN_OUTPUT", "./outputs"))
BASE = Path(
    os.environ.get(
        "METHOD_A_BASELINE_ROOT", _BASE_RHF / "processed/method_a_focus_stack_v1"
    )
)
ROOT = Path(
    os.environ.get(
        "METHOD_A_DATA_ROOT", _BASE_RHF / "processed/method_a_sparse_labels_20261017"
    )
)
CODE = Path(os.environ.get("METHOD_A_CODE_ROOT", Path(__file__).resolve().parent))
LOGS = Path(
    os.environ.get(
        "METHOD_A_LOG_ROOT", _BASE_PROJECTS / "runs/method_a_sparse_labels_20261017"
    )
)
RESULTS = Path(
    os.environ.get(
        "METHOD_A_RESULTS_ROOT",
        _BASE_PROJECTS / "results/method_a_sparse_labels_20261017",
    )
)
WEIGHTS = Path(
    os.environ.get(
        "METHOD_A_PRETRAINED_WEIGHTS", _BASE_RHF / "weights/rf-detr-xxlarge.pth"
    )
)
WEIGHT_SHA = "bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae"


def make_scratch(protocol):
    """Allocate a distinct cache directory for every smoke/full process."""
    return Path(
        tempfile.mkdtemp(
            prefix=f"method-a-{protocol}-", dir=os.environ.get("SLURM_TMPDIR") or None
        )
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    """Write owned metadata atomically, confined to the configured output roots.

    Validate resolved parents before creating anything. Dataset image links are
    read-only inputs and must never be accepted as metadata destinations.
    """
    path = Path(path).absolute()
    roots = (ROOT, CODE, LOGS, RESULTS)
    parent = path.parent.resolve()
    if not any(parent.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(
            f"Metadata destination is outside configured output roots: {path}"
        )
    if path.is_symlink():
        raise PermissionError(f"Refusing to overwrite a symlink: {path}")
    if path.exists() and path.stat().st_uid != os.getuid():
        raise PermissionError(f"Refusing to overwrite another owner's file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    created = False
    try:
        with temporary.open("x") as stream:
            created = True
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def load(path):
    return json.loads(Path(path).read_text())
