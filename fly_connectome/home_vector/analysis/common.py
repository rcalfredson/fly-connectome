"""Explicit bundle selection, source integrity checks, and fresh audit outputs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from ..__main__ import load
from ..data import Circuit, digest, verify_bundle, write_json


class Audit:
    """Verify relocated input bundles without changing their recorded provenance."""

    def __init__(self, args):
        self.args = args
        self.run = Path(args.run).resolve()
        self.source = Path(args.source).resolve()
        self.out = Path(args.output).resolve()
        for source in (self.run, self.source):
            if (self.out == source or self.out.is_relative_to(source)
                    or source.is_relative_to(self.out)):
                raise ValueError("Audit output must be separate from both input bundles.")
        if self.out.exists():
            raise FileExistsError(self.out)
        self.manifest = verify_bundle(self.run)
        if self.manifest.get("kind") != "home_vector_training_v1":
            raise ValueError("Require a completed home-vector training run.")
        verify_bundle(self.source)
        if digest(self.source / "manifest.json") != self.manifest["settings"]["source_manifest_sha256"]:
            raise ValueError("Prepared source manifest does not match the training run.")
        _, self.groups, self.task = load(self.source)
        # The run circuit may have been rewired; never silently substitute the source circuit.
        self.circuit = Circuit(**json.loads((self.run / "circuit.json").read_text()))
        self.hashes = {str(p): digest(p) for folder in (self.run, self.source)
                       for p in folder.iterdir() if p.is_file()}

    def predictions(self, split):
        """Load exactly one finite Cartesian prediction/target pair per trajectory."""
        count = len(self.groups[split])
        expected = {f"{kind}_{i}" for kind in ("prediction", "target") for i in range(count)}
        with np.load(self.run / f"{split}_predictions.npz", allow_pickle=False) as data:
            if set(data.files) != expected:
                raise ValueError(f"Prediction archive does not match {split} trajectory count.")
            predictions = [data[f"prediction_{i}"] for i in range(count)]
            targets = [data[f"target_{i}"] for i in range(count)]
        for row, prediction, target in zip(self.groups[split], predictions, targets):
            n = round(row["protocol"][-1][0] / self.task["dt_s"]) + 1
            if (prediction.shape != (n, 2) or target.shape != (n, 2)
                    or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
                raise ValueError(f"Invalid Cartesian prediction/target samples in {split}.")
        return predictions, targets

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=False)
        code_root = Path(__file__).parents[1]
        self.record = dict(
            status="running", kind=f"home_vector_{self.args.command}_audit_v1",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            settings=dict(run=str(self.run), source=str(self.source), output=str(self.out),
                          **{k: v for k, v in vars(self.args).items()
                             if k not in ("run", "source", "output", "command")}),
            source_sha256=self.hashes, refitting=False,
            code_sha256={str(p.relative_to(code_root)): digest(p)
                         for p in sorted(code_root.rglob("*.py"))},
            python=platform.python_version(), numpy=np.__version__)
        write_json(self.out / "manifest.json", self.record)
        return self

    def __exit__(self, error_type, error, traceback):
        if error is None:
            try:
                current = {str(p): digest(p) for folder in (self.run, self.source)
                           for p in folder.iterdir() if p.is_file()}
                if current != self.hashes:
                    raise ValueError("Input bundle files changed during audit.")
                self.record.update(status="complete", artifacts={
                    p.name: digest(p) for p in self.out.iterdir()
                    if p.is_file() and p.name != "manifest.json"})
            except Exception as integrity_error:
                self.record.update(status="failed", error=str(integrity_error))
                write_json(self.out / "manifest.json", self.record)
                raise
        else:
            self.record.update(status="failed", error=f"{error_type.__name__}: {error}")
        self.record["libraries"] = {
            name: str(sys.modules[name].__version__)
            for name in ("numpy", "pandas", "matplotlib", "torch") if name in sys.modules}
        self.record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.out / "manifest.json", self.record)
        return False
