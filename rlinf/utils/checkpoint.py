# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any


def parse_global_step_from_checkpoint_path(
    checkpoint_path: str | os.PathLike[str],
) -> int:
    """Extract the global step from a checkpoint directory path.

    Args:
        checkpoint_path: Path ending in a ``global_step_<step>`` directory.

    Returns:
        The non-negative global step encoded in the directory name.

    Raises:
        ValueError: If the final directory is not named ``global_step_<step>``.
    """
    checkpoint_dir = os.path.basename(os.path.normpath(checkpoint_path))
    match = re.fullmatch(r"global_step_(\d+)", checkpoint_dir)
    if match is None:
        raise ValueError(
            "Checkpoint path must end with a 'global_step_<step>' directory, "
            f"but got {os.fspath(checkpoint_path)!r}."
        )
    return int(match.group(1))


class ValidationCheckpointTracker:
    """Keep the best validation checkpoint and a bounded number of recent ones.

    Only checkpoints registered after a successful save can be removed. The
    JSON manifest records the selection metric and survives runner restarts.
    """

    def __init__(
        self,
        checkpoint_root: str | os.PathLike[str],
        metric: str,
        mode: str = "min",
        keep_last: int = 1,
    ) -> None:
        if not metric or mode not in ("min", "max") or keep_last < 1:
            raise ValueError(
                "Checkpoint selection needs a metric, min/max mode, and keep_last >= 1."
            )
        self.root = Path(checkpoint_root)
        self.metric = metric
        self.mode = mode
        self.keep_last = keep_last
        self.best_step: int | None = None
        self.best_value: float | None = None
        self.saved_steps: list[int] = []
        self.manifest_path = self.root / "validation_selection.json"
        if self.manifest_path.exists():
            state = json.loads(self.manifest_path.read_text())
            if (state["metric"], state["mode"], state["keep_last"]) != (
                metric,
                mode,
                keep_last,
            ):
                raise ValueError(
                    "Checkpoint selection configuration differs from the saved manifest."
                )
            self.best_step = state["best_step"]
            self.best_value = state["best_value"]
            self.saved_steps = state["saved_steps"]
            if self.best_step is not None and not self.path(self.best_step).is_dir():
                raise FileNotFoundError("The selected best checkpoint is missing.")

    def path(self, step: int) -> Path:
        """Return the checkpoint directory for a non-negative optimizer step."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Checkpoint step must be a non-negative integer.")
        return self.root / f"global_step_{step}"

    def improves(self, metrics: dict[str, float]) -> bool:
        """Compare a finite validation metric with the best saved value."""
        value = float(metrics[self.metric])
        if not math.isfinite(value):
            raise ValueError(f"Validation metric {self.metric!r} must be finite.")
        if self.best_value is None:
            return True
        return (
            value < self.best_value if self.mode == "min" else value > self.best_value
        )

    def record_saved(self, step: int, metrics: dict[str, float] | None = None) -> None:
        """Register a completed save, persist selection, and prune owned saves.

        Args:
            step: Optimizer step whose checkpoint has finished saving.
            metrics: Validation metrics from this step, or None for a regular
                checkpoint saved without validation.
        """
        checkpoint = self.path(step)
        if self.saved_steps and step < max(self.saved_steps):
            raise ValueError(
                "Checkpoint steps must not move backward within one output directory."
            )
        if checkpoint.is_symlink() or not (checkpoint / "actor").is_dir():
            raise FileNotFoundError(
                f"Completed checkpoint directory is missing: {checkpoint}"
            )
        if metrics is not None and self.improves(metrics):
            self.best_step = step
            self.best_value = float(metrics[self.metric])
        self.saved_steps = sorted(set(self.saved_steps) | {step})
        state: dict[str, Any] = {
            "metric": self.metric,
            "mode": self.mode,
            "keep_last": self.keep_last,
            "best_step": self.best_step,
            "best_value": self.best_value,
            "best_checkpoint": None
            if self.best_step is None
            else str(self.path(self.best_step)),
            "saved_steps": self.saved_steps,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.manifest_path)
        # Preserve the selection alongside the checkpoint for resume provenance.
        (checkpoint / "validation_selection.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )
        keep = set(self.saved_steps[-self.keep_last :]) | {self.best_step}
        for old_step in self.saved_steps:
            if old_step in keep:
                continue
            obsolete = self.path(old_step)
            if obsolete.is_symlink():
                raise ValueError(f"Refusing to prune a checkpoint symlink: {obsolete}")
            if obsolete.exists():
                shutil.rmtree(obsolete)
        self.saved_steps = [saved for saved in self.saved_steps if saved in keep]
        state["saved_steps"] = self.saved_steps
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.manifest_path)
