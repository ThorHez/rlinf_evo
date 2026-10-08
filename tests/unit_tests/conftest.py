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

"""Make repository and shared test modules importable by unit tests."""

import sys
from pathlib import Path

import pytest

for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture
def action_probe_prepared(tmp_path):
    """Write a complete tiny prepared dataset for data and worker contracts."""
    import hashlib
    import json

    import numpy as np
    from omegaconf import OmegaConf

    directory = tmp_path / "prepared_probe"
    directory.mkdir()
    norm = directory / "norm_stats.json"
    norm.write_text('{"norm_stats": {}}\n')
    dataset = tmp_path / "original_dataset"
    split = {
        "fit_episode_ids": [1],
        "dev_episode_ids": [2],
        "holdout_episode_ids": [3, 4],
        "parent_stage1_split_signature_sha256": "test-parent-split",
        "dataset_path": str(dataset),
    }
    split_path = directory / "probe_split.json"
    split_path.write_text(json.dumps(split))
    samples = directory / "samples.jsonl"
    samples.write_text("{}\n" * 8)
    arrays = {
        "state": np.arange(8 * 32, dtype=np.float32).reshape(8, 32),
        "actions": np.arange(8 * 16 * 32, dtype=np.float32).reshape(8, 16, 32),
        "tokenized_prompt": np.ones((8, 200), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((8, 200), dtype=bool),
        "episode_id": np.repeat(np.arange(1, 5, dtype=np.int64), 2),
        "frame_index": np.tile(np.arange(2, dtype=np.int64), 4),
        "dataset_global_index": np.arange(100, 108, dtype=np.int64),
        "split": np.array([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.uint8),
    }
    for camera in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        arrays[f"image_{camera}"] = np.broadcast_to(
            np.arange(8, dtype=np.uint8)[:, None, None, None], (8, 224, 224, 3)
        )
        arrays[f"image_mask_{camera}"] = np.full(
            8, camera == "left_wrist_0_rgb", dtype=bool
        )
    records = {}
    for name, array in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, array)
        records[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "logical_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    provenance = {
        "complete": True,
        "dataset_path": str(dataset),
        "norm_stats_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        "probe_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "parent_stage1_split_signature_sha256": "test-parent-split",
        "image_augmentation": False,
        "arrays": records,
        "source_files_sha256": {},
    }
    (directory / "provenance.json").write_text(json.dumps(provenance))
    (directory / "input_quality_check.json").write_text(
        json.dumps({"passed": True, "sample_count": 8})
    )
    cfg = OmegaConf.create(
        {
            "data": {
                "train_data_paths": str(dataset),
                "val_data_paths": str(dataset),
                "train_episode_ids": [1, 2],
                "val_episode_ids": [3, 4],
            },
            "actor": {
                "model": {
                    "openpi_data": {"norm_stats_path": str(norm)},
                    "openpi": {
                        "rlt_embed_dim": 4,
                        "model_action_dim": 32,
                        "action_horizon": 16,
                        "max_token_len": 200,
                    },
                }
            },
        }
    )
    return directory, cfg
