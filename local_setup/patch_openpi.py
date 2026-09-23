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

"""Apply the upstream OpenPI Transformers overlay inside the active venv."""

import shutil
import sys
import sysconfig
from pathlib import Path

if sys.prefix == sys.base_prefix:
    raise SystemExit("Activate the RLinf virtual environment first.")
site = Path(sysconfig.get_paths()["purelib"])
source = site / "openpi/models_pytorch/transformers_replace"
destination = site / "transformers"
if not source.is_dir() or not destination.is_dir():
    raise SystemExit("Install rlinf-openpi and rlinf-transformer-openpi first.")
for item in source.rglob("*"):
    target = destination / item.relative_to(source)
    if item.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        # Keep shared uv cache inodes unchanged when replacing package files.
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(item, target)
print(f"Applied OpenPI overlay to {destination}")
