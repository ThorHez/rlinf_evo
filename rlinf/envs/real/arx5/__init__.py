# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""ARX tasks using the existing SDK service and a wrist camera."""

from rlinf.envs.real.registry import register_tasks

from .rlt import Arx5RLTEnv, Arx5RLTEnvConfig

TASKS = {"Arx5RLTEnv-v1": Arx5RLTEnv}
_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)
__all__ = ["Arx5RLTEnv", "Arx5RLTEnvConfig", "TASKS", *_ENTRY_POINTS]
