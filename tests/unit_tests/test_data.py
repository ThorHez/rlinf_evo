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

"""Datasets, batching, and the on-the-wire shapes they produce."""

import asyncio
import copy
import dataclasses
import inspect
import io
import json
import random
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import rlinf.utils.obs_compression as obs_compression
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.schema.embodied_trajectory import (
    LeRobotEpisodeAccumulator,
    RolloutGeometry,
    TrajectoryCollector,
    TrajectoryMode,
    TrajectoryPlan,
    select_trajectory_collector,
    select_trajectory_dispatcher,
)
from rlinf.data.schema.embodied_types import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    LeRobotFrame,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    Trajectory,
    TrajectoryKey,
    TrajectorySource,
    TrajectoryStep,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_episode_data,
)
from rlinf.data.storage.lerobot import add_frame_to_dataset, episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.channel.hooks import ChannelContext
from rlinf.scheduler.cluster.utils import (
    TensorPlaceholder,
    pack_dataclass_tensors,
    unpack_dataclass_tensors,
)
from rlinf.utils.env_helpers import SmoothInterveneController
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.obs_compression import (
    _CODEC_KEY,
    compress_obs,
    decompress_obs,
    infer_obs_batch_size,
    is_compressed_image,
    is_compression_enabled,
)
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class TestMathDatasetMultithread:
    """Tests for ReasoningDataset multithread processing consistency."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = mock.Mock()
        tokenizer.is_fast = True
        tokenizer.eos_token_id = 2

        def apply_chat_template_side_effect(
            prompts, tokenize=False, add_generation_prompt=True
        ):
            """Mock apply_chat_template that handles generator input."""
            # Convert generator to list if needed
            prompts_list = list(prompts) if not isinstance(prompts, list) else prompts
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n"
                if isinstance(prompt, str)
                else prompt
                for prompt in prompts_list
            ]

        tokenizer.apply_chat_template = mock.Mock(
            side_effect=apply_chat_template_side_effect
        )
        tokenizer.batch_encode_plus = mock.Mock(
            side_effect=lambda texts: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            }
        )
        tokenizer.encode = mock.Mock(side_effect=lambda text: [1] * len(text.split()))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = DictConfig(
            {
                "data": {
                    "max_prompt_length": 1000,
                    "prompt_key": "question",
                    "answer_key": "answer",
                    "apply_chat_template": True,
                    "filter_prompt_by_length": False,
                    "process_workers": 4,
                    "process_batch_size": 32,
                }
            }
        )
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing (at least 10000 items)."""
        # Generate at least 10000 math problems
        data = []
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a // b if b != 0 else 0),
        ]

        for i in range(10000):
            op_symbol, op_func = random.choice(operations)
            a = random.randint(1, 1000)
            b = random.randint(1, 1000) if op_symbol != "/" else random.randint(1, 100)
            if op_symbol == "/" and b == 0:
                b = 1
            result = op_func(a, b)
            question = f"What is {a} {op_symbol} {b}?"
            data.append({"question": question, "answer": str(result)})

        return data

    def test_multithread_vs_singlethread_consistency(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test that multithread processing produces identical results to single-thread processing.

        This test verifies that:
        1. Results from multi-worker processing match single-worker processing
        2. All keys are the same
        3. All values are the same
        """
        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        # We need to reload the raw data to avoid double processing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        time_start = time.time()
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )
        time_elapse_multithread = time.time() - time_start

        # Test with single thread
        time_start = time.time()
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)
        time_elapse_singlethread = time.time() - time_start

        # Verify lengths are equal
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify all items have the same keys and values
        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            keys_mt, keys_st = item_mt.keys(), item_st.keys()
            assert keys_mt == keys_st, (
                f"Keys mismatch at index {idx}: "
                f"multithread={list(keys_mt)}, singlethread={list(keys_st)}"
            )

            # Check all values are equal
            unequal_keys = [key for key in keys_mt if item_mt[key] != item_st[key]]
            assert len(unequal_keys) == 0, (
                f"Values mismatch at index {idx} for keys: {unequal_keys}"
            )

        # Log timing information (for debugging)
        print(
            f"Data count: {len(data_multithread)}, "
            f"Multithread processing time: {time_elapse_multithread:.2f}s, "
            f"Singlethread processing time: {time_elapse_singlethread:.2f}s"
        )

    def test_multithread_consistency_with_filter(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test multithread processing consistency when filter_prompt_by_length is enabled.
        """
        # Update config to enable filtering
        mock_config.data.filter_prompt_by_length = True
        mock_config.data.max_prompt_length = 50  # Reasonable limit to test filtering

        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )

        # Test with single thread
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)

        # Verify consistency
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify that some data was filtered (not all data passed)
        assert len(data_multithread) <= len(raw_data), (
            f"Filtering should reduce data size, but got {len(data_multithread)} >= {len(raw_data)}"
        )

        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            assert item_mt.keys() == item_st.keys(), f"Keys mismatch at index {idx}"
            for key in item_mt.keys():
                assert item_mt[key] == item_st[key], (
                    f"Mismatch at index {idx}, key {key}"
                )

        print(
            f"Filtered data count: {len(data_multithread)}/{len(raw_data)} "
            f"(max_prompt_length={mock_config.data.max_prompt_length})"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def test_split_dict_to_chunk_keeps_mixed_fields_aligned():
    batch = {
        "values": torch.arange(10),
        "sample_ids": list(range(10)),
        "nested": {"values": torch.arange(10) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3)

    assert [chunk["values"].tolist() for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["sample_ids"] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["nested"]["values"].tolist() for chunk in chunks] == [
        [100, 101, 102, 103],
        [104, 105, 106],
        [107, 108, 109],
    ]


def test_split_dict_to_chunk_returns_requested_number_of_chunks():
    batch = {"values": torch.arange(2), "sample_ids": ["a", "b"]}

    chunks = split_dict_to_chunk(batch, 4)

    assert len(chunks) == 4
    assert [chunk["values"].tolist() for chunk in chunks] == [[0], [1], [], []]
    assert [chunk["sample_ids"] for chunk in chunks] == [["a"], ["b"], [], []]


class _LegacyDataset:
    """Mimics lerobot < 0.2: the task lives inside the frame dict."""

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class _PostRevertDataset(_LegacyDataset):
    """Mimics lerobot >= 0.4: back to ``add_frame(frame)``, but it pops the task.

    0.4 reverted the 0.3.x signature, so a version-number check would dispatch
    this one wrongly. It also mutates the caller's dict.
    """

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append({**frame, "task": frame.pop("task")})


class _CurrentDataset:
    """Mimics lerobot >= 0.3: the task is a separate argument.

    Like the real implementation, a ``task`` key inside *frame* is rejected
    because it is not part of the feature schema.
    """

    def __init__(self):
        self.frames = []
        self.tasks = []
        self.saved_episodes = 0

    def add_frame(self, frame, task, timestamp=None):
        if "task" in frame:
            raise ValueError("Extra features: {'task'}")
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        self.saved_episodes += 1


def _make_writer(dataset):
    # ``create()`` needs a real lerobot install, so attach the dataset the way
    # ``create()`` would.
    writer = LeRobotDatasetWriter()
    writer.dataset = dataset
    return writer


def _episode(n=2):
    return [{"state": i, "actions": i, "task": "pick up the cube"} for i in range(n)]


def test_legacy_dataset_keeps_task_in_frame():
    dataset = _LegacyDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


def test_current_dataset_gets_task_as_argument():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode(_episode())

    assert dataset.tasks == ["pick up the cube"] * 2
    assert all("task" not in f for f in dataset.frames)
    assert dataset.frames[0]["state"] == 0
    assert dataset.saved_episodes == 1


ALL_SHAPES = [_LegacyDataset, _CurrentDataset, _PostRevertDataset]


def test_post_revert_dataset_keeps_task_in_frame():
    # lerobot >= 0.4 took the 0.3.x signature back out again.
    dataset = _PostRevertDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_caller_frames_are_not_mutated(dataset_cls):
    # The DAgger worker shares these dicts with the in-memory training store,
    # and lerobot >= 0.4 pops "task" out of whatever frame it is handed.
    episode = _episode()
    before = [dict(f) for f in episode]
    _make_writer(dataset_cls()).add_episode(episode)

    assert episode == before


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_frame_without_task_is_rejected(dataset_cls):
    with pytest.raises(ValueError, match="missing the required 'task' field"):
        add_frame_to_dataset(dataset_cls(), {"state": 0, "actions": 0})


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_add_frame_to_dataset_is_usable_standalone(dataset_cls):
    # The toolkit collectors drive LeRobotDataset directly, without the writer.
    dataset = dataset_cls()
    add_frame_to_dataset(dataset, {"state": 0, "task": "wipe the table"})

    assert len(dataset.frames) == 1


def test_empty_episode_is_skipped():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode([])

    assert dataset.frames == []
    assert dataset.saved_episodes == 0


# --------------------------------------------------------------------------
# episode_boundaries: dataset format v2.1 vs v3.0
# --------------------------------------------------------------------------


class _V21Dataset:
    """Dataset format v2.1: a dict of two tensors on the dataset itself."""

    def __init__(self, starts, ends):
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }


class _V30Meta:
    def __init__(self, starts, ends):
        self.episodes = {"dataset_from_index": starts, "dataset_to_index": ends}


class _V30Dataset:
    """Dataset format v3.0 (lerobot >= 0.4): columns on ``meta.episodes``."""

    def __init__(self, starts, ends):
        self.episode_data_index = None
        self.meta = _V30Meta(starts, ends)


@pytest.mark.parametrize("dataset_cls", [_V21Dataset, _V30Dataset])
def test_episode_boundaries_agree_across_formats(dataset_cls):
    starts, ends = episode_boundaries(dataset_cls([0, 3, 7], [3, 7, 9]))

    assert starts == [0, 3, 7]
    assert ends == [3, 7, 9]
    assert all(isinstance(x, int) for x in starts + ends)


def test_episode_boundaries_reports_an_unknown_layout():
    class _Alien:
        episode_data_index = None
        meta = None

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Alien())


def test_episode_boundaries_rejects_v30_meta_without_the_columns():
    class _Partial:
        episode_data_index = None
        meta = _V30Meta([0], [1])

    _Partial.meta.episodes = {"length": [1]}

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Partial())


# Skip codec round-trip tests when the optional backends are not installed.
_CODECS = []
try:
    import lz4.frame  # noqa: F401

    _CODECS.append("lz4")
except ImportError:
    pass
try:
    import zstandard  # noqa: F401

    _CODECS.append("zstd")
except ImportError:
    pass

requires_codec = pytest.mark.skipif(
    not _CODECS, reason="no observation compression codec (lz4/zstd) installed"
)


def _make_payload(num_envs: int = 4) -> dict:
    """A payload shaped like EnvWorker._build_rollout_input_data output."""
    obs = {
        "main_images": torch.randint(0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (num_envs, 6, 6, 3), dtype=torch.uint8
        ),
        "states": torch.randn(num_envs, 7, dtype=torch.float32),
        "task_descriptions": ["put carrot on plate"] * num_envs,
    }
    return {
        "obs": obs,
        "final_obs": {
            "main_images": torch.randint(
                0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8
            ),
            "states": torch.randn(num_envs, 7, dtype=torch.float32),
        },
        "rlt_switch_flags": None,
    }


def _assert_payload_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_payload_equal(va, vb)
        elif isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"tensor mismatch for {key!r}"
        else:
            assert va == vb, f"value mismatch for {key!r}"


def _cfg(**overrides):
    # A plain dict is sufficient: the codec only calls ``config.get(...)``,
    # which both ``dict`` and OmegaConf's ``DictConfig`` support identically.
    base = {"enable": True, "codec": "lz4", "level": 1, "xor_delta": True}
    base.update(overrides)
    return base


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
@pytest.mark.parametrize("xor_delta", [True, False])
def test_compress_decompress_is_lossless(codec, xor_delta):
    payload = _make_payload()
    config = _cfg(codec=codec, xor_delta=xor_delta)

    compressed = compress_obs(payload, config)
    # Image tensors are replaced by self-describing marker dicts...
    assert _CODEC_KEY in compressed["obs"]["main_images"]
    assert _CODEC_KEY in compressed["obs"]["extra_view_images"]
    # ...while non-image fields are passed through untouched.
    assert isinstance(compressed["obs"]["states"], torch.Tensor)
    assert compressed["obs"]["task_descriptions"] == payload["obs"]["task_descriptions"]

    restored = decompress_obs(compressed)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_single_env_batch_roundtrip():
    # XOR-delta is skipped when there is only one frame; must still be lossless.
    payload = _make_payload(num_envs=1)
    restored = decompress_obs(compress_obs(payload, _cfg(xor_delta=True)))
    _assert_payload_equal(payload, restored)


def test_disabled_config_is_passthrough():
    payload = _make_payload()
    assert compress_obs(payload, _cfg(enable=False)) is payload
    assert compress_obs(payload, None) is payload
    assert not is_compression_enabled(None)
    assert not is_compression_enabled(_cfg(enable=False))
    assert is_compression_enabled(_cfg(enable=True))


def test_decompress_on_uncompressed_payload_is_noop():
    # The rollout worker always routes received data through decompress_obs, so
    # it must be a no-op on payloads sent without compression.
    payload = _make_payload()
    restored = decompress_obs(payload)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_only_uint8_images_are_compressed():
    # A float image-shaped tensor is not a uint8 observation and must be left
    # untouched, as must low-rank uint8 tensors (e.g. flags).
    payload = {
        "obs": {
            "float_map": torch.randn(4, 8, 8, 3),
            "uint8_flags": torch.ones(4, dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert isinstance(compressed["obs"]["float_map"], torch.Tensor)
    assert isinstance(compressed["obs"]["uint8_flags"], torch.Tensor)


def test_unknown_codec_raises():
    payload = _make_payload()
    with pytest.raises(ValueError, match="Unknown observation compression codec"):
        compress_obs(payload, _cfg(codec="bogus"))


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
def test_routing_split_then_compress_roundtrip(codec):
    """Compression must be compatible with the Env->Rollout channel routing.

    The env worker installs compression as a ``split_fn`` so it runs *after*
    the scheduler splits the batch: ``infer_batch_size`` and ``split_batch``
    see plain tensors, and each shard is compressed independently. This test
    reproduces that flow with the real routing helpers and asserts the payload
    survives split -> compress -> decompress -> merge unchanged.
    """
    routing = pytest.importorskip("rlinf.scheduler.worker.routing")

    payload = _make_payload(num_envs=6)
    # The scheduler infers the batch size from the *uncompressed* payload.
    assert routing.infer_batch_size(payload) == 6

    # split_fn = split_batch first, then compress each shard (env send path).
    split_sizes = [2, 1, 3]
    shards = routing.split_batch(payload, split_sizes)
    compressed_shards = [compress_obs(shard, _cfg(codec=codec)) for shard in shards]

    # Rollout side: decompress each shard, then merge (merge_obs path).
    restored_shards = [decompress_obs(shard) for shard in compressed_shards]
    merged = routing.merge_batches(restored_shards)
    _assert_payload_equal(payload, merged)


def test_infer_obs_batch_size_uncompressed():
    payload = _make_payload(num_envs=5)
    assert infer_obs_batch_size(payload) == 5
    # Also accepts a bare obs dict (no "obs" wrapper).
    assert infer_obs_batch_size(payload["obs"]) == 5


@requires_codec
def test_infer_obs_batch_size_with_compressed_images():
    # The rollout worker infers batch size on the receive path, before
    # decompression, so a compressed image must still report its batch size.
    payload = _make_payload(num_envs=5)
    compressed = compress_obs(payload, _cfg())
    assert is_compressed_image(compressed["obs"]["main_images"])
    assert infer_obs_batch_size(compressed) == 5


@requires_codec
def test_infer_obs_batch_size_images_only():
    # Regression: a batch whose only batched field is a (compressed) image,
    # with no states/task_descriptions, must not break batch-size inference.
    payload = {
        "obs": {
            "main_images": torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert infer_obs_batch_size(compressed) == 3


def test_infer_obs_batch_size_raises_when_unbatched():
    with pytest.raises(ValueError, match="Cannot infer batch size"):
        infer_obs_batch_size({"obs": {}})


def test_env_output_composes_one_transition_object():
    transition = EnvTransition(
        rewards=torch.ones(2, 1),
        dones=torch.zeros(2, 1, dtype=torch.bool),
    )
    output = EnvOutput(obs={"states": torch.zeros(2, 3)}, transition=transition)

    assert output.transition is transition
    assert output.rewards is transition.rewards
    assert output.dones is transition.dones
    assert set(output.__dataclass_fields__) == {
        "obs",
        "transition",
        "final_obs",
        "env_infos",
    }


def test_removed_duplicate_types_are_not_schema_api():
    import rlinf.data.schema as schema

    for name in (
        "ChunkStepResult",
        "DummyPolicyInput",
        "EmbodiedRolloutResult",
        "EnvResult",
        "PolicyCompletion",
    ):
        assert not hasattr(schema, name)


def test_env_part_completion_reuses_the_environment_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    transition = EnvTransition(rewards=torch.ones(2, 1))
    part = EnvPart(
        sources=[TrajectorySource(key, 2)],
        transition=transition,
        next_obs={"states": torch.zeros(2, 3)},
        requires_inference=True,
    )

    completed = part.complete(
        next_obs=part.next_obs,
        next_rlt_obs={"states": torch.ones(2, 3)},
        final_prev_values=torch.tensor([[2.0], [3.0]]),
    )

    assert completed.transition is transition
    assert not completed.requires_inference
    assert torch.equal(completed.bootstrap_values, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(completed.final_prev_values, completed.bootstrap_values)


def test_transport_results_store_contiguous_cpu_tensors():
    non_contiguous = torch.arange(12).reshape(3, 4).T

    env_transition = EnvTransition(rewards=non_contiguous)
    policy_output = PolicyOutput(
        forward_inputs={"states": non_contiguous},
    )

    assert env_transition.rewards.device.type == "cpu"
    assert env_transition.rewards.is_contiguous()
    assert not hasattr(policy_output, "actions")
    assert policy_output.forward_inputs["states"].device.type == "cpu"
    assert policy_output.forward_inputs["states"].is_contiguous()


def test_policy_output_detaches_nested_model_tensors_for_transport():
    source = torch.arange(12.0, requires_grad=True)
    model_output = (source * 2).reshape(3, 4).T
    policy_output = PolicyOutput(
        forward_inputs={
            "nested": {
                "states": model_output,
                "features": [model_output[:, :2]],
            }
        },
        prev_logprobs=model_output,
        prev_values=model_output[:, :1],
        versions=model_output,
    )
    part = PolicyPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 4)],
        obs={},
        output=policy_output,
    )

    _, tensors = pack_dataclass_tensors(part)
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(not tensor.requires_grad for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_nested_dataclass_transport_separates_tensors_from_skeleton():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    shared = torch.arange(4).reshape(2, 2)
    event = PolicyPart(
        sources=[TrajectorySource(key, 2)],
        obs={"states": shared, "task_descriptions": ["a", "b"]},
        output=PolicyOutput(
            forward_inputs={"nested": {"states": shared}},
            prev_values=torch.ones(2, 1),
        ),
    )

    skeleton, tensors = pack_dataclass_tensors(event)
    restored = unpack_dataclass_tensors(skeleton, tensors)

    assert isinstance(skeleton.obs["states"], TensorPlaceholder)
    assert isinstance(
        skeleton.output.forward_inputs["nested"]["states"],
        TensorPlaceholder,
    )
    assert len(tensors) == 2
    assert restored.obs["states"] is restored.output.forward_inputs["nested"]["states"]
    assert torch.equal(restored.output.prev_values, torch.ones(2, 1))


def test_policy_input_split_merge_preserves_sources_and_nested_payloads():
    keys = [TrajectoryKey(0, 0, rank, 0, 2) for rank in range(2)]
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(12).reshape(4, 3),
            "labels": np.arange(4),
        },
        rlt_switch_flags=torch.tensor([False, True, False, True]),
        sources=[TrajectorySource(key, 2) for key in keys],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[0].sources == [TrajectorySource(keys[0], 1)]
    assert shards[1].sources == [
        TrajectorySource(keys[0], 1, offset=1),
        TrajectorySource(keys[1], 2),
    ]
    assert merged.sources == policy_input.sources
    assert torch.equal(merged.obs["states"], policy_input.obs["states"])
    assert np.array_equal(merged.obs["labels"], policy_input.obs["labels"])
    assert torch.equal(merged.rlt_switch_flags, policy_input.rlt_switch_flags)


def test_external_policy_input_split_merge_preserves_actions():
    key = TrajectoryKey(0, 0, 0, 0, 1)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        external_actions=torch.arange(24).reshape(4, 2, 3),
        sources=[TrajectorySource(key, 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert all(not shard.requires_inference for shard in shards)
    assert not merged.requires_inference
    assert torch.equal(merged.external_actions, policy_input.external_actions)
    assert merged.sources == policy_input.sources


def test_online_lerobot_payload_survives_source_routing():
    episode_data = {
        "chunk_actions": torch.arange(12).reshape(3, 2, 2),
        "obs_list": [
            {"images": torch.arange(6).reshape(3, 2)},
            {"images": torch.arange(6, 12).reshape(3, 2)},
        ],
        "terminations": torch.tensor([False, True, False]),
        "truncations": torch.tensor([False, False, True]),
        "infos_list": [
            {"score": np.arange(3)},
            {"score": np.arange(3, 6)},
        ],
    }
    split_data = split_episode_data(episode_data, [1, 2])
    merged_data = merge_episode_data(split_data)

    assert merged_data is not None
    assert torch.equal(merged_data["chunk_actions"], episode_data["chunk_actions"])
    assert torch.equal(
        merged_data["obs_list"][1]["images"],
        episode_data["obs_list"][1]["images"],
    )
    assert np.array_equal(
        merged_data["infos_list"][0]["score"],
        episode_data["infos_list"][0]["score"],
    )


def test_env_part_split_merge_preserves_offsets():
    current_key = TrajectoryKey(1, 2, 3, 0, 4)
    previous_key = TrajectoryKey(1, 2, 3, 0, 3)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        sources=[TrajectorySource(current_key, 4)],
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(previous_key, 4)],
                transition=EnvTransition(rewards=torch.arange(4).reshape(4, 1)),
                next_obs={"states": torch.arange(12).reshape(4, 3)},
                requires_inference=False,
                initial_transition=EnvTransition(
                    dones=torch.zeros(4, 1, dtype=torch.bool)
                ),
            )
        ],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[1].sources == [TrajectorySource(current_key, 3, offset=1)]
    assert shards[1].env_parts[0].sources == [
        TrajectorySource(previous_key, 3, offset=1)
    ]
    assert merged.request_sizes == [1, 3]
    assert torch.equal(
        shards[1].env_parts[0].initial_transition.dones,
        torch.zeros(3, 1, dtype=torch.bool),
    )
    assert torch.equal(
        merged.env_parts[1].next_obs["states"],
        torch.arange(12).reshape(4, 3)[1:],
    )


def test_scalar_batch_leaves_survive_a_split_merge_round_trip():
    """Scalars broadcast by a split must collapse back to the original value.

    Source fragments carry scalar info flags (e.g. ``record_reset``) unchanged,
    so reassembling them must not turn one flag into a list of per-shard copies.
    """
    for value in (True, 3, 1.5, "reset"):
        shards = split_batch_value(value, [2, 2])
        assert shards == [value, value]
        assert merge_batch_values(shards) == value

    nested = {"flag": True, "name": "abc", "states": torch.arange(4).reshape(4, 1)}
    shards = split_batch_value(nested, [3, 1])
    merged = merge_batch_values(shards)
    assert merged["flag"] is True
    assert merged["name"] == "abc"
    assert torch.equal(merged["states"], nested["states"])


def test_merging_conflicting_scalar_batch_values_is_rejected():
    with pytest.raises(ValueError, match="conflicting scalar"):
        merge_batch_values([True, False])


def test_episode_data_round_trip_keeps_scalar_info_flags_intact():
    episode_data = {
        "chunk_actions": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "obs_list": [{"states": torch.arange(4).reshape(4, 1)}],
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
        "truncations": torch.zeros(4, 1, dtype=torch.bool),
        "infos_list": [{"record_reset": True, "segment_advance": False}],
    }

    merged = merge_episode_data(split_episode_data(episode_data, [3, 1]))

    assert merged["infos_list"][0]["record_reset"] is True
    assert merged["infos_list"][0]["segment_advance"] is False
    assert torch.equal(merged["chunk_actions"], episode_data["chunk_actions"])


def test_policy_part_owns_routed_fragment_split_and_merge():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    actions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    part = PolicyPart(
        sources=[TrajectorySource(key, 1), TrajectorySource(key, 3, offset=1)],
        obs={"states": torch.arange(12).reshape(4, 3)},
        output=PolicyOutput(
            forward_inputs={"action": actions},
            prev_values=torch.arange(4).reshape(4, 1),
        ),
    )

    fragments = part.split()
    merged = PolicyPart.merge(fragments)

    assert len(fragments) == 2
    assert fragments[1].sources == [TrajectorySource(key, 3, offset=1)]
    assert torch.equal(fragments[0].output.forward_inputs["action"], actions[:1])
    assert torch.equal(merged.obs["states"], part.obs["states"])


def test_trajectory_step_owns_intervention_and_transition_conversion():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    model_actions = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    policy = PolicyPart(
        sources=[TrajectorySource(key, 1)],
        obs={"states": torch.zeros(1, 2), "task_descriptions": ["pick"]},
        output=PolicyOutput(
            forward_inputs={"action": model_actions, "model_action": model_actions},
            prev_values=torch.ones(1, 1),
        ),
    )
    env = EnvPart(
        sources=[TrajectorySource(key, 1)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            intervene_actions=torch.tensor([[9.0, 8.0, 7.0, 6.0]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
        next_obs={"states": torch.ones(1, 2), "task_descriptions": ["pick"]},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=torch.full((1, 1), 5.0),
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )

    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=True,
        collect_transitions=True,
        enable_rlt=False,
        include_final_value=True,
    )

    assert torch.equal(step.actions, torch.tensor([[1.0, 2.0, 7.0, 6.0]]))
    assert torch.equal(step.forward_inputs["action"], step.actions)
    assert "model_action" not in step.forward_inputs
    assert "task_descriptions" not in step.curr_obs
    assert "task_descriptions" not in step.next_obs
    assert torch.equal(step.final_prev_values, torch.full((1, 1), 5.0))


def test_trajectory_owns_step_materialization_splitting_and_batching():
    steps = [
        TrajectoryStep(
            actions=torch.tensor([[1.0], [2.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            initial_dones=torch.zeros(2, 1, dtype=torch.bool),
            forward_inputs={"action": torch.tensor([[1.0], [2.0]])},
            versions=torch.ones(2, 1),
        ),
        TrajectoryStep(
            actions=torch.tensor([[3.0], [4.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            final_prev_values=torch.zeros(2, 1),
            forward_inputs={"action": torch.tensor([[3.0], [4.0]])},
            versions=torch.ones(2, 1),
        ),
    ]

    trajectory = Trajectory.from_steps(steps, max_episode_length=8)
    shards = trajectory.split(2)
    batch = Trajectory.to_batch(shards)

    assert trajectory.actions.shape == (2, 2, 1)
    assert trajectory.dones.shape == (3, 2, 1)
    assert [shard.actions.shape for shard in shards] == [(2, 1, 1)] * 2
    assert torch.equal(batch["actions"], trajectory.actions)
    assert torch.equal(batch["forward_inputs"]["action"], trajectory.actions)


def test_policy_input_methods_replace_legacy_routing_helpers():
    policy_input = PolicyInput(
        obs={"states": torch.arange(8).reshape(4, 2)},
        sources=[TrajectorySource(TrajectoryKey(1, 0, 0, 0, 0), 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert torch.equal(merged.obs["states"], policy_input.obs["states"])


def test_accumulators_are_not_public_schema_api():
    import rlinf.data.schema as schema

    assert not hasattr(schema, "TrajectoryAccumulator")
    assert not hasattr(schema, "EmbodiedTrajectoryBuilder")
    assert not hasattr(schema, "EmbodiedLerobotTrajectoryBuilder")


def _config(**overrides):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "max_episode_steps": 4,
                    "max_steps_per_rollout_epoch": 1,
                    "rollout_epoch": 1,
                    "total_num_envs": 1,
                }
            },
            "rollout": {
                "collect_prev_infos": True,
                "collect_transitions": False,
                "pipeline_stage_num": 1,
            },
            "actor": {
                "micro_batch_size": 1,
                "seed": 1,
                "model": {"action_dim": 1, "num_action_chunks": 1},
            },
            "algorithm": {
                "adv_type": "gae",
                "dagger": {"online_lerobot": {"enabled": False}},
                "gae_lambda": 1.0,
                "gamma": 0.5,
                "group_size": 1,
                "loss_type": "actor_critic",
                "normalize_advantages": False,
                "reward_type": "chunk_level",
                "shuffle_rollout": False,
            },
            "reward": {"env_reward_weight": 1.0, "reward_weight": 1.0},
            "runner": {
                "enable_decoupled_mode": False,
                "task_type": "embodied",
                "use_training_pipeline": False,
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _source(key: TrajectoryKey, size: int = 1, offset: int = 0):
    return TrajectorySource(key, size, offset)


def _policy(
    key: TrajectoryKey,
    value: float = 1.0,
    *,
    external: bool = False,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    kwargs = (
        {"external_actions": value_tensor.unsqueeze(1)}
        if external
        else {
            "output": PolicyOutput(
                forward_inputs={"action": value_tensor},
                prev_logprobs=torch.zeros(1, 1),
                prev_values=torch.zeros(1, 1),
                versions=torch.full((1, 1), 7.0),
            )
        }
    )
    return PolicyPart(
        sources=[_source(key, **source_kwargs)],
        obs={"states": value_tensor},
        **kwargs,
    )


def _env_part(
    key: TrajectoryKey,
    *,
    episode_data=None,
    initial_transition=None,
    value: float = 2.0,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    return EnvPart(
        sources=[_source(key, **source_kwargs)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.ones(1, 1, dtype=torch.bool),
            truncations=torch.ones(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            episode_data=episode_data,
        ),
        next_obs={"states": value_tensor},
        next_rlt_obs={"states": value_tensor},
        bootstrap_values=torch.tensor([[4.0]]),
        final_prev_values=torch.tensor([[5.0]]),
        initial_transition=initial_transition,
    )


def _episode_data():
    return {
        "chunk_actions": torch.zeros(1, 1, 1),
        "obs_list": [{"states": torch.zeros(1, 1)}],
        "terminations": torch.zeros(1, 1, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, dtype=torch.bool),
        "infos_list": [{}],
    }


def _make(cfg=None, **geometry):
    """Build the public collector with a deterministic cluster geometry."""
    shape = {
        "source_count": 1,
        "chunk_count": 1,
        "shards_per_source": 1,
        "actor_world_size": 1,
    }
    shape.update(geometry)
    collector = TrajectoryCollector()
    with patch.object(
        RolloutGeometry, "from_cfg", return_value=RolloutGeometry(**shape)
    ):
        collector.setup(
            ChannelContext(name="Actor", cfg=cfg if cfg is not None else _config())
        )
    return collector


def _collect(collector, *parts):
    outputs = []
    for part in parts:
        outputs.extend(collector.collect(part, DEFAULT_KEY))
    return outputs


def test_policy_part_requires_exactly_one_policy_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    common = {"sources": [_source(key)], "obs": {"states": torch.zeros(1, 1)}}

    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(**common)
    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(
            **common,
            output=PolicyOutput(forward_inputs={}),
            external_actions=torch.zeros(1, 1),
        )


def test_setup_requires_the_run_config():
    with pytest.raises(ValueError, match="needs the run config"):
        TrajectoryCollector().setup(ChannelContext(name="Actor", cfg=None))


@pytest.mark.parametrize(
    ("overrides", "mode", "dispatcher"),
    [
        ({}, TrajectoryMode.ROLLOUT, "least_loaded"),
        (
            {"runner": {"enable_decoupled_mode": False}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"enable_decoupled_mode": True}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"use_training_pipeline": True}},
            TrajectoryMode.PIPELINE,
            None,
        ),
        (
            {"algorithm": {"dagger": {"online_lerobot": {"enabled": True}}}},
            TrajectoryMode.LEROBOT,
            "least_loaded",
        ),
    ],
    ids=["sync", "async", "decoupled", "pipeline", "lerobot"],
)
def test_plan_is_the_single_mode_and_dispatcher_source(overrides, mode, dispatcher):
    cfg = _config(**overrides)

    assert TrajectoryPlan.mode_from_cfg(cfg) is mode
    assert select_trajectory_collector(cfg) is TrajectoryCollector
    assert select_trajectory_dispatcher(cfg) == dispatcher


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "runner": {
                "enable_decoupled_mode": True,
                "use_training_pipeline": True,
            }
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"dagger": {"online_lerobot": {"enabled": True}}},
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"adv_type": "opd"},
        },
    ],
    ids=["pipeline-decoupled", "pipeline-lerobot", "pipeline-opd"],
)
def test_plan_rejects_unsupported_mode_combinations(overrides):
    with pytest.raises(ValueError, match="does not support"):
        TrajectoryPlan.mode_from_cfg(_config(**overrides))


@pytest.mark.parametrize(
    ("decoupled", "part_order"),
    [
        (False, ("policy", "env")),
        (False, ("env", "policy")),
        (True, ("policy", "env")),
        (True, ("env", "policy")),
    ],
    ids=[
        "sync-policy-first",
        "sync-env-first",
        "decoupled-policy-first",
        "decoupled-env-first",
    ],
)
def test_rollout_output_matches_main_fields_for_sync_and_decoupled(
    decoupled, part_order
):
    """Assert the fields built by main's EnvWorker remain byte-for-byte equal."""
    cfg = _config(runner={"enable_decoupled_mode": decoupled})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    parts = {
        "policy": _policy(key),
        "env": _env_part(key, initial_transition=initial),
    }

    [(_, trajectory)] = _collect(collector, *(parts[name] for name in part_order))

    # These are the exact values main appends in EnvWorker._run_interact_once:
    # one initial boundary, one policy step, and gamma * terminal bootstrap.
    assert torch.equal(trajectory.actions, torch.tensor([[[1.0]]]))
    assert torch.equal(trajectory.prev_logprobs, torch.tensor([[[0.0]]]))
    assert torch.equal(trajectory.prev_values, torch.tensor([[[0.0]], [[5.0]]]))
    assert torch.equal(trajectory.rewards, torch.tensor([[[3.0]]]))
    assert torch.equal(trajectory.dones, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.truncations, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.terminations, torch.tensor([[[False]], [[False]]]))
    assert torch.equal(trajectory.versions, torch.tensor([[[7.0]]]))
    assert torch.equal(trajectory.forward_inputs["action"], torch.tensor([[[1.0]]]))


@pytest.mark.parametrize("decoupled", [False, True], ids=["async", "decoupled"])
def test_rollout_sources_flush_independently_like_main_env_workers(decoupled):
    """A slow env source must not add a global barrier outside pipeline mode."""
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"enable_decoupled_mode": decoupled},
    )
    collector = _make(cfg, source_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    first_key = TrajectoryKey(4, 0, 0, 0, 0)
    first_outputs = _collect(
        collector,
        _policy(first_key),
        _env_part(first_key, initial_transition=initial),
    )
    assert len(first_outputs) == 1

    second_key = TrajectoryKey(4, 0, 1, 0, 0)
    second_outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )
    assert len(second_outputs) == 1


def test_collector_joins_out_of_order_routed_fragments_and_initial_state():
    collector = _make(source_count=1, chunk_count=1, shards_per_source=1)
    collector._joiner._source_batch_size = 2
    key = TrajectoryKey(0, 0, 0, 0, 0)
    initial = EnvTransition(dones=torch.zeros(2, 1, dtype=torch.bool))
    policy_tail = _policy(key, 11.0, size=1, offset=1)
    policy_head = _policy(key, 10.0, size=1, offset=0)
    env = EnvPart(
        sources=[_source(key, size=2)],
        transition=EnvTransition(
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            truncations=torch.zeros(2, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.tensor([[20], [21]])},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=initial,
    )

    assert _collect(collector, policy_tail, policy_head) == []
    [(_, trajectory)] = _collect(collector, env)

    assert torch.equal(trajectory.actions, torch.tensor([[[10.0], [11.0]]]))
    assert torch.equal(trajectory.dones[0], initial.dones)


def test_collector_rejects_initial_state_after_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 1)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="Only chunk zero"):
        _collect(collector, _env_part(key, initial_transition=EnvTransition()))


def test_collector_requires_initial_state_for_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="missing its initial state"):
        _collect(collector, _env_part(key))


def test_collector_keeps_joined_parts_when_output_materialization_fails():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy = _policy(key)
    env = _env_part(key, initial_transition=EnvTransition())
    original_emit = collector._output.emit
    collector._output.emit = Mock(side_effect=RuntimeError("invalid output"))

    _collect(collector, policy)
    with pytest.raises(RuntimeError, match="invalid output"):
        _collect(collector, env)

    collector._output.emit = original_emit
    assert len(_collect(collector, policy)) == 1


def test_rollout_collector_rejects_duplicate_completed_key():
    cfg = _config(env={"train": {"rollout_epoch": 2}})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )
        == []
    )
    with pytest.raises(ValueError, match="duplicate trajectory event"):
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )


def test_rollout_materializes_each_epoch_boundary_and_final_value_in_order():
    collector = _make(_config(env={"train": {"rollout_epoch": 2}}))
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 1, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    [(_, trajectory)] = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert torch.equal(
        trajectory.dones,
        torch.tensor([[[False]], [[True]], [[False]], [[True]]]),
    )
    assert torch.equal(
        trajectory.prev_values,
        torch.tensor([[[0.0]], [[5.0]], [[0.0]], [[5.0]]]),
    )


def test_rollout_omits_policy_statistics_when_collection_is_disabled():
    collector = _make(_config(rollout={"collect_prev_infos": False}))
    key = TrajectoryKey(3, 0, 0, 0, 0)

    [(_, trajectory)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=EnvTransition()),
    )

    assert trajectory.prev_logprobs is None
    assert trajectory.prev_values is None


def test_history_reward_assignment_matches_main_across_chunks():
    cfg = _config(
        env={
            "train": {
                "auto_reset": False,
                "max_steps_per_rollout_epoch": 2,
            }
        },
        reward={
            "env_reward_weight": 1.0,
            "history_reward_assign": True,
            "reward_mode": "history_buffer",
            "reward_weight": 2.0,
        },
    )
    collector = _make(cfg, chunk_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 0, 0, 0, 1)
    first_env = _env_part(first_key, initial_transition=initial)
    first_env.transition.dones = torch.zeros(1, 1, dtype=torch.bool)
    first_env.transition.truncations = torch.zeros(1, 1, dtype=torch.bool)
    second_env = _env_part(second_key)
    second_env.transition.reward_model_output = torch.tensor([[3.0]])
    second_env.transition.reward_assign_lengths = [2]

    assert _collect(collector, _policy(first_key), first_env) == []
    [(_, trajectory)] = _collect(
        collector,
        second_env,
        _policy(second_key),
    )

    # Current reward: 1 + 2 * 3. History assignment adds 2 * 3 to step 0.
    assert torch.equal(trajectory.rewards, torch.tensor([[[7.0]], [[7.0]]]))


@pytest.mark.parametrize("loss_type", ["rlt_ac", "rlt_td3"])
def test_rlt_output_matches_main_transition_and_intervention_behavior(loss_type):
    collector = _make(_config(algorithm={"loss_type": loss_type}))
    key = TrajectoryKey(3, 0, 0, 0, 0)
    current_ref_chunk = torch.tensor([[1.0, 2.0]])
    intervened_chunk = torch.tensor([[9.0, 8.0]])
    policy = PolicyPart(
        sources=[_source(key)],
        obs={"states": torch.zeros(1, 1)},
        output=PolicyOutput(
            forward_inputs={
                "action": current_ref_chunk,
                "z_rl": torch.tensor([[3.0, 4.0]]),
                "proprio": torch.tensor([[5.0, 6.0, 7.0]]),
                "ref_chunk": current_ref_chunk,
            },
            prev_logprobs=torch.zeros(1, 1),
            prev_values=torch.zeros(1, 1),
        ),
    )
    env = EnvPart(
        sources=[_source(key)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
            intervene_actions=intervened_chunk,
            intervene_flags=torch.ones(1, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.ones(1, 1)},
        next_rlt_obs={
            "z_rl": torch.tensor([[13.0, 14.0]]),
            "proprio": torch.tensor([[15.0, 16.0, 17.0]]),
            "ref_chunk": torch.tensor([[11.0, 12.0]]),
        },
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=EnvTransition(),
    )

    [(_, trajectory)] = _collect(collector, env, policy)

    assert set(trajectory.curr_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert set(trajectory.next_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert torch.equal(trajectory.curr_obs["ref_chunk"][0], intervened_chunk)
    assert torch.equal(
        trajectory.next_obs["ref_chunk"][0], torch.tensor([[11.0, 12.0]])
    )


def test_pipeline_output_contains_main_actor_training_fields():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    key = TrajectoryKey(2, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    [(queue_key, batch)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=initial),
    )

    assert queue_key == "0_0_pipeline_actor"
    assert {
        "actions",
        "advantages",
        "dones",
        "prev_logprobs",
        "prev_values",
        "returns",
        "rewards",
    } <= batch.keys()
    assert batch["actions"].shape[0] == 1
    assert batch["advantages"].shape == batch["returns"].shape


def test_pipeline_waits_for_all_sources_and_routes_each_actor_like_main():
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"use_training_pipeline": True},
    )
    collector = _make(
        cfg,
        source_count=2,
        actor_world_size=2,
    )
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(2, 0, 0, 0, 0)
    second_key = TrajectoryKey(2, 0, 1, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert {queue_key for queue_key, _ in outputs} == {
        "0_0_pipeline_actor",
        "1_1_pipeline_actor",
    }
    assert all("advantages" in batch for _, batch in outputs)


def test_pipeline_flushes_each_epoch_to_the_actor_specific_key():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    collector._output._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    collector._output._pipeline_micro_batches = lambda batch, actor_rank: [batch]

    for epoch_id in (0, 1):
        key = TrajectoryKey(2, epoch_id, 0, 0, 0)
        outputs = _collect(
            collector,
            _env_part(key, initial_transition=EnvTransition()),
            _policy(key),
        )
        assert outputs[0][0] == "0_0_pipeline_actor"
        assert (2, epoch_id) not in collector._output._accumulators


def test_online_lerobot_accepts_external_policy_actions_and_emits_shards():
    cfg = _config(
        algorithm={"dagger": {"online_lerobot": {"enabled": True}}},
    )
    collector = _make(cfg, shards_per_source=2)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    outputs = _collect(
        collector,
        _policy(key, external=True),
        _env_part(
            key,
            episode_data=_episode_data(),
            initial_transition=EnvTransition(),
        ),
    )

    assert len(outputs) == 2
    assert all(key == DEFAULT_KEY for key, _ in outputs)


def test_online_lerobot_accumulator_records_external_intervened_action():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=2,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0, 2.0]]]),
        obs_list=[
            {
                "states": torch.tensor([[0.0, 0.0]]),
                "task_descriptions": ["pick"],
            }
        ],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "intervene_action": torch.tensor([[9.0, 8.0]]),
                "intervene_flag": torch.tensor([True]),
            }
        ],
    )

    [episode] = accumulator.drain_episodes()
    assert torch.equal(
        torch.from_numpy(episode[0]["actions"]), torch.tensor([9.0, 8.0])
    )


def test_online_lerobot_accumulator_preserves_auto_reset_observation():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0]]]),
        obs_list=[{"states": torch.tensor([[100.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "final_observation": {"states": torch.tensor([[10.0]])},
                "final_info": {"success_once": torch.tensor([True])},
            }
        ],
    )
    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[2.0]]]),
        obs_list=[{"states": torch.tensor([[20.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[{}],
    )

    first, second = accumulator.drain_episodes()
    assert first[0]["state"].item() == 10.0
    assert first[0]["is_success"].item()
    assert second[0]["state"].item() == 100.0


def test_online_lerobot_accumulator_filters_unsuccessful_episodes():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        only_success=True,
        num_action_chunks=1,
        action_dim=1,
    )

    for terminated, truncated, success in (
        (True, False, False),
        (False, True, True),
        (True, False, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[1.0]]]),
            obs_list=[{"states": torch.tensor([[1.0]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[truncated]]),
            infos_list=[{"success_once": torch.tensor([success])}],
        )

    [episode] = accumulator.drain_episodes()
    assert episode[-1]["is_success"].item()
    assert episode[-1]["done"].item()


def test_online_lerobot_accumulator_expands_vectorized_action_chunks():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=2,
        num_action_chunks=2,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        obs_list=[
            {"states": torch.tensor([[10.0], [30.0]])},
            {"states": torch.tensor([[20.0], [40.0]])},
        ],
        terminations=torch.tensor([[False, True], [False, True]]),
        truncations=torch.zeros(2, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    episodes = accumulator.drain_episodes()
    assert len(episodes) == 2
    assert [frame["actions"].item() for frame in episodes[0]] == [1.0, 2.0]
    assert [frame["actions"].item() for frame in episodes[1]] == [3.0, 4.0]


def test_online_lerobot_accumulator_uses_policy_intervention_metadata():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=2,
        action_dim=1,
    )
    policy_output = PolicyOutput(
        forward_inputs={"action": torch.tensor([[[9.0], [8.0]]])},
        intervene_flags=torch.tensor([[False, True]]),
    )
    chunk_actions = torch.tensor([[[1.0], [2.0]]])

    accumulator.append_chunk_episode_data(
        policy_output=policy_output,
        chunk_actions=chunk_actions,
        obs_list=[
            {"states": torch.tensor([[10.0]])},
            {"states": torch.tensor([[20.0]])},
        ],
        terminations=torch.tensor([[False, True]]),
        truncations=torch.zeros(1, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    [episode] = accumulator.drain_episodes()
    assert [frame["actions"].item() for frame in episode] == [1.0, 8.0]
    assert [frame["intervene_flag"].item() for frame in episode] == [False, True]


def test_online_lerobot_accumulator_applies_recording_controls():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    for state, action, info, terminated in (
        (10.0, 1.0, {"record_reset": True}, False),
        (20.0, 2.0, {"pre_record": True}, False),
        (30.0, 3.0, {"segment_advance": True}, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[action]]]),
            obs_list=[{"states": torch.tensor([[state]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[False]]),
            infos_list=[info],
        )

    [episode] = accumulator.drain_episodes()
    assert len(episode) == 1
    assert episode[0]["state"].item() == 10.0
    assert episode[0]["actions"].item() == 3.0
    assert episode[0]["segment_id"].item() == 1


def test_lerobot_frame_owns_observation_and_image_conversion():
    frame = LeRobotFrame.from_step(
        observation={
            "main_images": torch.full((2, 2, 3), 0.5),
            "wrist_images": torch.zeros(2, 2, 2, 3),
            "states": torch.tensor([1.0, 2.0]),
            "task_descriptions": ["pick"],
        },
        action=torch.tensor([1.0, 2.0]).numpy(),
        info={
            "intervene_action": torch.tensor([9.0, 8.0]),
            "intervene_flag": torch.tensor(True),
            "success_once": torch.tensor(True),
        },
        segment_id=3,
        action_dim=2,
    )

    assert frame is not None
    output = frame.to_dict(episode_success=True, done=True)
    assert output["actions"].tolist() == [9.0, 8.0]
    assert output["image"].dtype.name == "uint8"
    assert {"wrist_image-0", "wrist_image-1"} <= output.keys()
    assert output["task"] == "pick"
    assert output["segment_id"].item() == 3


def test_offline_lerobot_export_reuses_canonical_frame_conversion():
    collector = object.__new__(CollectEpisode)
    collector.num_envs = 1
    buffer = {
        "observations": [
            {
                "main_images": torch.full((2, 2, 3), 0.5),
                "states": torch.tensor([1.0, 2.0]),
                "task_descriptions": ["pick"],
            }
        ],
        "actions": [torch.tensor([1.0, 2.0])],
        "terminated": [True],
        "infos": [
            {},
            {
                "intervene_action": torch.tensor([9.0, 8.0]),
                "intervene_flag": torch.tensor([True]),
            },
        ],
        "segment_ids": [4],
    }

    [frame] = collector._buffer_to_lerobot_ep(buffer, env_idx=0, is_success=True)
    assert frame["actions"].tolist() == [9.0, 8.0]
    assert frame["image"].dtype.name == "uint8"
    assert frame["task"] == "pick"
    assert frame["segment_id"].item() == 4
    assert frame["done"].item()


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_policy_input_compression_preserves_attached_env_parts(monkeypatch):
    """Trajectory requests remain lossless across split/compress/merge routing."""
    monkeypatch.setattr(
        obs_compression,
        "_get_backend",
        lambda _codec: (
            lambda raw, _level: raw,
            lambda raw: raw,
        ),
    )
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key=key, size=4)
    current_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    next_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    policy_input = PolicyInput(
        obs={"main_images": current_images},
        sources=[source],
        env_parts=[
            EnvPart(
                sources=[source],
                transition=EnvTransition(rewards=torch.ones(4, 2)),
                next_obs={"main_images": next_images},
            )
        ],
    )
    env_worker = object.__new__(EnvWorker)
    env_worker.obs_compression_cfg = {
        "enable": True,
        "codec": "lz4",
        "level": 1,
        "xor_delta": True,
    }

    shards = env_worker._split_and_compress_policy_input(policy_input, [1, 3])

    assert is_compressed_image(shards[0].obs["main_images"])
    assert is_compressed_image(shards[0].env_parts[0].next_obs["main_images"])
    merged = MultiStepRolloutWorker._merge_policy_inputs(shards)
    assert torch.equal(merged.obs["main_images"], current_images)
    assert torch.equal(
        torch.cat([part.next_obs["main_images"] for part in merged.env_parts], dim=0),
        next_images,
    )


def test_sync_and_async_workers_share_the_part_channel_contract():
    """Only Rollout publishes parts; Env communicates exclusively with Rollout."""
    assert "actor_channel" not in inspect.signature(EnvWorker.interact).parameters
    assert "actor_channel" not in inspect.signature(AsyncEnvWorker.interact).parameters
    assert (
        "actor_channel" in inspect.signature(MultiStepRolloutWorker.generate).parameters
    )
    assert (
        "actor_channel"
        in inspect.signature(AsyncMultiStepRolloutWorker.generate).parameters
    )


def test_decoupled_evaluation_service_is_reused_and_cancelled_on_stop():
    async def run():
        worker = object.__new__(AsyncMultiStepRolloutWorker)
        worker.env_decoupled_mode = True
        worker._generate_task = None
        worker._evaluate_task = None
        blocker = asyncio.Event()
        calls = 0

        async def serve(_input_channel, _output_channel):
            nonlocal calls
            calls += 1
            await blocker.wait()

        worker._run_evaluate_service = serve
        await worker.ensure_evaluate_service(Mock(), Mock())
        first_task = worker._evaluate_task
        await worker.ensure_evaluate_service(Mock(), Mock())

        assert worker._evaluate_task is first_task
        assert calls == 1

        worker.stop()
        await asyncio.sleep(0)
        assert first_task.cancelled()

    asyncio.run(run())


def test_async_runner_reuses_decoupled_evaluation_service_across_validations():
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"enable_decoupled_mode": True}})
    runner.env_channel = Mock()
    runner.rollout_channel = Mock()
    runner.rollout = Mock()
    runner.rollout.ensure_evaluate_service.return_value = Mock(
        wait=Mock(return_value=None)
    )
    runner.env = Mock()
    runner.env.evaluate.return_value = Mock(wait=Mock(return_value=[{}]))

    with patch(
        "rlinf.runners.async_embodied_runner.compute_evaluate_metrics",
        return_value={},
    ):
        runner.evaluate()
        runner.evaluate()

    assert runner.rollout.ensure_evaluate_service.call_count == 2
    runner.rollout.evaluate.assert_not_called()


def test_decoupled_policy_route_round_trip():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"policy": ["stale"]}
    rollout.recv_from_and_record_batch_routes_with_timeout = AsyncMock(
        return_value=(_policy_input(), [8, 8])
    )
    rollout.send_to_recorded_batch_routes = Mock()

    policy_input, split_sizes = asyncio.run(
        rollout._receive_policy_input(None, "policy", 0)
    )
    source = torch.zeros(16, 4, requires_grad=True)
    output = source * 2
    rollout._send_actions(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=rollout._merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    sent_output = rollout.send_to_recorded_batch_routes.call_args.kwargs["data"]
    assert torch.equal(sent_output, output)
    assert not sent_output.requires_grad
    assert sent_output.grad_fn is None
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=sent_output,
        tag="policy",
        split_fn=rollout._split_actions,
        split_sizes=[8, 8],
    )


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=torch.zeros(8, 4))

    env._recv_actions(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_action_batch_size,
        decoupled_mode=True,
    )


def test_smooth_intervention_builds_external_input_without_policy_history():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    env = SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0, 3.0]])

    policy_input = controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    assert isinstance(policy_input, PolicyInput)
    assert not policy_input.requires_inference
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )


def test_smooth_intervention_requires_online_lerobot_dagger():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 2, "action_dim": 3}},
            "algorithm": {
                "loss_type": "ppo",
                "dagger": {"online_lerobot": {"enabled": True}},
            },
            "env": {
                "train": {
                    "smooth_intervene": True,
                    "env_type": "realworld",
                    "use_pico": True,
                    "use_spacemouse": False,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="loss_type=embodied_dagger"):
        SmoothInterveneController.from_cfg(
            cfg,
            stage_num=1,
            enable_train=True,
            train_num_envs_per_stage=1,
        )


def test_env_sends_external_request_after_intervention_continues():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = True
    env.env_decoupled_mode = False
    env.env_list = [
        SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0]])
    ]
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 2, enabled=True)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(
        obs={"states": torch.zeros(1, 4)},
        transition=EnvTransition(
            dones=torch.tensor([[False, False]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, {}, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert not policy_input.requires_inference
    assert policy_input.env_parts[0] is not None
    assert policy_input.env_parts[0].next_obs is None
    assert torch.equal(policy_input.obs["states"], env_output.obs["states"])
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
    )


def _publish_step_env(**overrides):
    """Build the minimal EnvWorker used by _publish_step tests."""
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = False
    env.env_decoupled_mode = False
    env.collect_final_values = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    for name, value in overrides.items():
        setattr(env, name, value)
    return env


def _terminal_env_output(reset_obs, terminal_obs, **transition):
    flags = {"dones": torch.ones(1, 1, dtype=torch.bool), **transition}
    return EnvOutput(
        obs={"states": reset_obs},
        final_obs={"states": terminal_obs},
        transition=EnvTransition(**flags),
    )


def test_env_sends_terminal_observation_only_as_a_boundary_override():
    env = _publish_step_env()
    reset_obs = torch.zeros(1, 4)
    terminal_obs = torch.ones(1, 4)
    env_output = _terminal_env_output(
        reset_obs,
        terminal_obs,
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, None, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert torch.equal(policy_input.obs["states"], reset_obs)
    assert torch.equal(policy_input.env_parts[0].next_obs["states"], terminal_obs)
    assert policy_input.env_parts[0].requires_inference


def test_rollout_completes_each_epoch_through_policy_inputs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        side_effect=[
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[1]])},
            ),
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[2]])},
            ),
        ]
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[2]])},
        requires_inference=False,
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[3]])},
        requires_inference=True,
    )
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.tensor([[1]])},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[2]])},
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[3]])},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        side_effect=[
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[1]])}}),
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[2]])}}),
            (
                torch.zeros(1, 1),
                {
                    "forward_inputs": {"states": torch.tensor([[3]])},
                    "prev_values": torch.tensor([[4.0, 5.0]]),
                },
            ),
        ]
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(
        generate_one_epoch(
            rollout,
            input_channel=Mock(),
            output_channel=Mock(),
            actor_channel=actor_channel,
        )
    )

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    assert sum(isinstance(part, PolicyPart) for part in parts) == 2
    env_parts = [part for part in parts if isinstance(part, EnvPart)]
    assert len(env_parts) == 2
    assert env_parts[0].initial_transition is completion0.initial_transition
    assert env_parts[0].next_rlt_obs is None
    assert env_parts[0].next_obs is None
    assert torch.equal(env_parts[1].bootstrap_values, torch.tensor([[4.0]]))
    assert torch.equal(env_parts[1].final_prev_values, torch.tensor([[4.0, 5.0]]))
    assert rollout._predict_rollout_actions.call_count == 3


def test_rollout_routes_external_input_without_model_inference():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        return_value=PolicyOutput(
            forward_inputs={"action": torch.ones(1, 6)},
        )
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    dummy_actions = torch.full((1, 2, 3), 2.0)
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.zeros(1, 4)},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    external_actions=dummy_actions,
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.ones(1, 2, 3),
            {"forward_inputs": {"action": torch.ones(1, 6)}},
        )
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(generate_one_epoch(rollout, Mock(), Mock(), actor_channel))

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    inferred_parts = [
        part for part in parts if isinstance(part, PolicyPart) and part.inferred
    ]
    assert len(inferred_parts) == 1
    external_part = next(
        part for part in parts if isinstance(part, PolicyPart) and not part.inferred
    )
    assert torch.equal(external_part.external_actions, dummy_actions)
    assert rollout._predict_rollout_actions.call_count == 1
    assert torch.equal(rollout._send_actions.call_args_list[1].args[1], dummy_actions)


def test_decoupled_final_completion_uses_next_bootstrap():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.stage_num = 1
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 1
    env.enable_online_lerobot = False
    env.env_decoupled_mode = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(obs={"states": torch.zeros(1, 4)})

    completion = env._publish_step(
        rollout_channel=Mock(),
        env_output=env_output,
        initial_transition=EnvTransition(),
        reward_model_output=None,
        chunk_step_data=None,
        epoch_id=0,
        chunk_id=0,
        stage_id=0,
    )

    assert completion is not None
    assert completion.initial_transition is not None
    assert torch.equal(completion.next_obs["states"], env_output.obs["states"])
    env._send_policy_input.assert_not_called()

    env._send_train_bootstrap(
        rollout_channel=Mock(),
        env_outputs=[env_output],
        step_id=1,
        epoch_id=0,
        previous_env_parts={0: completion},
    )

    policy_input = env._send_policy_input.call_args.args[1]
    assert policy_input.env_parts == [completion]


def test_env_closes_reward_stream_with_final_chunk():
    env = object.__new__(EnvWorker)
    env.rollout_epoch = 2
    env.stage_num = 1
    env.n_train_chunk_steps = 2
    env._prefetched_train_bootstrap = None
    env._trajectory_step = 0
    env._rank = 0
    env.env_list = [SimpleNamespace()]
    env.use_training_pipeline = False
    env.env_decoupled_mode = False
    env.enable_online_lerobot = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_actions = Mock(return_value=torch.zeros(1, 4))
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 4)
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                transition=EnvTransition(rewards=torch.zeros(1, 1)),
            ),
            {},
            {},
        )
    )
    env.get_reward_model_output = Mock(return_value=torch.zeros(1, 1))
    env._publish_step = Mock()
    env.record_env_metrics = Mock()
    env.store_last_obs_and_intervened_info = Mock()
    env.finish_rollout = Mock()
    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]


def test_smooth_intervention_holds_at_the_last_commanded_step():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    seen = []

    def get_hold_actions(fallback=None):
        seen.append(fallback)
        return [[1.0, 2.0, 3.0]]

    env = SimpleNamespace(get_hold_actions=get_hold_actions)
    controller.remember_actions(0, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))

    controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    # The wrapper receives the final step of the chunk that was just executed,
    # so it can hold that pose instead of snapping back to its own default.
    assert len(seen) == 1
    assert seen[0].tolist() == [[4.0, 5.0, 6.0]]


def test_rollout_runs_requested_terminal_inference_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "prev_values": torch.zeros(1, 1),
                "forward_inputs": {"states": torch.ones(1, 1)},
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once_with(completion.next_obs)
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.bootstrap_values, torch.zeros(1, 1))
    assert torch.equal(part.final_prev_values, torch.zeros(1, 1))
    assert part.next_rlt_obs is None
    assert part.next_obs is None


def test_rollout_reuses_policy_obs_for_transition_next_obs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = True
    actor_channel = Mock()
    policy_obs = {"states": torch.ones(1, 4)}
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
    )

    rollout._publish_env_part(
        completion,
        policy_obs,
        {"unused": torch.ones(1, 1)},
        actor_channel,
    )

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is policy_obs
    assert part.next_rlt_obs is None


def test_rollout_runs_terminal_inference_for_rlt_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    rollout.hf_model = SimpleNamespace()
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "forward_inputs": {
                    "rlt_transition_z_rl": torch.ones(1, 2),
                    "rlt_transition_proprio": torch.ones(1, 3),
                    "rlt_transition_ref_chunk": torch.ones(1, 4),
                }
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once()
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.next_rlt_obs["z_rl"], torch.ones(1, 2))


def test_rollout_keeps_next_forward_inputs_only_for_rlt():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    actor_channel = Mock()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy_input = PolicyInput(
        obs={"states": torch.zeros(1, 4)},
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(key, 1)],
                transition=EnvTransition(),
            )
        ],
        request_sizes=[1],
    )
    forward_inputs = {
        "rlt_transition_z_rl": torch.ones(1, 2),
        "rlt_transition_proprio": torch.ones(1, 3),
        "rlt_transition_ref_chunk": torch.ones(1, 4),
        "unrelated": torch.ones(1, 8),
    }

    rollout._publish_env_parts(policy_input, forward_inputs, actor_channel)

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is None
    assert torch.equal(
        part.next_rlt_obs["z_rl"],
        forward_inputs["rlt_transition_z_rl"],
    )
    assert set(part.next_rlt_obs) == {"z_rl", "proprio", "ref_chunk"}


def test_vlm_trend_batch_video_metadata_stays_nested_per_sample():
    pytest.importorskip("transformers.video_utils")
    from transformers.video_utils import VideoMetadata

    from rlinf.data.datasets.vlm.vlm_trend_reward import VLMTrendRewardSFTDataset

    captured = {}

    class _Processor:
        video_token = "<|video_pad|>"

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured["videos_kwargs"] = kwargs["videos_kwargs"]
            batch = len(kwargs["text"])
            return {
                "input_ids": torch.zeros(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    VLMTrendRewardSFTDataset.process_inputs(
        processor=_Processor(),
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[["task a"], ["task b"]],
        videos=[
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        ],
        answer_text=None,
        video_fps=24.0,
    )

    metadata = captured["videos_kwargs"]["video_metadata"]
    assert len(metadata) == 2
    assert all(len(sample) == 2 for sample in metadata)
    assert all(
        isinstance(item, VideoMetadata) for sample in metadata for item in sample
    )
    assert metadata[0][0].total_num_frames == 5
    assert metadata[0][0].frames_indices == [0, 1, 2, 3, 4]
    assert metadata[1][1].total_num_frames == 5


@pytest.fixture
def umi_v4_data_config(tmp_path, monkeypatch):
    pytest.importorskip("openpi")
    sentencepiece = pytest.importorskip("sentencepiece")
    from openpi.shared import download

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_bytes = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(
            ["Task: pick cube, State: [0 1 2 3 4 5 6 7 8 9]; Action:"]
        ),
        model_writer=model_bytes,
        model_type="char",
        vocab_size=64,
        hard_vocab_limit=False,
        num_threads=1,
        minloglevel=2,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(model_bytes.getvalue())
    monkeypatch.setattr(
        download, "maybe_download", lambda *args, **kwargs: tokenizer_path
    )
    stats = {
        key: {
            "min": [-2.0] * dim,
            "max": [2.0] * dim,
            "q01": [-0.1] * dim,
            "q99": [0.1] * dim,
        }
        for key, dim in (("state", 16), ("actions", 10))
    }
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(json.dumps({"norm_stats": stats}))
    config = get_openpi_config(
        "pi05_umi_v4",
        repo_id=str(tmp_path / "dataset"),
        data_kwargs={"norm_stats_path": str(stats_path)},
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    processor = sentencepiece.SentencePieceProcessor(model_proto=model_bytes.getvalue())
    return data_config, processor


def _umi_v4_sample():
    return {
        "observation.robot0_eef_pos": np.array(
            [[1, -1, 0], [0.5, 0, -0.5]], dtype=np.float32
        ),
        "observation.robot0_eef_rot_axis_angle": np.ones((2, 6), dtype=np.float32),
        "observation.robot0_eef_rot_axis_angle_wrt_start": np.zeros(
            (2, 6), dtype=np.float32
        ),
        "observation.robot0_gripper_width": np.full((2, 1), 0.08, dtype=np.float32),
        "observation.left_wrist_0_rgb_1": np.full((3, 224, 224), 0.5, dtype=np.float32),
        "actions": np.arange(160, dtype=np.float32).reshape(16, 10) / 100,
        "task": "pick_cube",
    }


def test_umi_v4_official_loader_preserves_history_and_action_contract(
    umi_v4_data_config,
):
    from openpi.training.data_loader import transform_dataset
    from openpi.transforms import Unnormalize, compose

    data_config, processor = umi_v4_data_config
    sample = _umi_v4_sample()
    result = transform_dataset([sample], data_config)[0]
    assert data_config.action_sequence_keys == ()
    assert data_config.norm_stats == {}
    assert result["state"].shape == (32,)
    state = result["state"].reshape(2, 16)
    np.testing.assert_array_equal(
        state[:, :3], sample["observation.robot0_eef_pos"] / 2
    )
    np.testing.assert_array_equal(state[:, 3:9], np.ones((2, 6)))
    np.testing.assert_array_equal(
        state[:, -1:], sample["observation.robot0_gripper_width"]
    )
    decoded = processor.decode(
        result["tokenized_prompt"][result["tokenized_prompt_mask"]].tolist()
    )
    # History rows must remain separate bracketed arrays during tokenization.
    assert decoded.count("[") == decoded.count("]") == 2
    assert "pick cube" in decoded
    assert result["actions"].shape == (16, 32)
    np.testing.assert_array_equal(
        result["actions"][:, :3], sample["actions"][:, :3] / 2
    )
    np.testing.assert_array_equal(result["actions"][:, 3:10], sample["actions"][:, 3:])
    np.testing.assert_array_equal(result["actions"][:, 10:], 0)
    assert result["image_mask"] == {
        "base_0_rgb": False,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], 127)
    output = compose(
        [Unnormalize(data_config.norm_stats), *data_config.data_transforms.outputs]
    )({"actions": result["actions"]})
    np.testing.assert_array_equal(output["actions"], sample["actions"])


@pytest.mark.parametrize("use_minmax", [True, False])
def test_umi_v4_normalization_constant_xyz_and_unscaled_gripper(use_minmax):
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.umi_policy import UmiV4Normalize

    stats = {}
    for key, dim in (("state", 16), ("actions", 10)):
        lower = [3.0, -2.0, 1.0] + [-5.0] * (dim - 3)
        upper = [3.0, 2.0, 1.0 + 1e-8] + [5.0] * (dim - 3)
        stats[key] = {"q01": lower, "q99": upper}
        if use_minmax:
            stats[key].update(min=lower, max=upper)
    normalizer = UmiV4Normalize.from_stats(stats)
    state = np.array([[3.0, 1.0, 1.0] + [0.8] * 13] * 2)
    result = normalizer({"state": state})["state"]
    np.testing.assert_array_equal(result[:, :3], [[0.0, 0.5, 0.0]] * 2)
    np.testing.assert_array_equal(result[:, 3:], state[:, 3:])


def test_umi_v4_rejects_double_chunked_actions(umi_v4_data_config):
    from openpi.training.data_loader import transform_dataset

    data_config, _ = umi_v4_data_config
    sample = _umi_v4_sample()
    sample["actions"] = np.zeros((16, 16, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="already contain one"):
        transform_dataset([sample], data_config)[0]


class _OpenPiResumeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 120

    def __getitem__(self, index):
        return {
            "state": np.asarray([index], dtype=np.float32),
            "image": {"base_0_rgb": np.zeros((8, 8, 3), dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.True_},
            "actions": np.asarray([[index]], dtype=np.float32),
        }


def _make_openpi_resume_loader(rank, batch_size):
    loader_module = pytest.importorskip("openpi.training.data_loader")
    config_module = pytest.importorskip("openpi.training.config")
    dataset = _OpenPiResumeDataset()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=2, rank=rank, shuffle=True, seed=0
    )
    inner = loader_module.TorchDataLoader(
        dataset, batch_size, sampler=sampler, seed=42, framework="pytorch"
    )
    return loader_module.DataLoaderImpl(
        config_module.DataConfig(repo_id="umi-test"), inner
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("batch_size", [1, 3, 9])
def test_openpi_resume_preserves_next_samples_when_batch_changes(rank, batch_size):
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_openpi_sft_data_order,
        restore_openpi_sft_iterator,
    )

    original = _make_openpi_resume_loader(rank, 1)
    iterator = iter(original)
    for _ in range(18):
        next(iterator)
    expected = torch.cat([next(iterator)[1].flatten() for _ in range(9)])
    state = get_openpi_sft_data_order(original)
    state.update(consumed_samples=18, data_epoch=0)
    resumed = _make_openpi_resume_loader(rank, batch_size)
    recovered, skipped = restore_openpi_sft_iterator(resumed, state)
    actual = torch.cat([next(recovered)[1].flatten() for _ in range(9 // batch_size)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert skipped == 18 // batch_size


@pytest.mark.parametrize("changed", ["dataset", "sampler", "unaligned", "past_epoch"])
def test_openpi_resume_rejects_incompatible_data_order(changed):
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_openpi_sft_data_order,
        restore_openpi_sft_iterator,
    )

    loader = _make_openpi_resume_loader(0, 3)
    state = get_openpi_sft_data_order(loader)
    state.update(consumed_samples=18, data_epoch=0)
    if changed == "dataset":
        state["dataset_repo_id"] = "different-dataset"
    elif changed == "sampler":
        state["sampler"]["seed"] = 3
    elif changed == "unaligned":
        state["consumed_samples"] = 19
    else:
        state["consumed_samples"] = 63
    with pytest.raises(ValueError):
        restore_openpi_sft_iterator(loader, state)


class _OpenPiEpisodeDataset(torch.utils.data.Dataset):
    """In-memory LeRobot boundary with observable episode and original row IDs."""

    def __init__(self, lengths=(7, 8, 9, 10, 11, 12)):
        ends = np.cumsum(lengths)
        self.episode_data_index = {
            "from": torch.tensor(np.r_[0, ends[:-1]]),
            "to": torch.tensor(ends),
        }
        self.meta = SimpleNamespace(
            episodes={i: {"length": length} for i, length in enumerate(lengths)}
        )
        self.episodes = np.repeat(np.arange(len(lengths)), lengths)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        index = int(index)
        return {
            "state": np.asarray([self.episodes[index], index], dtype=np.float32),
            "image": {"base_0_rgb": np.zeros((8, 8, 3), dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.True_},
            "actions": np.asarray([[self.episodes[index], index]], dtype=np.float32),
        }


@pytest.fixture
def openpi_episode_loader(monkeypatch, tmp_path):
    """Use real OpenPI transforms, batching and distributed samplers."""
    from omegaconf import OmegaConf

    module = pytest.importorskip("openpi.training.data_loader")
    config_module = pytest.importorskip("openpi.training.config")
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )
    from rlinf.models.embodiment.openpi import dataconfig

    dataset = _OpenPiEpisodeDataset()
    data_config = config_module.DataConfig(
        repo_id=str(tmp_path), norm_stats={}, action_sequence_keys=()
    )

    @dataclasses.dataclass
    class TestConfig:
        data: object
        model: object
        assets_dirs: object
        num_workers: int = 0
        seed: int = 11

    official = TestConfig(
        data=SimpleNamespace(create=lambda *_: data_config),
        model=SimpleNamespace(action_horizon=1, action_dim=2),
        assets_dirs=tmp_path,
    )
    monkeypatch.setattr(dataconfig, "get_openpi_config", lambda *a, **k: official)
    # The external dataset factory opens the full dataset once; Subset and all
    # subsequent transforms/sampling use production implementations.
    monkeypatch.setattr(
        module,
        "create_torch_dataset",
        lambda *a: module.TransformedDataset(dataset, []),
    )
    cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 3,
                "eval_batch_size": 3,
                "seed": 11,
                "model": {
                    "model_type": "openpi_rlinf",
                    "model_path": str(tmp_path),
                    "num_action_chunks": 1,
                    "openpi": {"config_name": "test", "model_action_dim": 2},
                },
            },
            "data": {
                "num_workers": 0,
                "train_data_paths": str(tmp_path),
                "val_data_paths": str(tmp_path),
                "train_episode_ids": [0, 2, 4],
                "val_episode_ids": [1, 3, 5],
                "val_frames_per_episode": 4,
                "val_num_batches": None,
                "eval_seed": 17,
            },
        }
    )

    def build(rank=0, world_size=2, evaluate=False):
        return build_official_openpi_sft_dataloader(
            cfg, world_size, rank, str(tmp_path), eval_dataset=evaluate
        )[0]

    return cfg, dataset, build


def test_openpi_episode_split_preserves_original_frames_and_rank_disjointness(
    openpi_episode_loader,
):
    from itertools import islice

    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_official_openpi_sft_num_batches,
    )

    _, dataset, build = openpi_episode_loader
    rank_rows = []
    for rank in range(2):
        loader = build(rank)
        batches = list(islice(loader, get_official_openpi_sft_num_batches(loader)))
        actions = torch.cat([batch[1][:, 0] for batch in batches])
        assert set(actions[:, 0].tolist()) <= {0, 2, 4}
        rows = actions[:, 1].long().tolist()
        assert len(set(rows)) == len(rows)
        assert loader.episode_selection["selected_frame_count"] == 27
        assert loader.episode_selection["source_frame_count"] == len(dataset)
        assert loader.episode_selection["tail_frames_dropped"] == 3
        rank_rows.append(set(rows))
    assert rank_rows[0].isdisjoint(rank_rows[1])
    assert len(rank_rows[0] | rank_rows[1]) == 24


def test_openpi_validation_is_finite_repeatable_and_interleaves_episodes(
    openpi_episode_loader,
):
    _, dataset, build = openpi_episode_loader
    combined = set()
    for rank in range(2):
        loader = build(rank, evaluate=True)
        batches = list(loader)
        assert len(batches) == 2
        repeated = list(loader)
        for first, second in zip(batches, repeated, strict=True):
            torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
            episodes = first[1][:, 0, 0]
            assert torch.all(episodes != episodes.roll(1))
            assert set(episodes.tolist()) == {1, 3, 5}
        indices = torch.cat([item[1][:, 0, 1] for item in batches]).long().tolist()
        assert combined.isdisjoint(indices)
        combined.update(indices)
        assert loader.episode_selection["cross_episode_shuffle_verified"]
        assert loader.episode_selection["tail_frames_dropped"] == 0
    expected = set()
    for episode in (1, 3, 5):
        start = int(dataset.episode_data_index["from"][episode])
        end = int(dataset.episode_data_index["to"][episode])
        expected.update(
            (np.linspace(0, end - start - 1, 4, dtype=np.int64) + start).tolist()
        )
    assert combined == expected


def test_openpi_validation_budget_does_not_cycle_dataset(openpi_episode_loader):
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_official_openpi_sft_num_batches,
    )

    cfg, _, build = openpi_episode_loader
    cfg.data.val_num_batches = 1
    loader = build(evaluate=True)
    assert len(list(loader)) == get_official_openpi_sft_num_batches(loader) == 1
    cfg.data.val_num_batches = 100
    loader = build(evaluate=True)
    assert len(list(loader)) == get_official_openpi_sft_num_batches(loader) == 2


def test_openpi_episode_ids_can_be_loaded_from_json(openpi_episode_loader, tmp_path):
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_openpi_sft_data_order,
    )

    cfg, _, build = openpi_episode_loader
    original = get_openpi_sft_data_order(build())
    path = tmp_path / "train_episode_ids.json"
    path.write_text(json.dumps([0, 2, 4]))
    cfg.data.train_episode_ids = str(path)
    assert get_openpi_sft_data_order(build()) == original


def test_openpi_episode_subsets_work_with_huggingface_integer_keys(
    openpi_episode_loader, monkeypatch
):
    datasets = pytest.importorskip("datasets")
    _, dataset, build = openpi_episode_loader
    storage = datasets.Dataset.from_dict({"original_index": list(range(len(dataset)))})
    read_observation = dataset.__getitem__

    def read_huggingface_row(self, index):
        # Exercise the real Arrow-backed Dataset key validation before the
        # small LeRobot stand-in turns the row into observation tensors.
        row = storage[index]
        return read_observation(row["original_index"])

    monkeypatch.setattr(_OpenPiEpisodeDataset, "__getitem__", read_huggingface_row)
    for rank in range(2):
        train_actions = next(iter(build(rank)))[1]
        assert set(train_actions[:, 0, 0].tolist()) <= {0, 2, 4}
        val_batches = list(build(rank, evaluate=True))
        assert len(val_batches) == 2
        for _, actions in val_batches:
            assert set(actions[:, 0, 0].tolist()) == {1, 3, 5}


@pytest.mark.parametrize(
    "case", ["overlap", "duplicate", "unknown", "too_short", "donor"]
)
def test_openpi_episode_selection_rejects_invalid_splits(openpi_episode_loader, case):
    cfg, _, build = openpi_episode_loader
    world_size = 2
    if case == "overlap":
        cfg.data.val_episode_ids = [0, 1, 3]
    elif case == "duplicate":
        cfg.data.val_episode_ids = [1, 1, 3]
    elif case == "unknown":
        cfg.data.val_episode_ids = [1, 3, 60]
    elif case == "too_short":
        cfg.data.val_frames_per_episode = 20
    else:
        world_size = 3
    with pytest.raises(ValueError):
        build(world_size=world_size, evaluate=True)


def test_openpi_resume_rejects_same_size_different_episode_selection(
    openpi_episode_loader,
):
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        get_openpi_sft_data_order,
        restore_openpi_sft_iterator,
    )

    cfg, _, build = openpi_episode_loader
    loader = build()
    saved = get_openpi_sft_data_order(loader)
    saved.update(consumed_samples=6, data_epoch=0)
    # The new episodes have the same 27 frames, but a different underlying order.
    cfg.data.train_episode_ids = [0, 1, 5]
    cfg.data.val_episode_ids = [2, 3, 4]
    with pytest.raises(ValueError, match="episode selection"):
        restore_openpi_sft_iterator(build(), saved)


def test_arx_transition_split_merge_preserves_executed_actions_and_validity():
    executed = torch.arange(60, dtype=torch.float32).reshape(3, 20)
    valid = torch.tensor([[True], [False], [True]])
    transition = EnvTransition(
        executed_actions=executed,
        transition_valid=valid,
        rewards=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        dones=torch.tensor([[False, False], [False, True], [False, False]]),
    )
    shards = transition.split([1, 2])
    torch.testing.assert_close(shards[1].executed_actions, executed[1:])
    reconstructed = EnvTransition.merge(shards)
    for field in ("executed_actions", "transition_valid", "rewards", "dones"):
        torch.testing.assert_close(
            getattr(reconstructed, field), getattr(transition, field)
        )
    exposed = reconstructed.to_dict()
    torch.testing.assert_close(exposed["executed_actions"], executed)
    torch.testing.assert_close(exposed["transition_valid"], valid)


def _arx_trajectory_parts(preserve_reference=True):
    batch, chunk, dim = 2, 2, 10
    proposed = torch.zeros(batch, chunk * dim)
    executed = torch.full_like(proposed, 0.3)
    human = torch.full_like(proposed, 0.7)
    reference = torch.full((batch, 16, dim), 0.2)
    inputs = {
        "action": proposed,
        "model_action": proposed,
        "z_rl": torch.ones(batch, 4),
        "proprio": torch.ones(batch, 32),
        "ref_chunk": reference,
    }
    if preserve_reference:
        inputs["preserve_rlt_reference"] = torch.tensor([[True], [False]])
    policy = PolicyPart(
        sources=[_source(TrajectoryKey(0, 0, 0, 0, 0), size=batch)],
        obs={"states": torch.zeros(batch, 32)},
        output=PolicyOutput(forward_inputs=inputs),
    )
    env = EnvPart(
        sources=policy.sources,
        transition=EnvTransition(
            rewards=torch.ones(batch, chunk),
            dones=torch.zeros(batch, chunk, dtype=torch.bool),
            executed_actions=executed,
            transition_valid=torch.tensor([[True], [False]]),
            intervene_actions=human,
            intervene_flags=torch.tensor([[False, True], [True, False]]),
        ),
        next_rlt_obs={
            "z_rl": torch.full((batch, 4), 2.0),
            "proprio": torch.full((batch, 32), 2.0),
            "ref_chunk": torch.full((batch, 16, dim), 0.4),
        },
    )
    return policy, env


@pytest.mark.parametrize("preserve_reference", [False, True])
def test_arx_trajectory_actual_action_and_human_target_preserve_reference(
    preserve_reference,
):
    policy, env = _arx_trajectory_parts(preserve_reference)
    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=False,
        collect_transitions=True,
        enable_rlt=True,
        include_final_value=False,
    )
    expected = torch.tensor([[0.3] * 10 + [0.7] * 10, [0.7] * 10 + [0.3] * 10])
    torch.testing.assert_close(step.actions, expected)
    torch.testing.assert_close(step.forward_inputs["action"], expected)
    assert "model_action" not in step.forward_inputs
    assert step.intervene_flags[0].tolist() == [False] * 10 + [True] * 10
    assert step.intervene_flags[1].tolist() == [True] * 10 + [False] * 10
    torch.testing.assert_close(
        step.forward_inputs["transition_valid"], torch.tensor([[True], [False]])
    )
    if preserve_reference:
        torch.testing.assert_close(
            step.curr_obs["ref_chunk"][0], policy.output.forward_inputs["ref_chunk"][0]
        )
    else:
        torch.testing.assert_close(
            step.curr_obs["ref_chunk"][0, 1], torch.full((10,), 0.7)
        )
    # Mixed batches retain legacy intervention replacement for the other row.
    torch.testing.assert_close(step.curr_obs["ref_chunk"][1, 0], torch.full((10,), 0.7))
    torch.testing.assert_close(
        step.next_obs["ref_chunk"], env.next_rlt_obs["ref_chunk"]
    )


def test_arx_observation_fields_survive_env_and_policy_communication():
    raw = {
        "states": torch.zeros(2, 32),
        "main_images": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
        "task_descriptions": ["pick", "place"],
        "robot0_eef_pos": torch.randn(2, 2, 3),
        "robot0_eef_rot_axis_angle": torch.randn(2, 2, 6),
        "robot0_eef_rot_axis_angle_wrt_start": torch.randn(2, 2, 6),
        "robot0_gripper_width": torch.rand(2, 2, 1),
        "left_wrist_0_rgb_1": torch.randint(
            0, 255, (2, 224, 224, 3), dtype=torch.uint8
        ),
    }
    output = EnvOutput(
        obs=raw,
        final_obs={
            key: value.clone() if torch.is_tensor(value) else list(value)
            for key, value in raw.items()
        },
    )
    for observations in (output.to_dict()["obs"], output.to_dict()["final_obs"]):
        for key in raw:
            if torch.is_tensor(raw[key]):
                torch.testing.assert_close(observations[key], raw[key])
        policy = PolicyInput(obs=observations)
        restored = PolicyInput.merge(policy.split([1, 1]))
        for key in raw:
            if torch.is_tensor(raw[key]):
                torch.testing.assert_close(restored.obs[key], raw[key])


def _arx_replay_trajectory(with_bootstrap=False):
    # Each row has a distinct identity so row shifts cannot pass unnoticed.
    identity = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1)
    initial = torch.ones(1, 2, 1, dtype=torch.bool)
    terminal = torch.zeros(3, 2, 1, dtype=torch.bool)
    terminal[1, 1] = True
    valid = torch.tensor([True, False, False, True, True, False]).reshape(3, 2, 1)
    human = torch.tensor([False, True, False, True, False, False]).reshape(3, 2, 1)
    prev_values = identity + 10
    if with_bootstrap:
        prev_values = torch.cat((prev_values, torch.full((1, 2, 1), 999.0)))
    return Trajectory(
        actions=identity.expand(3, 2, 10).clone(),
        rewards=identity + 0.5,
        dones=torch.cat((initial, terminal)),
        terminations=torch.cat((initial, terminal)),
        truncations=torch.zeros(4, 2, 1, dtype=torch.bool),
        intervene_flags=human.expand(3, 2, 10).clone(),
        prev_values=prev_values,
        forward_inputs={
            "transition_valid": valid,
            "action": identity.expand(3, 2, 10).clone(),
        },
        curr_obs={
            "z_rl": identity,
            "proprio": identity + 20,
            "ref_chunk": identity.expand(3, 2, 160).clone(),
        },
        next_obs={
            "z_rl": identity + 100,
            "proprio": identity + 120,
            "ref_chunk": (identity + 100).expand(3, 2, 160).clone(),
        },
    )


@pytest.fixture
def arx_replay_ingestion(monkeypatch):
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.workers.actor import fsdp_rlt_ac_policy_worker as worker_module

    class Ingestor(worker_module.RLTACReplayMixin):
        pass

    ingestor = Ingestor()
    ingestor.cfg = OmegaConf.create(
        {
            "env": {"train": {"env_type": "real"}},
            "algorithm": {"replay_buffer": {"filter_invalid_transitions": True}},
        }
    )
    ingestor.replay_buffer = TrajectoryReplayBuffer(seed=123)
    ingestor.demo_buffer = TrajectoryReplayBuffer(seed=123)
    owned_buffers = (ingestor.replay_buffer, ingestor.demo_buffer)
    # Single-rank metric reduction; storage and ingestion use their real code.
    monkeypatch.setattr(worker_module, "all_reduce_dict", lambda values: values)
    yield ingestor
    for buffer in owned_buffers:
        buffer.close()


def _sample_arx_replay(buffer):
    # The buffer caps each draw to its stored row count and samples with
    # replacement; repeat seeded public draws to cover the small fixture.
    return merge_batch_values([buffer.sample(num_chunks=64) for _ in range(16)])


@pytest.mark.parametrize("with_bootstrap", [False, True])
def test_arx_replay_filters_invalid_rows_without_shifting_next_state_or_terminal(
    arx_replay_ingestion, with_bootstrap
):
    ingestor = arx_replay_ingestion
    added, completed = ingestor._ingest_rollout_trajectories(
        [_arx_replay_trajectory(with_bootstrap)]
    )
    assert (added, completed) == (3, 1)
    sample = _sample_arx_replay(ingestor.replay_buffer)
    ids = sample["actions"][:, 0]
    assert set(ids.tolist()) == {0.0, 3.0, 4.0}
    torch.testing.assert_close(sample["curr_obs"]["z_rl"][:, 0], ids)
    torch.testing.assert_close(sample["next_obs"]["z_rl"][:, 0], ids + 100)
    torch.testing.assert_close(sample["rewards"][:, 0], ids + 0.5)
    torch.testing.assert_close(sample["prev_values"][:, 0], ids + 10)
    torch.testing.assert_close(sample["dones"][:, 0], ids == 3)
    assert sample["forward_inputs"]["transition_valid"].all()
    demo = ingestor.demo_buffer.sample(num_chunks=16)
    assert torch.all(demo["actions"][:, 0] == 3)
    assert demo["intervene_flags"].all()


def test_arx_replay_filter_is_opt_in_and_requires_validity(arx_replay_ingestion):
    ingestor = arx_replay_ingestion
    trajectory = _arx_replay_trajectory()
    trajectory.forward_inputs.pop("transition_valid")
    with pytest.raises(ValueError, match="transition_valid"):
        ingestor._ingest_rollout_trajectories([trajectory])
    ingestor.cfg.algorithm.replay_buffer.filter_invalid_transitions = False
    ingestor.demo_buffer = None
    added, _ = ingestor._ingest_rollout_trajectories([trajectory])
    assert added == 6
    sample = _sample_arx_replay(ingestor.replay_buffer)
    assert set(sample["actions"][:, 0].tolist()) == set(map(float, range(6)))


def test_arx_replay_drops_entirely_invalid_chunks(arx_replay_ingestion):
    trajectory = _arx_replay_trajectory()
    trajectory.forward_inputs["transition_valid"].fill_(False)
    assert arx_replay_ingestion._ingest_rollout_trajectories([trajectory]) == (0, 0)
    assert arx_replay_ingestion.replay_buffer.size == 0
    assert arx_replay_ingestion.demo_buffer.size == 0


def _checkpoint_replay_trajectory(identity, length):
    values = torch.arange(length, dtype=torch.float32).reshape(length, 1, 1)
    values += identity
    terminal = torch.zeros(length, 1, 1, dtype=torch.bool)
    terminal[-1] = True
    return Trajectory(
        model_weights_id="snapshot",
        actions=values.expand(length, 1, 2).clone(),
        rewards=values + 0.5,
        dones=terminal,
        terminations=terminal.clone(),
        truncations=torch.zeros_like(terminal),
        curr_obs={"states": values.clone()},
        next_obs={"states": values + 100},
    )


@pytest.mark.parametrize("auto_save", [False, True])
def test_replay_checkpoint_roundtrip_has_matching_files_index_and_samples(
    tmp_path, auto_save
):
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer

    source = TrajectoryReplayBuffer(
        auto_save=auto_save,
        auto_save_path=str(tmp_path / "live"),
        enable_cache=False,
        sample_window_size=2,
    )
    restored = TrajectoryReplayBuffer(sample_window_size=1)
    copied = TrajectoryReplayBuffer(sample_window_size=10)
    snapshot = tmp_path / "snapshot"
    second_snapshot = tmp_path / "second_snapshot"
    try:
        source.add_trajectories(
            [_checkpoint_replay_trajectory(10 * size, size) for size in (1, 2, 3)]
        )
        # No sampling precedes checkpointing: saving must not depend on the
        # lazily populated sampling window. Memory-only storage forces a cache.
        source.save_checkpoint(str(snapshot))
        expected_ids = [0, 1, 2] if auto_save else [1, 2]
        expected_values = {20.0, 21.0, 30.0, 31.0, 32.0}
        if auto_save:
            expected_values.add(10.0)
        metadata = json.loads((snapshot / "metadata.json").read_text())
        index = json.loads((snapshot / "trajectory_index.json").read_text())
        assert index["trajectory_id_list"] == expected_ids
        assert set(index["trajectory_index"]) == set(map(str, expected_ids))
        assert metadata["size"] == len(expected_ids)
        assert metadata["total_samples"] == len(expected_values)
        assert metadata["trajectory_counter"] == 3
        assert {path.name for path in snapshot.glob("trajectory_*.pt")} == {
            f"trajectory_{identity}_snapshot.pt" for identity in expected_ids
        }
        for identity in expected_ids:
            saved = torch.load(
                snapshot / f"trajectory_{identity}_snapshot.pt", map_location="cpu"
            )
            torch.testing.assert_close(
                saved["rewards"], saved["actions"][..., :1] + 0.5
            )
            assert (
                saved["rewards"].numel()
                == index["trajectory_index"][str(identity)]["num_samples"]
            )

        restored.load_checkpoint(str(snapshot))
        # The restored cache retains one trajectory, but a new checkpoint must
        # also copy the other trajectories backed by files from the first one.
        restored.save_checkpoint(str(second_snapshot))
        snapshot.rename(tmp_path / "unavailable_snapshot")
        copied.load_checkpoint(str(second_snapshot))
        assert copied.size == len(expected_ids)
        assert copied.total_samples == len(expected_values)
        batch = _sample_arx_replay(copied)
        actions = batch["actions"][:, :1]
        assert set(actions.flatten().tolist()) == expected_values
        torch.testing.assert_close(batch["rewards"], actions + 0.5)
        torch.testing.assert_close(batch["curr_obs"]["states"], actions)
        torch.testing.assert_close(batch["next_obs"]["states"], actions + 100)
        torch.testing.assert_close(
            batch["dones"], (actions == 10) | (actions == 21) | (actions == 32)
        )
    finally:
        for buffer in (source, restored, copied):
            buffer.close()


def test_replay_checkpoint_propagates_background_write_failure(tmp_path):
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer

    live = tmp_path / "live"
    # A directory where the trajectory file belongs produces a real filesystem
    # write failure in the background executor, without replacing storage code.
    (live / "trajectory_0_snapshot.pt").mkdir(parents=True)
    buffer = TrajectoryReplayBuffer(auto_save=True, auto_save_path=str(live))
    snapshot = tmp_path / "snapshot"
    try:
        buffer.add_trajectories([_checkpoint_replay_trajectory(10, 1)])
        with pytest.raises((OSError, RuntimeError), match="directory"):
            buffer.save_checkpoint(str(snapshot))
        assert not (snapshot / "metadata.json").exists()
        assert not (snapshot / "trajectory_index.json").exists()
    finally:
        with pytest.raises((OSError, RuntimeError), match="directory"):
            buffer.close()


def test_arx_human_bc_learns_correction_without_reference_target_leakage():
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin

    actor = RLTMLPPolicy(
        z_dim=4,
        proprio_dim=32,
        action_dim=10,
        num_action_chunks=2,
        ref_num_action_chunks=16,
        action_parameterization="eef_rot6d",
        gripper_width_max=0.085,
    )
    reference = (
        torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.04]).expand(2, 16, 10).clone()
    )
    obs = {
        "z_rl": torch.zeros(2, 4),
        "proprio": torch.zeros(2, 32),
        "ref_chunk": reference,
    }
    actions, result = actor.predict_action_batch(obs, mode="eval")
    policy, env = _arx_trajectory_parts()
    policy.output = PolicyOutput(**result)
    env.transition.executed_actions = actions.flatten(1)
    corrected = actions.clone()
    corrected[..., 0] += 0.2
    corrected[..., 9] = 0.015
    env.transition.intervene_actions = corrected.flatten(1)
    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=False,
        collect_transitions=True,
        enable_rlt=True,
        include_final_value=False,
    )
    torch.testing.assert_close(step.curr_obs["ref_chunk"], reference)
    pi, _, _ = actor.sac_forward(step.curr_obs, deterministic=True)
    loss_worker = SimpleNamespace(
        _flatten_chunk=RLTACLossMixin._flatten_chunk, _chunk_shape=lambda: (2, 10)
    )
    bc_loss, metrics = RLTACLossMixin._bc_metrics(
        loss_worker,
        pi,
        step.actions,
        step.curr_obs["ref_chunk"],
        step.intervene_flags,
    )
    assert bc_loss.item() > 0 and metrics["bc_human_loss"] > 0
    assert metrics["bc_ref_loss"] == 0.0
    bc_loss.backward()
    assert actor.actor_mean.weight.grad.abs().sum() > 0


@pytest.fixture(params=["ac", "td3"])
def arx_rlt_cpu_learner(request, monkeypatch):
    """Run the production losses and update loop with a CPU FSDP adapter."""
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.data.storage.replay.dataset import ReplayBufferDataset
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy
    from rlinf.workers.actor import fsdp_rlt_ac_policy_worker as ac_worker
    from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3LossMixin
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

    mixin = ac_worker.RLTACLossMixin if request.param == "ac" else RLTTD3LossMixin

    class CPUModel(torch.nn.Module):
        # FSDP supplies this clipping method; all policy computation is real.
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, **kwargs):
            return self.policy(**kwargs)

        def clip_grad_norm_(self, max_norm):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    class CPULearner(mixin, ac_worker.RLTACReplayMixin):
        forward_actor = inspect.unwrap(mixin.forward_actor)
        forward_critic = inspect.unwrap(mixin.forward_critic)
        update_one_epoch = inspect.unwrap(EmbodiedSACFSDPPolicy.update_one_epoch)
        soft_update_target_model = EmbodiedSACFSDPPolicy.soft_update_target_model

        def worker_timer(self, _name):
            return nullcontext()

    torch.manual_seed(104)
    kwargs = {
        "z_dim": 4,
        "proprio_dim": 32,
        "action_dim": 10,
        "num_action_chunks": 2,
        "ref_num_action_chunks": 16,
        "action_parameterization": "eef_rot6d",
        "gripper_width_max": 0.085,
    }
    policy = (
        RLTMLPPolicy(**kwargs, fixed_std=0.025)
        if request.param == "ac"
        else RLTTD3MLPPolicy(**kwargs, actor_noise_sigma=0.025, mlp_hidden_dim=32)
    )
    learner = CPULearner()
    learner.algorithm = request.param
    learner.model = CPUModel(policy)
    learner.target_model = copy.deepcopy(learner.model)
    learner.cfg = OmegaConf.create(
        {
            "env": {"train": {"env_type": "real"}},
            "algorithm": {
                "replay_buffer": {"filter_invalid_transitions": True},
                "q_weight": 0.1,
                "bc_weight": 5.0,
                "gamma": 0.8,
                "tau": 0.25,
                "target_update_freq": 2,
                "actor_agg_q": "q1" if request.param == "ac" else "min",
                "actor_update_action_noise": True,
            },
            "actor": {
                "model": {"num_action_chunks": 2, "action_dim": 10},
                "global_batch_size": 3,
                "micro_batch_size": 3,
                "critic_optim": {"clip_grad": 100.0},
                "optim": {"clip_grad": 100.0},
            },
        }
    )
    learner.device = torch.device("cpu")
    learner.torch_dtype = torch.float32
    learner._world_size = 1
    learner.gradient_accumulation = 1
    learner.enable_drq = False
    learner.critic_actor_ratio = 2
    learner.target_update_type = "all"
    learner.target_model_initialized = True
    learner.update_step = 0
    learner.alpha_optimizer = None
    learner.entropy_temp = SimpleNamespace(alpha=0.0)
    learner.optimizer = torch.optim.Adam(
        [p for name, p in learner.model.named_parameters() if "q_head" not in name],
        lr=0.003,
    )
    learner.qf_optimizer = torch.optim.Adam(policy.q_head.parameters(), lr=0.003)
    learner.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        learner.optimizer, factor=1.0
    )
    learner.qf_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        learner.qf_optimizer, factor=1.0
    )
    learner.replay_buffer = TrajectoryReplayBuffer(seed=3)
    learner.demo_buffer = None
    learner.buffer_dataloader_iter = iter(
        ReplayBufferDataset(learner.replay_buffer, None, 3, 1, 0)
    )
    # Only the distributed metric reduction is replaced for this single rank.
    monkeypatch.setattr(ac_worker, "all_reduce_dict", lambda values: values)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield learner
    learner.replay_buffer.close()
    torch.set_num_threads(previous_threads)


def _arx_rlt_training_rollout(learner):
    identity = torch.arange(4, dtype=torch.float32)
    reference = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.04]).repeat(4, 16, 1)
    reference[..., 0] = identity[:, None] / 10
    obs = {
        "z_rl": identity[:, None].expand(4, 4).clone(),
        "proprio": identity[:, None].expand(4, 32).clone() / 10,
        "ref_chunk": reference,
    }
    proposed, output = learner.model.policy.predict_action_batch(obs, mode="eval")
    executed = proposed.clone()
    executed[..., 0] += 0.025
    human = proposed.clone()
    human[..., 0] += 0.2
    human[..., 9] = 0.015
    flags = torch.tensor([[False, True], [True, False], [False, False], [True, True]])
    terminal = torch.tensor(
        [[False, False], [False, True], [False, False], [False, False]]
    )
    truncated = torch.tensor(
        [[False, False], [False, False], [False, True], [False, False]]
    )
    policy = PolicyPart(
        sources=[_source(TrajectoryKey(0, 0, 0, 0, 0), size=4)],
        obs={"states": obs["proprio"]},
        output=PolicyOutput(**output),
    )
    next_obs = {key: value.clone() for key, value in obs.items()}
    next_obs["z_rl"] += 10
    env = EnvPart(
        sources=policy.sources,
        transition=EnvTransition(
            rewards=torch.arange(1, 9, dtype=torch.float32).reshape(4, 2),
            dones=terminal | truncated,
            terminations=terminal,
            truncations=truncated,
            executed_actions=executed.flatten(1),
            intervene_actions=human.flatten(1),
            intervene_flags=flags,
            transition_valid=torch.tensor([[True], [True], [True], [False]]),
        ),
        next_rlt_obs=next_obs,
    )
    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=False,
        collect_transitions=True,
        enable_rlt=True,
        include_final_value=False,
    )
    expected_actions = torch.where(flags[..., None], human, executed).flatten(1)
    torch.testing.assert_close(step.actions, expected_actions)
    torch.testing.assert_close(step.curr_obs["ref_chunk"], reference)
    trajectory = Trajectory.from_steps([step])
    assert learner._ingest_rollout_trajectories([trajectory]) == (3, 2)
    batch = _sample_arx_replay(learner.replay_buffer)
    ids = batch["curr_obs"]["z_rl"][:, 0].long()
    assert set(ids.tolist()) == {0, 1, 2}
    torch.testing.assert_close(batch["actions"], expected_actions[ids])
    torch.testing.assert_close(batch["curr_obs"]["ref_chunk"], reference[ids])
    torch.testing.assert_close(batch["next_obs"]["z_rl"][:, 0], ids.float() + 10)
    torch.testing.assert_close(
        batch["intervene_flags"], flags[ids].repeat_interleave(10, 1)
    )
    assert batch["forward_inputs"]["transition_valid"].all()
    return batch


def test_arx_rlt_human_replay_optimizes_real_actor_and_critic(arx_rlt_cpu_learner):
    from rlinf.models.embodiment.base_policy import ForwardType

    learner = arx_rlt_cpu_learner
    batch = _arx_rlt_training_rollout(learner)
    # Use the same stochastic actor draw to verify the actual mixed objective.
    torch.manual_seed(19)
    pi, _, _ = learner.model(forward_type=ForwardType.SAC, obs=batch["curr_obs"])
    human_mask = batch["intervene_flags"].reshape(-1, 2, 10).any(-1)
    ref = batch["curr_obs"]["ref_chunk"][:, :2]
    executed = batch["actions"].reshape(-1, 2, 10)
    target = torch.where(human_mask[..., None], executed, ref)
    expected_bc = (pi.reshape_as(target) - target).square().mean()
    q_values = learner.model(
        forward_type=ForwardType.SAC_Q, obs=batch["curr_obs"], actions=pi
    )
    expected_q = (
        q_values[:, 0].mean()
        if learner.algorithm == "ac"
        else q_values.min(-1).values.mean()
    )
    torch.manual_seed(19)
    actor_loss, _, metrics = learner.forward_actor(batch)
    torch.testing.assert_close(actor_loss, -0.1 * expected_q + 5 * expected_bc)
    assert metrics["bc_human_loss"] > metrics["bc_ref_loss"]
    assert metrics["human_mask_ratio"] == pytest.approx(
        human_mask.float().mean().item()
    )

    actor_before = {
        n: p.detach().clone()
        for n, p in learner.model.named_parameters()
        if "q_head" not in n
    }
    critic_before = {
        n: p.detach().clone() for n, p in learner.model.policy.q_head.named_parameters()
    }
    learner.optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    gradients = [
        p.grad
        for n, p in learner.model.named_parameters()
        if n in actor_before and p.grad is not None
    ]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    learner.optimizer.step()
    assert any(
        not torch.equal(p, actor_before[n])
        for n, p in learner.model.named_parameters()
        if n in actor_before
    )
    for name, parameter in learner.model.policy.q_head.named_parameters():
        torch.testing.assert_close(parameter, critic_before[name], rtol=0, atol=0)

    learner.model.zero_grad(set_to_none=True)
    critic_loss, _ = learner.forward_critic(batch)
    assert torch.isfinite(critic_loss)
    critic_loss.backward()
    assert all(p.grad is None for p in learner.target_model.parameters())
    assert all(
        p.grad is None for n, p in learner.model.named_parameters() if "q_head" not in n
    )
    assert all(
        torch.isfinite(p.grad).all() for p in learner.model.policy.q_head.parameters()
    )
    learner.qf_optimizer.step()
    assert any(
        not torch.equal(p, critic_before[n])
        for n, p in learner.model.policy.q_head.named_parameters()
    )


@pytest.mark.parametrize("bootstrap_type", ["standard", "always"])
def test_arx_rlt_critic_bootstrap_uses_chunk_horizon_and_terminal_mask(
    arx_rlt_cpu_learner, bootstrap_type
):
    from rlinf.models.embodiment.base_policy import ForwardType

    learner = arx_rlt_cpu_learner
    learner.cfg.algorithm.bootstrap_type = bootstrap_type
    batch = _arx_rlt_training_rollout(learner)
    # Give the target distinct parameters so selecting the online network fails.
    with torch.no_grad():
        for parameter in learner.target_model.parameters():
            parameter.add_(0.01)
        torch.manual_seed(23)
        next_actor = (
            learner.model if learner.algorithm == "ac" else learner.target_model
        )
        next_actions, _, _ = next_actor(
            forward_type=ForwardType.SAC, obs=batch["next_obs"]
        )
        next_q = (
            learner.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=batch["next_obs"],
                actions=next_actions,
            )
            .min(-1)
            .values
        )
        discounted_reward = batch["rewards"][:, 0] + 0.8 * batch["rewards"][:, 1]
        bootstrap = torch.ones_like(next_q)
        if bootstrap_type == "standard":
            bootstrap = (~batch["terminations"].any(-1)).float()
        expected_target = discounted_reward + 0.8**2 * bootstrap * next_q
        data_q = learner.model(
            forward_type=ForwardType.SAC_Q,
            obs=batch["curr_obs"],
            actions=batch["actions"],
        )
        expected_loss = (data_q - expected_target[:, None]).square().mean()
    torch.manual_seed(23)
    actual_loss, _ = learner.forward_critic(batch)
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    assert all(p.grad is None for p in learner.target_model.parameters())


def test_arx_rlt_production_update_delays_actor_and_updates_target(arx_rlt_cpu_learner):
    learner = arx_rlt_cpu_learner
    _arx_rlt_training_rollout(learner)
    before = {n: p.detach().clone() for n, p in learner.model.named_parameters()}
    target_before = {
        n: p.detach().clone() for n, p in learner.target_model.named_parameters()
    }
    learner.update_step = 1
    skipped = learner.update_one_epoch()
    assert skipped["actor/updated"] == 0.0
    assert "sac/actor_loss" not in skipped
    assert "sac/critic_loss" in skipped
    assert any(
        not torch.equal(p, before[n])
        for n, p in learner.model.named_parameters()
        if "q_head" in n
    )
    for name, parameter in learner.model.named_parameters():
        if "q_head" not in name:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    for name, parameter in learner.target_model.named_parameters():
        torch.testing.assert_close(parameter, target_before[name], rtol=0, atol=0)

    learner.update_step = 2
    updated = learner.update_one_epoch()
    assert updated["actor/updated"] == 1.0
    assert "sac/actor_loss" in updated
    assert all(np.isfinite(float(value)) for value in updated.values())
    assert any(
        not torch.equal(p, before[n])
        for n, p in learner.model.named_parameters()
        if "q_head" not in n
    )
    online = dict(learner.model.named_parameters())
    for name, parameter in learner.target_model.named_parameters():
        torch.testing.assert_close(
            parameter, 0.75 * target_before[name] + 0.25 * online[name]
        )
        assert parameter.grad is None


def test_arx_rlt_train_actor_false_updates_only_critic(arx_rlt_cpu_learner):
    from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature

    learner = arx_rlt_cpu_learner
    _arx_rlt_training_rollout(learner)
    learner.cfg.algorithm.train_actor = False
    learner.update_step = 2  # This is otherwise an actor-update iteration.
    learner.entropy_temp = EntropyTemperature(initial_alpha=0.1, device="cpu")
    learner.alpha_optimizer = torch.optim.Adam(
        learner.entropy_temp.parameters(), lr=0.01
    )
    learner.alpha_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        learner.alpha_optimizer, factor=1.0
    )
    before = {
        name: value.detach().clone() for name, value in learner.model.named_parameters()
    }
    alpha_before = learner.entropy_temp.base_alpha.detach().clone()
    actor_scheduler_before = copy.deepcopy(learner.lr_scheduler.state_dict())
    alpha_scheduler_before = copy.deepcopy(learner.alpha_lr_scheduler.state_dict())
    metrics = learner.update_one_epoch(train_actor=True)
    assert metrics["actor/updated"] == 0.0
    assert "sac/actor_loss" not in metrics and "sac/alpha_loss" not in metrics
    assert np.isfinite(metrics["sac/critic_loss"])
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in learner.model.named_parameters()
        if "q_head" in name
    )
    for name, parameter in learner.model.named_parameters():
        if "q_head" not in name:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            assert parameter.grad is None
    torch.testing.assert_close(
        learner.entropy_temp.base_alpha, alpha_before, rtol=0, atol=0
    )
    assert learner.entropy_temp.base_alpha.grad is None
    assert not learner.optimizer.state and not learner.alpha_optimizer.state
    assert learner.lr_scheduler.state_dict() == actor_scheduler_before
    assert learner.alpha_lr_scheduler.state_dict() == alpha_scheduler_before


@pytest.mark.parametrize("algorithm", ["ac", "td3"])
@pytest.mark.parametrize("layout", ["step", "flat", "chunk"])
@pytest.mark.parametrize("intervention", ["none", "policy", "human", "mixed"])
def test_arx_rlt_bc_target_selection_and_gradients(algorithm, layout, intervention):
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
    from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3LossMixin

    mixin = RLTACLossMixin if algorithm == "ac" else RLTTD3LossMixin
    learner = mixin()
    learner.cfg = OmegaConf.create(
        {"actor": {"model": {"num_action_chunks": 2, "action_dim": 10}}}
    )
    pi = torch.zeros(2, 20, requires_grad=True)
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])[..., None].expand(2, 2, 10)
    human_actions = torch.tensor([[5.0, 6.0], [7.0, 8.0]])[..., None].expand(2, 2, 10)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    if intervention == "human":
        mask.fill_(True)
    elif intervention == "mixed":
        mask[0, 1] = mask[1, 0] = True
    flags = mask if layout == "step" else mask[..., None].expand(2, 2, 10)
    if layout == "flat":
        flags = flags.flatten(1)
    if intervention == "none":
        flags = None
    bc = learner._bc_metrics if algorithm == "ac" else learner._td3_bc_metrics
    loss, metrics = bc(pi, human_actions, reference, flags)
    target = torch.where(mask[..., None], human_actions, reference).flatten(1)
    expected_loss = {"none": 7.5, "policy": 7.5, "human": 43.5, "mixed": 25.5}
    assert loss.item() == pytest.approx(expected_loss[intervention])
    assert metrics["human_mask_ratio"] == pytest.approx(mask.float().mean().item())
    if intervention in ("none", "policy"):
        assert metrics["bc_human_loss"] == 0.0
    if intervention == "human":
        assert metrics["bc_ref_loss"] == 0.0
    loss.backward()
    torch.testing.assert_close(pi.grad, -2 * target / target.numel())


@pytest.mark.parametrize("algorithm", ["ac", "td3"])
@pytest.mark.parametrize("shape", [(1, 2), (2, 3), (2, 2, 9), (4, 10)])
def test_arx_rlt_bc_rejects_misaligned_intervention_masks(algorithm, shape):
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
    from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3LossMixin

    mixin = RLTACLossMixin if algorithm == "ac" else RLTTD3LossMixin
    learner = mixin()
    learner.cfg = OmegaConf.create(
        {"actor": {"model": {"num_action_chunks": 2, "action_dim": 10}}}
    )
    actions = torch.zeros(2, 20)
    bc = learner._bc_metrics if algorithm == "ac" else learner._td3_bc_metrics
    with pytest.raises(ValueError, match="[Mm]ask|[Ff]lag|intervene"):
        bc(actions, actions, actions, torch.zeros(shape, dtype=torch.bool))


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("min", [3.0, 2.0]), ("q1", [3.0, 7.0]), ("mean", [4.0, 4.5])],
)
def test_arx_rlt_td3_actor_q_aggregation(aggregation, expected):
    from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3LossMixin

    learner = RLTTD3LossMixin()
    learner.cfg = OmegaConf.create({"algorithm": {"actor_agg_q": aggregation}})
    twin_q = torch.tensor([[3.0, 5.0], [7.0, 2.0]], requires_grad=True)
    result = learner._td3_actor_q(twin_q)
    torch.testing.assert_close(result, torch.tensor(expected)[:, None])
    result.sum().backward()
    assert twin_q.grad is not None and torch.isfinite(twin_q.grad).all()
    learner.cfg.algorithm.actor_agg_q = "unsupported"
    with pytest.raises(ValueError, match="actor_agg_q"):
        learner._td3_actor_q(twin_q)


def test_prepared_action_probe_padding_and_collective_schedule(action_probe_prepared):
    from rlinf.data.datasets.openpi_rlinf.action_probe import PreparedActionProbeInputs

    directory, cfg = action_probe_prepared
    inputs = PreparedActionProbeInputs(directory, cfg, image_seed=7)
    batches = [
        inputs.batch(round_index, rank, world_size=3, batch_size=2)
        for round_index in range(inputs.batch_count(3, 2))
        for rank in range(3)
    ]
    actual = np.concatenate([batch.indices[batch.valid] for batch in batches])
    np.testing.assert_array_equal(actual, np.arange(8))
    assert all(batch.globalbatch_has_holdout for batch in batches)
    assert not np.any(batches[0].metadata["split"] == 2)
    assert not batches[-1].valid.any()
    np.testing.assert_array_equal(batches[-1].indices, [7, 7])
    assert all(batch.observation["state"].shape == (2, 32) for batch in batches)
    assert (
        inputs.batch(0, 0, world_size=1, batch_size=2).globalbatch_has_holdout is False
    )
    with pytest.raises(ValueError, match="positive"):
        inputs.batch_count(0, 2)
    with pytest.raises(ValueError, match="rank or round"):
        inputs.batch(2, 0, world_size=3, batch_size=2)


def test_prepared_action_probe_image_control_is_holdout_only(action_probe_prepared):
    from rlinf.data.datasets.openpi_rlinf.action_probe import PreparedActionProbeInputs

    directory, cfg = action_probe_prepared
    inputs = PreparedActionProbeInputs(directory, cfg, image_seed=7)
    same = PreparedActionProbeInputs(directory, cfg, image_seed=7)
    np.testing.assert_array_equal(inputs.image_donors, same.image_donors)
    np.testing.assert_array_equal(inputs.image_donors[:4], np.arange(4))
    np.testing.assert_array_equal(np.sort(inputs.image_donors[4:]), np.arange(4, 8))
    episodes = inputs.arrays["episode_id"]
    assert np.all(episodes[inputs.image_donors[4:]] != episodes[4:])
    batch = inputs.batch(0, 0, world_size=1, batch_size=8)
    for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
        np.testing.assert_array_equal(
            batch.observation[key], batch.donor_observation[key]
        )
    for key in batch.observation["image"]:
        np.testing.assert_array_equal(
            batch.observation["image_mask"][key],
            batch.donor_observation["image_mask"][key],
        )
        np.testing.assert_array_equal(
            batch.donor_observation["image"][key][:4],
            batch.observation["image"][key][:4],
        )
    assert not np.array_equal(
        batch.observation["image"]["left_wrist_0_rgb"][4:],
        batch.donor_observation["image"]["left_wrist_0_rgb"][4:],
    )


@pytest.mark.parametrize(
    "mismatch",
    ["train", "validation", "norm", "split_hash", "qc", "sample_hash", "header"],
)
def test_prepared_action_probe_rejects_mismatched_inputs(
    action_probe_prepared, mismatch
):
    from pathlib import Path

    from rlinf.data.datasets.openpi_rlinf.action_probe import PreparedActionProbeInputs

    directory, cfg = action_probe_prepared
    if mismatch == "train":
        cfg.data.train_episode_ids = [1, 2, 9]
    elif mismatch == "validation":
        cfg.data.val_episode_ids = [3, 5]
    elif mismatch == "norm":
        Path(cfg.actor.model.openpi_data.norm_stats_path).write_text("changed")
    elif mismatch == "split_hash":
        (directory / "probe_split.json").write_text("{}")
    elif mismatch == "qc":
        (directory / "input_quality_check.json").write_text('{"passed": false}')
    elif mismatch == "sample_hash":
        (directory / "samples.jsonl").write_text("changed")
    else:
        np.save(directory / "state.npy", np.zeros((8, 31), dtype=np.float32))
    with pytest.raises(ValueError, match="Action-probe|Action probe"):
        PreparedActionProbeInputs(directory, cfg)


def test_prepared_action_probe_checks_original_pipeline_config(action_probe_prepared):
    import hashlib

    from rlinf.data.datasets.openpi_rlinf.action_probe import PreparedActionProbeInputs

    directory, cfg = action_probe_prepared
    cfg.actor.model.openpi.config_name = "original_pipeline"
    source = directory / "resolved_config.yaml"
    OmegaConf.save(cfg, source)
    path = directory / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["source_files_sha256"][str(source)] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(provenance))
    cfg.actor.model.openpi.config_name = "different_pipeline"
    with pytest.raises(ValueError, match="model/data configuration mismatch"):
        PreparedActionProbeInputs(directory, cfg)


def _write_action_probe_shards(directory, inputs, fault=None):
    for rank in range(3):
        batches = [
            inputs.batch(index, rank, 3, 2) for index in range(inputs.batch_count(3, 2))
        ]
        indices = np.concatenate([batch.indices for batch in batches])
        valid = np.concatenate([batch.valid for batch in batches])
        if rank == 0 and fault == "duplicate":
            indices[1] = indices[0]
        if rank == 0 and fault == "missing":
            valid[0] = False
        values = np.repeat(indices[:, None], inputs.embedding_dim, axis=1).astype(
            np.float32
        )
        payload = {
            "indices": indices,
            "valid": valid,
            "z": values,
            "z_random": values + 1,
            "z_image_swapped": values + 2,
            **{key: inputs.arrays[key][indices].copy() for key in batches[0].metadata},
        }
        if rank == 0 and fault == "metadata":
            payload["actions"][0, 0, 0] += 1
        if rank == 0 and fault == "nonfinite":
            payload["z"][0, 0] = np.nan
        np.savez(directory / f"rank_{rank:05d}.npz", **payload)


def test_merge_action_probe_shards_preserves_order_and_metadata(
    action_probe_prepared, tmp_path
):
    import hashlib

    from rlinf.data.datasets.openpi_rlinf.action_probe import (
        PreparedActionProbeInputs,
        merge_action_probe_shards,
    )

    directory, cfg = action_probe_prepared
    inputs = PreparedActionProbeInputs(directory, cfg)
    output = tmp_path / "features"
    output.mkdir()
    _write_action_probe_shards(output, inputs)
    report = merge_action_probe_shards(output, inputs, 3)
    assert report["sample_count"] == 8
    assert (
        report["features_sha256"]
        == hashlib.sha256((output / "features.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "features.npz", allow_pickle=False) as merged:
        np.testing.assert_array_equal(merged["z"][:, 0], np.arange(8))
        np.testing.assert_array_equal(merged["image_donor_index"], inputs.image_donors)
        for key in (
            "state",
            "actions",
            "episode_id",
            "frame_index",
            "dataset_global_index",
            "split",
        ):
            np.testing.assert_array_equal(merged[key], inputs.arrays[key])
    with pytest.raises(FileExistsError):
        merge_action_probe_shards(output, inputs, 3)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "metadata", "nonfinite"])
def test_merge_action_probe_shards_rejects_invalid_results(
    action_probe_prepared, tmp_path, fault
):
    from rlinf.data.datasets.openpi_rlinf.action_probe import (
        PreparedActionProbeInputs,
        merge_action_probe_shards,
    )

    directory, cfg = action_probe_prepared
    inputs = PreparedActionProbeInputs(directory, cfg)
    output = tmp_path / "features"
    output.mkdir()
    _write_action_probe_shards(output, inputs, fault)
    with pytest.raises(ValueError, match="Action-probe|action-probe"):
        merge_action_probe_shards(output, inputs, 3)
    assert not (output / "features.npz").exists()


def test_prepared_action_probe_detects_input_changes_during_extraction(
    action_probe_prepared,
):
    from rlinf.data.datasets.openpi_rlinf.action_probe import PreparedActionProbeInputs

    directory, cfg = action_probe_prepared
    inputs = PreparedActionProbeInputs(directory, cfg)
    changed = np.load(directory / "state.npy", mmap_mode="r+")
    changed[0, 0] = 99
    changed.flush()
    with pytest.raises(ValueError, match="changed during extraction"):
        inputs.validate_unchanged()
