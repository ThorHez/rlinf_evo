# Copyright 2025 The RLinf Authors.
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
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.checkpoint import (
    ValidationCheckpointTracker,
    parse_global_step_from_checkpoint_path,
)
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import EarlyStopController, check_progress

if TYPE_CHECKING:
    from rlinf.workers.reward.reward_worker import FSDPRewardWorker
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

logger = logging.getLogger(__name__)


class SFTRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union["FSDPSftWorker", "FSDPRewardWorker"],
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        early_stop_cfg = cfg.runner.get("early_stop", None)
        self.early_stop = (
            EarlyStopController(early_stop_cfg) if early_stop_cfg is not None else None
        )

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)
        selection = cfg.runner.get("checkpoint_selection")
        self.checkpoint_tracker = None
        if selection is not None:
            if self.early_stop is not None:
                raise ValueError(
                    "Use checkpoint_selection without the legacy accuracy early_stop controller."
                )
            if cfg.runner.val_check_interval <= 0:
                raise ValueError("Checkpoint selection requires periodic validation.")
            self.checkpoint_tracker = ValidationCheckpointTracker(
                os.path.join(
                    cfg.runner.logger.log_path,
                    cfg.runner.logger.experiment_name,
                    "checkpoints",
                ),
                metric=selection.metric,
                mode=selection.get("mode", "min"),
                keep_last=int(selection.get("keep_last", 1)),
            )
            if self.checkpoint_tracker.saved_steps:
                resume_dir = cfg.runner.get("resume_dir")
                if resume_dir is None:
                    raise ValueError(
                        "Start a fresh training run in a new output directory."
                    )
                if os.path.realpath(os.path.dirname(resume_dir)) != os.path.realpath(
                    self.checkpoint_tracker.root
                ):
                    raise ValueError(
                        "Existing checkpoint selection belongs to a different run."
                    )
                resume_step = parse_global_step_from_checkpoint_path(resume_dir)
                if resume_step < max(self.checkpoint_tracker.saved_steps):
                    raise ValueError(
                        "Resume the latest saved step or use a new output directory for rollback."
                    )

        probe = cfg.runner.get("action_probe")
        self.action_probe = (
            probe if probe is not None and probe.get("enabled", True) else None
        )
        if self.action_probe is not None:
            interval = self.action_probe.get("interval", 5000)
            if (
                isinstance(interval, bool)
                or not isinstance(interval, int)
                or interval < 1
                or cfg.runner.val_check_interval < 1
                or interval % cfg.runner.val_check_interval
            ):
                raise ValueError(
                    "Action probe interval must be a positive multiple of val_check_interval."
                )
            if (
                cfg.actor.model.model_type != "openpi_rlinf"
                or not cfg.actor.model.openpi.get("use_rlt", False)
            ):
                raise ValueError(
                    "Action probes require an openpi_rlinf model with RLT enabled."
                )
            if cfg.cluster.num_nodes != 1:
                raise ValueError(
                    "Action probes currently require a shared single-node output directory."
                )
            if not self.action_probe.get("input_dir"):
                raise ValueError(
                    "Action probes require prepared input_dir with episode splits."
                )
            methods = list(self.action_probe.get("methods", ["ridge", "mlp"]))
            if (
                not methods
                or len(set(methods)) != len(methods)
                or set(methods) - {"ridge", "mlp"}
            ):
                raise ValueError(
                    "Action probe methods must be distinct ridge and/or mlp entries."
                )
            if (
                self.checkpoint_tracker is not None
                and self.checkpoint_tracker.metric.startswith("rlt_probe/")
                and interval != cfg.runner.val_check_interval
            ):
                raise ValueError(
                    "Selecting checkpoints by a probe metric requires probes at every validation."
                )
            if (
                self.checkpoint_tracker is not None
                and self.checkpoint_tracker.metric.startswith("rlt_probe/")
                and cfg.runner.get("eval_at_start", False)
                and not self.action_probe.get("at_start", False)
            ):
                raise ValueError(
                    "Initial checkpoint selection by a probe metric requires action_probe.at_start."
                )

    def init_workers(self) -> None:
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()

    def run(self) -> None:
        start_step = self.global_step
        if start_step == 0 and self.cfg.runner.get("eval_at_start", False):
            initial_metrics = self.actor.run_eval().wait()[0]
            if self._action_probe_due():
                initial_metrics.update(self._run_action_probe())
            self.metric_logger.log(
                {f"eval/{key}": value for key, value in initial_metrics.items()}, 0
            )
            self._record_validation(initial_metrics)
            if self.checkpoint_tracker is not None:
                self.checkpoint_tracker.improves(initial_metrics)
                self._save_checkpoint()
                self.checkpoint_tracker.record_saved(0, initial_metrics)
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            if hasattr(self.actor, "set_global_step"):
                # set global step
                self.actor.set_global_step(self.global_step)

            with self.timer("step"):
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                eval_model, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model and self.checkpoint_tracker is None:
                    self._save_checkpoint()

                should_stop = False
                if eval_model:
                    eval_handle: Handle = self.actor.run_eval()
                    eval_metrics = eval_handle.wait()
                    if self._action_probe_due():
                        eval_metrics[0].update(self._run_action_probe())
                    self._record_validation(eval_metrics[0])

                    if self.early_stop is not None:
                        should_stop, best_val_acc_improved = self.early_stop.update(
                            eval_metrics[0]
                        )
                        if best_val_acc_improved:
                            self._save_checkpoint(is_best=True)

                if self.checkpoint_tracker is not None:
                    validation = eval_metrics[0] if eval_model else None
                    improved = (
                        validation is not None
                        and self.checkpoint_tracker.improves(validation)
                    )
                    if save_model or improved:
                        self._save_checkpoint()
                        self.checkpoint_tracker.record_saved(
                            self.global_step, validation
                        )

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            if eval_model:
                time_metrics["evaluate"] = eval_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            merged_actor_metrics = {}
            # get the merged actor metrics from all ranks
            for metrics in actor_metrics:
                for k, v in metrics.items():
                    if k not in merged_actor_metrics:
                        merged_actor_metrics[k] = v
            training_metrics = {
                f"train/{k}": v for k, v in merged_actor_metrics.items()
            }
            log_step = (
                self.global_step
                if self.checkpoint_tracker is not None or self.action_probe is not None
                else _step
            )
            self.metric_logger.log(time_metrics, log_step)
            self.metric_logger.log(training_metrics, log_step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)

            if eval_model:
                evaluate_metrics = {f"eval/{k}": v for k, v in eval_metrics[0].items()}
                logging_metrics.update(evaluate_metrics)
                self.metric_logger.log(evaluate_metrics, log_step)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)
            if should_stop:
                break

        if self.early_stop is not None and self.early_stop.best_val_acc > 0:
            logger.info(
                f"Early stopping triggered! Best val_acc: {self.early_stop.best_val_acc:.4f}"
            )
        self.metric_logger.finish()

    def _record_validation(self, metrics: dict[str, float]) -> None:
        """Persist optimizer-step metrics for selection or action diagnostics."""
        if self.checkpoint_tracker is None and self.action_probe is None:
            return
        path = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "validation_metrics.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(
                json.dumps({"global_step": self.global_step, "metrics": metrics}) + "\n"
            )
        logger.info("Validation at optimizer step %s: %s", self.global_step, metrics)

    def _action_probe_due(self) -> bool:
        """Run configured diagnostics at scheduled validation and the final step."""
        if self.action_probe is None:
            return False
        if self.global_step == 0:
            return bool(self.action_probe.get("at_start", False))
        return (
            self.global_step % int(self.action_probe.get("interval", 5000)) == 0
            or self.global_step == self.max_steps
        )

    def _run_action_probe(self) -> dict[str, float]:
        """Extract frozen features on all actors and fit CPU-only action heads."""
        from rlinf.utils.action_probe import run_action_probes

        output_dir = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "action_probe"
            / f"global_step_{self.global_step}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with self.timer("action_probe"):
            self.actor.extract_action_probe_features(
                str(output_dir), self.global_step
            ).wait()
            result = run_action_probes(
                output_dir / "features.npz",
                Path(self.action_probe.input_dir) / "probe_split.json",
                output_dir / "probe_results",
                methods=tuple(self.action_probe.get("methods", ["ridge", "mlp"])),
                threads=int(self.action_probe.get("cpu_threads", 4)),
                max_epochs=int(self.action_probe.get("max_epochs", 100)),
                patience=int(self.action_probe.get("patience", 15)),
            )
        logger.info(
            "Action probes at optimizer step %s: %s",
            self.global_step,
            result["reports"],
        )
        return {f"rlt_probe/{key}": value for key, value in result["metrics"].items()}

    def run_eval(self) -> dict[str, float]:
        """Evaluate the loaded model, including configured action probes."""
        with self.timer("evaluate"):
            eval_handle: Handle = self.actor.run_eval()
            eval_metrics = eval_handle.wait()
            if self.action_probe is not None:
                eval_metrics[0].update(self._run_action_probe())

        time_metrics = self.timer.consume_durations()
        time_metrics["evaluate"] = eval_handle.consume_duration()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

        raw_eval = (
            eval_metrics[0]
            if isinstance(eval_metrics, (list, tuple)) and len(eval_metrics) > 0
            else {}
        )
        evaluate_metrics = {f"eval/{k}": v for k, v in raw_eval.items()}

        logging_metrics = {}
        logging_metrics.update(time_metrics)
        logging_metrics.update(evaluate_metrics)

        logger.info(f"Eval metrics: {evaluate_metrics}")
        self._record_validation(raw_eval)
        self.metric_logger.log(time_metrics, self.global_step)
        self.metric_logger.log(evaluate_metrics, self.global_step)
        self.metric_logger.finish()
        return raw_eval

    def _save_checkpoint(self, is_best: bool = False) -> None:
        checkpoint_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        if is_best:
            base_output_dir = os.path.join(checkpoint_root, "checkpoints/best_model")
        else:
            base_output_dir = os.path.join(
                checkpoint_root,
                f"checkpoints/global_step_{self.global_step}",
            )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        if is_best and self.early_stop is not None:
            logger.info(
                f"Saved best model (val_acc={self.early_stop.best_val_acc:.4f}) to {base_output_dir}"
            )

    def set_max_steps(self) -> None:
        self.num_steps_per_epoch = self.actor.get_max_steps_per_epoch().wait()[0]
        max_epochs = self.cfg.runner.get("max_epochs", -1)
        max_steps = self.cfg.runner.get("max_steps", -1)

        step_limits = []
        if max_epochs > 0:
            step_limits.append(self.num_steps_per_epoch * max_epochs)
        if max_steps >= 0:
            step_limits.append(max_steps)

        # If both limits are configured, stop at whichever one is reached first.
        # If neither is configured, keep the historical default of one epoch.
        self.max_steps = min(step_limits) if step_limits else self.num_steps_per_epoch

    @property
    def epoch(self) -> int:
        return self.global_step // self.num_steps_per_epoch
