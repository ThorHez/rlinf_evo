RL Token Stage 2 on ARX
=======================

Train an RL Token Stage 2 policy on one ARX arm with a GoPro video feed and
same-arm drag interventions. This recipe connects the existing ARX SDK service
to RLinf, keeps the checkpoint's EEF representation, and uses the same execution
path for training and fixed-weight evaluation. Configure the model and measured
timing first, then verify the hardware before enabling motion.

Overview
--------

The configuration preserves the original 16-step prediction and execution
horizons. The first predicted action corresponds to the observation pose, so
executing only that first action would discard the future movement. Inference and execution are synchronous;
background camera capture, feedback sampling, and operator events continue
independently. Waiting for inference can still pause policy motion.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models

      Frozen π₀.₅ + RLT features; Stage 2 MLP actor and critic.

   .. grid-item-card:: Algorithms

      RLT AC or the existing RLT TD3 variant, with reference and intervention BC.

   .. grid-item-card:: Tasks

      Operator-scored manipulation and same-arm drag corrections.

   .. grid-item-card:: Hardware

      One ARX arm, wrist UVC capture, and a local NVIDIA GPU.

Tasks
~~~~~

Choose a task matching the Stage 1 checkpoint and configure its instruction.

.. list-table::
   :header-rows: 1

   * - Environment
     - Task
     - Configuration
   * - ``Arx5RLTEnv-v1``
     - Manual reset; operator success/failure labels
     - ``realworld_arx5_rlt_stage2_ac_mlp``
   * - ``Arx5RLTEnv-v1``
     - Same task and controls with the TD3 Stage 2 head
     - ``realworld_arx5_rlt_stage2_td3_mlp``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Keep training, inference, and human data in the same model coordinate system.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Contract
   * - Observation
     - One 224 × 224 RGB wrist image and two 16D state frames, flattened to 32D.
       The state contains relative XYZ, row rot6d, rotation relative to episode
       start, and gripper width.
   * - Action
     - Ten values: normalized XYZ, first two rotation-matrix rows, gripper metres.
       Each chunk uses the EEF pose associated with its input observation as a
       fixed anchor. Stage 1 and Stage 2 use 16-step chunks.
   * - Reward
     - Success +1, failure −1, operator abort 0; episode timeout is truncation.
   * - Prompt
     - ``ARX_TASK_DESCRIPTION``, matching the checkpoint's task.

``pi05_umi_v4`` is the existing configuration identifier for this checkpoint
contract. Supply its original ``norm_stats.json``: only XYZ is normalized,
using min/max when present and q01/q99 otherwise, without clipping. Rotation and
gripper values retain their original units. Historical gripper binarization,
world-Z offsets, and EEF-to-tool calibration must be verified explicitly rather
than inferred from the model name.

The actor adds a learned residual to the Stage 1 reference, with a zero-initialized
output head. XYZ stays in the original normalized domain without a global
``tanh``; row rot6d is projected onto a valid rotation and gripper width uses
the configured physical bounds. Reference-input dropout defaults to zero for
this residual policy.

Both heads use ``action_parameterization: eef_rot6d`` and the same action
conversion, human-intervention encoding, replay and synchronous execution.
The TD3 configuration retains this branch's variant: twin-Q minimum for the
actor, action noise during actor updates, two critic updates per actor update,
and fixed Q/BC weights of 1.0. It does not enable the simulator's collection or
weight schedules. These settings describe the implemented RLT TD3 variant,
rather than a claim that every canonical TD3 setting is used.

For TD3, ``actor_noise_sigma: 0.1`` applies in the learned residual coordinates.
The sampled residual is multiplied by ``eef_residual_scale`` before being
added to the reference: the default scales are 0.1 for XYZ and the six rotation
entries, and 0.005 for gripper width. Thus the added noise has standard deviation
0.01 in normalized XYZ/rotation entries and 0.0005 m in width before rotation
projection and width bounds. It is not 0.1 m of Cartesian or gripper motion;
the XYZ displacement in metres depends on the original normalization statistics.
Evaluation disables this sampling noise. AC keeps its existing
``fixed_std: 0.002``; that setting does not control the TD3 head.

Installation
------------

Use Python 3.11 for RLinf and the existing Python 3.10 environment for the
CPython 3.10 ARX SDK binary. From this branch's repository root, install the
model and client dependencies:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env arx5
   source .venv/bin/activate
   export REPO_PATH="$PWD"
   export EMBODIED_PATH="$REPO_PATH/examples/embodiment"

The installer adds OpenPI, UVC capture, and the ZMQ client to the RLinf
interpreter. The SDK-owning service below runs separately. If you already have
the model environment, install only the additional client requirements with
``python -m pip install -r requirements/embodied/envs/arx5.txt``.

Configure the Model and Timing
------------------------------

Point both sides of the action conversion at the same original statistics,
and use a complete Stage 1 checkpoint containing ``rlt_module.*`` weights.
A base π₀.₅ checkpoint alone cannot supply the trained RLT features. The default
RLT dimensions match ``umi_rlt_stage1_sft_openpi_pi05.yaml``; match any changes
you made during Stage 1 in ``rollout.rlt_feature_model.openpi``.

Set the following environment variables before composing the configuration:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Value to provide
   * - ``RLT_STAGE1_CHECKPOINT``
     - Complete Stage 1 checkpoint path, including the trained RLT module.
   * - ``RLT_NORM_STATS``
     - Original checkpoint's ``norm_stats.json`` file.
   * - ``ARX_ACTION_DT`` / ``ARX_HISTORY_DT``
     - Action interval and state-history interval, in seconds, derived from the actual dataset.
   * - ``ARX_CAMERA_LATENCY``
     - Measured image delay in seconds, subtracted from host frame receipt time.
   * - ``ARX_CAMERA_DEVICE``
     - Explicit ``/dev/v4l/by-id/...`` path for the wrist capture device.
   * - ``ARX_GRIPPER_MIN`` / ``ARX_GRIPPER_MAX``
     - Calibrated gripper-width bounds in metres, matching the service's conversion.
   * - ``ARX_TASK_DESCRIPTION``
     - Task instruction used by the model.

The timing variables have no defaults. Dataset output FPS alone does not prove
the physical action or history interval. Measure image delay with
``toolkits/realworld_check/measure_camera_latency.py`` and verify jitter, camera
crop, orientation, distortion, and the GoPro lens mode against the training
images. Set ``env.train.override_cfg.camera_resolution``, ``camera_fps``,
``camera_pixel_format`` (default ``MJPG``) and
``image_crop`` to the verified capture profile; apply matching settings under
``env.eval.override_cfg``.

The camera returns host monotonic receipt time, not exposure time. The
environment subtracts the configured delay and interpolates robot feedback at
that instant. Excessive frame age or feedback gaps stop the episode. The UVC
capture subprocess isolates native driver hangs; it does not perform model
inference or enable RTC.

Start the Robot Service
-----------------------

Prepare the arm, load, CAN interface, and calibrated tool frame before starting
the SDK-owning process. Only one process may control that CAN interface; stop
any earlier SDK server first. Save your verified service settings as a JSON
file with the following fields:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Meaning
   * - ``robot`` / ``controller``
     - Explicit overrides of SDK configuration fields, including validated gravity settings.
   * - ``controller.default_preview_time``
     - Validated SDK target-transition duration in seconds, greater than ``controller_dt``; distinct from the model action interval.
   * - ``adapter.eef_to_tcp``
     - Calibrated 4 × 4 transform from SDK EEF to the model tool frame.
   * - ``adapter.gripper_scale`` / ``gripper_offset``
     - Conversion into the gripper metres used by the checkpoint.
   * - ``adapter.drag_kd`` / ``drag_gripper``
     - Six validated drag damping values; ``hold`` or ``manual`` gripper behavior.
   * - ``limits``
     - Positive per-command ``translation`` (m), ``rotation`` (rad), ``gripper`` (m), and ``joint`` (rad) limits.
   * - ``service``
     - Feedback rate/history and state/command timeout settings.
   * - ``calibration_confirmed``
     - Set to ``true`` only after validating these settings on the actual rig.

Use the SDK interpreter to start the service initially with motion disabled:

.. code-block:: bash

   /media/lenovo/data/miniconda3/envs/arx-py310/bin/python \
     toolkits/realworld_check/arx5_server.py \
     --sdk-path /home/lenovo/kian/arx5-sdk --model X5 --interface can0 \
     --config /path/to/verified-arx5.json

The service binds to ``tcp://127.0.0.1:5557`` by default. SDK construction starts
its CAN/control loop and may apply gravity or damping; this is an active
controller connection, even with policy motion disabled. Check feedback from a
second terminal using ``python toolkits/realworld_check/arx5_server.py --operation state``.
Support the arm and payload before stopping the owning SDK process, because
SDK shutdown can change damping. Restart the service with ``--allow-motion``
only for supervised hardware validation and operation; that option requires
explicit limits, confirmed calibration and a valid SDK preview time.

Measure the largest normal gap between model commands, including synchronous
inference and learner updates, before choosing ``service.command_timeout``.
The default is two seconds; expiry holds the measured pose and changes the
control epoch. State reads do not renew this timer. Explicitly resume control
after diagnosing a timeout; an old action chunk cannot silently restart it.

Run It
------

Choose one training configuration and its matching evaluation configuration.
The existing AC commands remain available; select the second pair to use TD3:

.. code-block:: bash

   TRAIN_CONFIG=realworld_arx5_rlt_stage2_ac_mlp
   EVAL_CONFIG=realworld_arx5_rlt_stage2_eval

.. code-block:: bash

   TRAIN_CONFIG=realworld_arx5_rlt_stage2_td3_mlp
   EVAL_CONFIG=realworld_arx5_rlt_stage2_td3_eval

Use the selected pair throughout the run. First inspect the fully resolved
training configuration without connecting to the robot or loading GPU models:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-name "$TRAIN_CONFIG" --cfg job --resolve

After validating feedback, images, a small model command, drag entry, and return
to the measured pose, start synchronous training:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh "$TRAIN_CONFIG"

The episode waits for the operator to establish the starting pose and press
``s``. Training starts with the Stage 1 reference policy; press ``b`` when the
Stage 2 actor should take over. These keys are handled independently of model
inference, through the environment's keyboard listener:

.. list-table::
   :header-rows: 1

   * - Key
     - Operation
   * - ``s``
     - Hold the measured pose, enable policy control, and start the episode.
   * - ``b``
     - Select the Stage 2 actor for subsequent predictions.
   * - ``v``
     - Select the Stage 1 reference policy for subsequent predictions.
   * - ``g``
     - Enter gravity-compensated drag; invalidate queued model commands.
   * - ``r``
     - Hold the current measured pose and resume policy control with new observations.
   * - ``c`` / ``a`` / ``q``
     - Mark success / failure / abort and hold position. While waiting to start,
       ``q`` cancels the wait; ``c`` and ``a`` are ignored.

Reset is manual: establish the next starting pose, then press ``s`` again.
While waiting to start, ``r`` holds the measured pose without enabling policy
motion. The environment does not call a joint-home routine. Keyboard control
reads the local Linux keyboard through evdev. Grant the runtime user read
access to the intended device under ``/dev/input`` and select its stable path
before launching RLinf:

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE='/dev/input/by-id/<keyboard>-event-kbd'

Choose the keyboard attached to the robot host; the listener uses its device
events, so typing into an SSH terminal does not send these controls. From a
separate terminal, ``--operation
enter_drag`` and ``--operation hold`` reach the service even while inference is
waiting; those service operations do not provide episode labels.

Measured drag trajectories are encoded in the same normalized action domain as
the actor. Valid samples use the existing replay, critic and intervention BC
path. Interrupted or stale transitions are excluded by
``algorithm.replay_buffer.filter_invalid_transitions: true``. A success/failure
label holds the arm immediately; the environment then samples the remaining
chunk intervals at that held pose and records the terminal reward once at the
chunk boundary. These measured hold samples preserve the outcome in replay;
abort, fault and stale-control samples remain invalid. Operator events and measured
step records are saved under ``runner.logger.log_path/arx5_raw`` as JSONL;
these records are not a complete image dataset.

Save, Resume, and Evaluate
--------------------------

Keep a complete RLinf checkpoint to resume optimizer, target-network and the
retained replay window. ``auto_save: false`` disables per-trajectory automatic
saving; the actor checkpoint still writes the cached replay trajectories under
``actor/sac_components/replay_buffer``. The default saves every ten training
epochs. Resume by supplying the
``global_step_<N>`` directory:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh "$TRAIN_CONFIG" \
     runner.resume_dir=/path/to/checkpoints/global_step_10

Keep the algorithm and model dimensions from that checkpoint when resuming.
AC and TD3 checkpoints use different heads and are not interchangeable; both
reuse the same frozen Stage 1 checkpoint and normalization statistics.

For fixed-weight evaluation, set ``RLT_STAGE2_WEIGHTS`` to the selected head's
``actor/model_state_dict/full_weights.pt`` file, keep the same Stage 1 assets
and timing variables, and run:

.. code-block:: bash

   export RLT_STAGE2_WEIGHTS=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt
   bash evaluations/run_eval.sh "$EVAL_CONFIG"

This configuration supplies a complete rollout policy specification and starts
with Stage 2 selected after ``s``. Both algorithms use the same observations,
synchronous action execution and drag controls as training, with RTC disabled. See
:doc:`Checkpoint Resume </rst_source/guides/resume>` for checkpoint layout and
:doc:`RL Token <rlt>` for the existing learning objective.

Validation Status
-----------------

CPU tests cover the codec, service protocol, controller mocks, UVC process
lifecycle, environment integration and Hydra configuration. They do not establish
physical tracking, gravity compensation under your payload, GoPro latency or
full-checkpoint GPU memory use. Verify those on the actual arm before collecting
training data, then run a short training/save/resume/evaluation cycle. Frozen
Stage 1 and Stage 2 share the GPU in this configuration; reserve enough memory
for both rather than using the small-model test footprint as a capacity result.
