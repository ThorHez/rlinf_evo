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
     - Operator success/failure labels; smooth return to a configured start pose
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

Use Python 3.11 for RLinf and a separate Python 3.10 environment for the
ARX SDK service. From this branch's repository root, install the
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

The repository includes the ARX SDK under ``third_party/arx5-sdk``, based on
upstream commit ``c160db7`` with local emergency-handling fixes. Its source,
license, and base revision are recorded in
:download:`third_party/README.md <../../../../../third_party/README.md>`.
Build the Python extension with the SDK's Python 3.10 environment. In a separate
terminal, run these commands from the repository root:

.. code-block:: bash

   conda activate arx-py310
   (
   set -e
   cd third_party/arx5-sdk
   mkdir build && cd build
   cmake ..
   make -j
   )

These are the upstream README's build commands for a fresh build directory.
For an existing build, enter ``third_party/arx5-sdk/build`` and run
``cmake ..`` followed by ``make -j``. This builds all default targets, including
the extension under ``third_party/arx5-sdk/python``, using the C++ dependencies
in ``arx-py310``. Keep that directory together with the SDK's
``models`` and ``lib`` directories. For a new SDK environment, follow
``third_party/arx5-sdk/README.md`` and its
``conda_environments/py310_environment.yaml``. Use the SDK terminal for the
service commands below and the RLinf terminal for training.

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
   * - ``ARX_RESET_POSE``
     - Fixed return target as ``[x,y,z,rx,ry,rz]`` in the service TCP frame: XYZ in metres and a rotation vector in radians.
   * - ``ARX_RESET_GRIPPER``
     - Gripper width on return, in metres; defaults to 0.085 and must lie within the configured gripper bounds.

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

Choose where the arm should return after each completed episode. Replace the
placeholders below with a target verified for your setup before launching:

.. code-block:: bash

   export ARX_RESET_POSE='[<x>,<y>,<z>,<rx>,<ry>,<rz>]'
   export ARX_RESET_GRIPPER='<initial gripper width in metres>'

The target uses the same TCP frame as the service's measured feedback and
absolute commands. With the default identity ``adapter.eef_to_tcp``, this is
the SDK EEF frame. Convert poses from a previous deployment through its origin,
tool offset, and orientation transforms first; a UMI TCP rotation vector cannot
be copied directly into the SDK EEF frame. The target is independent of the
per-episode observation anchor.

The environment options below apply under both ``env.train.override_cfg`` and
``env.eval.override_cfg``. The default enables automatic return and requires a
configured target:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Default and behavior
   * - ``return_on_episode_end``
     - ``true``: return after success, failure, or the episode step limit. Set ``false`` for manual repositioning.
   * - ``reset_pose``
     - Read from ``ARX_RESET_POSE``; six finite values, with no assumed hardware target.
   * - ``reset_gripper_width``
     - Read from ``ARX_RESET_GRIPPER``; defaults to 0.085 m.
   * - ``reset_duration`` / ``reset_frequency``
     - Generate waypoints for 4.0 seconds at 20.0 Hz, using eased translation and shortest-path rotation interpolation. Sequential communication waits extend the movement beyond this minimum duration.
   * - ``reset_settle_timeout``
     - Up to 2.0 additional seconds to reach the final target before holding.

The return finishes when measured position error is at most 5 mm, orientation
error at most 0.05 rad, and gripper-width error at most 5 mm. Failure to reach
these tolerances before the settling timeout stops execution with an error.
The same service command limits and feedback checks apply throughout the return.

Start the Robot Service
-----------------------

Prepare the arm, load, CAN interface, and calibrated tool frame before starting
the SDK-owning process. Only one process may control that CAN interface; stop
any earlier SDK server first. RLinf uses a JSON service protocol on
``tcp://127.0.0.1:5557``. The SDK's original ``python/communication/zmq_server.py``
uses a different protocol on port ``8765`` and cannot serve RLinf clients.

The service loads the patched ``third_party/arx5-sdk`` by default;
``--sdk-path`` selects another SDK checkout, which does not receive these
local fixes. Supply the robot model and CAN interface as ``X5 can0``
or as ``--model X5 --interface can0``. Inspect the effective service settings
before connecting:

.. code-block:: bash

   "$CONDA_PREFIX/bin/python" toolkits/realworld_check/arx5_server.py \
     X5 can0 --print-config

``--print-config`` prints the merged software defaults and JSON overrides, then
exits without importing the SDK or opening hardware. The service uses SDK
robot/controller defaults with the settings in
``toolkits/realworld_check/arx5_server_defaults.json``, summarized below.
To carry over settings from an existing
deployment, copy its explicit Python configuration overrides into a JSON file
and pass ``--config /path/to/arx5.json``. Each supplied field overrides the
matching default within its section; other fields keep their defaults.

The service loads ``cartesian_controller`` from the SDK configuration factory
and constructs ``Arx5CartesianController``. It passes end-effector
targets to ``set_eef_cmd``; the SDK handles IK retries, joint targets, and
interpolation. The local emergency fixes preserve normal Cartesian IK behavior
and SDK defaults. The empty ``controller``
section in the service defaults preserves the SDK's Cartesian controller gains
and 0.1-second interpolation preview.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Meaning
   * - ``robot`` / ``controller``
     - Overrides of SDK configuration fields, including gripper calibration, gains, and gravity settings from your deployment.
   * - ``controller.default_preview_time``
     - SDK target-transition duration; uses the ``cartesian_controller`` default of 0.1 s and must exceed ``controller_dt``. This is separate from the model action interval.
   * - ``adapter.eef_to_tcp``
     - 4 × 4 transform from SDK EEF to the model tool frame; defaults to the identity transform.
   * - ``adapter.gripper_scale`` / ``gripper_offset``
     - Conversion into the checkpoint's gripper metres; defaults to 1.0 and 0.0, preserving the SDK's metre values.
   * - ``adapter.drag_kd`` / ``drag_gripper``
     - Six drag damping values, defaulting to the controller's ``default_kd``; ``hold`` (default) or ``manual`` gripper behavior.
   * - ``limits``
     - Positive per-command limits: ``translation`` 0.05 m, ``rotation`` 0.2 rad, and ``gripper`` 0.02 m by default.
   * - ``service``
     - Feedback rate/history and state/command timeout settings; ``command_timeout`` defaults to 10 s in this entrypoint.
   * - ``calibration_confirmed``
     - Optional deployment status. An explicit ``false`` blocks ``--allow-motion``; omitting the field permits startup with the motion flag.

The identity transform uses the SDK EEF frame directly. Check that this frame
and the SDK gripper widths match the training data; these defaults do not
establish compatibility with a UMI checkpoint. Review the motion limits for
your rig and task. If the old deployment changes gripper readout, width, or
gains in its Python script, carry those overrides into the JSON as well.

Use the SDK interpreter to start the service initially with motion disabled:

.. code-block:: bash

   "$CONDA_PREFIX/bin/python" toolkits/realworld_check/arx5_server.py X5 can0

The service binds to ``tcp://127.0.0.1:5557`` by default. SDK construction starts
its CAN/control loop and may apply gravity or damping; this is an active
controller connection, even with policy motion disabled. Check feedback from a
second terminal using ``python toolkits/realworld_check/arx5_server.py --operation state``.
Support the arm and payload before stopping the owning SDK process, because
SDK shutdown can change damping. Restart the service with ``--allow-motion``
for supervised hardware validation and operation, adding ``--config`` when you
have deployment overrides. RLinf checks end-effector step limits, gripper
travel, command deadlines, the control epoch, and live feedback before calling
the SDK. The motion flag does not bypass those checks or the command watchdog.
Normal IK and joint handling follow the upstream Cartesian controller implementation.

Measure the largest normal gap between model commands, including synchronous
inference and learner updates, before choosing ``service.command_timeout``.
The entrypoint default is ten seconds; expiry holds the measured pose and
changes the control epoch. State reads do not renew this timer. Explicitly resume control
after diagnosing a timeout; an old action chunk cannot silently restart it.

The local SDK fixes release state and command locks before emergency handling,
raise ``RuntimeError`` on startup validation failures, and let background
emergency damping stop during destruction. After a fault, both Joint and
Cartesian controllers reject further motion commands. Resolve the underlying fault
and recreate the controller or restart the service.

The SDK runs directly in the service process. During service operation,
``Ctrl+C`` or ``SIGTERM`` requests ``service.stop()``; the SDK handles normal
cleanup. Support the arm and payload before stopping. Use the SDK built under
``third_party/arx5-sdk`` to receive the emergency-handling fixes, and resolve
gripper calibration errors that caused an emergency before restarting.

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

After the environment starts, open the
`local camera preview <http://127.0.0.1:8765/>`_ in a browser on the robot host.
The log prints the URL; no window opens automatically. AC and TD3 training
enable the preview by default, while evaluation leaves it disabled. The page
shows the camera image scaled with its aspect ratio preserved, the EP number,
and the current state. It keeps updating while waiting for ``s``, during
inference, and during return movements, and indicates stale frames or a lost
connection.

EP numbering starts at 1 and advances only when a new episode successfully
starts; while waiting, it shows the next episode number. Repeated ``s`` presses,
``g``, ``r``, ``b``, ``v``, and a completed 64-step rollout do not increment it.
Restarting the process or resuming a checkpoint resets this display to 1. It
does not track the runner's ``global_step`` or the replay trajectory count.

Closing the page leaves collection running. To disable the preview, set
``env.train.override_cfg.enable_camera_preview=false``; configure the port with
``env.train.override_cfg.camera_preview_port``, which defaults to ``8765``.
Evaluation uses the corresponding ``env.eval.override_cfg`` settings; keep it
disabled when training uses the same port. A port conflict produces a warning and collection continues
without the preview. Display resizing and EP labels leave model and replay
images unchanged. Video recording is controlled independently by
``video_cfg.save_video``, which remains ``false`` in these configurations.

Choose which networks to update with ``algorithm.update_epoch`` and
``algorithm.train_actor``. The latter defaults to ``true`` in both ARX
configurations. Set these explicitly when moving between stages:

.. list-table::
   :header-rows: 1

   * - ``algorithm.update_epoch``
     - ``algorithm.train_actor``
     - Behavior
   * - ``0``
     - ``false``
     - Collect and save data without updating actor or Q.
   * - Positive
     - ``false``
     - Update Q once replay meets the training threshold.
   * - Positive
     - ``true``
     - Update Q and actor at the configured ratio once replay meets the training thresholds.

With ``algorithm.train_actor=false``, the actor and entropy temperature alpha
optimizers and their learning rate schedulers are skipped. The Q optimizer,
Q learning rate scheduler and target model EMA continue at their configured
rates. The current RLT MLP policies have separate actor and ``q_head`` parameters,
so the online actor keeps its initialized or resumed weights, while Stage 1
remains frozen. AC continues to generate TD target actions with the online actor
and its ``fixed_std: 0.002`` sampling noise. TD3 continues to use its target
actor, following the configured target model updates. Selecting the reference
branch with ``start_with_actor=false`` controls arm execution and does not
replace those TD target actions with exact reference actions.

When learner updates run, ``train/actor/updated`` reports the mean of the 0/1
indicators for actual actor optimization at each update; it stays at 0 when
only Q is trained. With RLT scheduling enabled,
``train/rlt/actor_updates_run`` reports the actual actor update count for the
iteration. The runner's ``global_step`` counts collection and training
iterations, not optimizer updates.

For the first episode, establish the starting arm and object poses and press
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
   * - ``c`` / ``a``
     - Mark success / failure, hold until the chunk ends, then return to the configured start pose. Ignored while waiting to start.
   * - ``q``
     - Abort and hold without returning. While waiting to start, cancel the wait.

After success or failure, the environment records the terminal transition,
returns smoothly, and holds the reached pose. Reaching the episode step limit
also triggers this return. Prepare the objects for the next episode, then press
``s`` again. While waiting to start, ``r`` only holds the measured pose.

During return, ``g`` cancels the movement and enters drag; ``q`` or ``r`` cancels
it and holds the current measured pose. The keys ``s``, ``b``, ``v``, ``c`` and
``a`` are ignored until return ends. A cancelled return is not retried
automatically. Drag takeover remains active through automatic reset until
``s`` or ``r`` restores hold. Aborting an episode, a fault, or closing the
environment does not initiate a return.

Return completes before the terminating step returns, including when
``auto_reset=false``. With the default ``auto_reset=true``, the next reset waits
for ``s`` even if the completed episode was in the last rollout chunk; that
wait can delay saving the current iteration. A 64-step training rollout is
shorter than the configured 320-step episode limit: ending a rollout alone
does not trigger return.

Keyboard control uses evdev to listen to every readable keyboard on the robot
host. Grant the runtime user read access to the keyboard event devices as described in
:doc:`franka`, then clear any previous single-device selection before launching
RLinf or starting Ray:

.. code-block:: bash

   unset RLINF_KEYBOARD_DEVICE

Press the controls on any keyboard attached to the robot host; typing into an
SSH terminal does not send these device events. The listener periodically
rescans for new readable keyboards, and other keyboards remain usable when one
is unplugged. At least one readable keyboard must be present at startup.
Reconnected or newly attached devices may need read permission granted again.
To restrict input to one device, optionally set ``RLINF_KEYBOARD_DEVICE`` to its
event device path. From a separate terminal, ``--operation
enter_drag`` and ``--operation hold`` reach the service even while inference is
waiting; those service operations do not provide episode labels.

Measured drag trajectories are encoded in the same normalized action domain as
the actor. Valid samples use the existing replay, critic and intervention BC
path. Interrupted or stale transitions are excluded by
``algorithm.replay_buffer.filter_invalid_transitions: true``. A success/failure
label holds the arm immediately; the environment then samples the remaining
chunk intervals at that held pose and records the terminal reward once at the
chunk boundary. These measured hold samples preserve the outcome in replay;
abort, fault and stale-control samples remain invalid. The terminal observation
is captured before return, and return movements do not enter replay. Operator
events, measured steps, and separate return events are saved under
``runner.logger.log_path/arx5_raw`` as JSONL; these records are not a complete
image dataset.

Save, Resume, and Evaluate
--------------------------

Keep a complete RLinf checkpoint to resume the optimizers, target network and
replay data. Replay checkpoint saving waits for pending trajectory writes, then
saves matching metadata, an index and trajectory files under
``actor/sac_components/replay_buffer``. With ``auto_save: true``, this includes
all indexed trajectories persisted to disk, including those that have never
been sampled when collecting with ``algorithm.update_epoch=0``. With
``auto_save: false``, individual trajectories are not automatically written to
disk; checkpoint saving includes the data still available in the retained cache
or on disk. Trajectories evicted from memory before being written to disk are
omitted; restored trajectory and sample counts reflect the saved data. Saving
again after a resume includes both trajectories loaded from the earlier
checkpoint and new trajectories that are still available, using each
trajectory's source.

A failed background write or a missing required trajectory file raises an error
during checkpoint saving. Wait for saving to complete successfully before
archiving the checkpoint; the presence of its directory alone does not establish
completion. This does not restore files missing from older checkpoints or
guarantee pending writes after a forced exit. Transitions still held by the
environment and not yet sent to replay are outside the saved snapshot.

The default saves every ten training epochs. To train Q from a collection
checkpoint, replace the path below with your actual ``global_step_1`` directory.
This example runs one more iteration and saves its result. Q updates still
require ``algorithm.replay_buffer.min_buffer_size: 64``, which counts trajectory
entries. After collecting only the first iteration, the next iteration may
perform no Q updates. This command demonstrates the resume settings; set an
appropriate cumulative iteration limit for the actual Q training run. Run it
from the repository root:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py --config-name "$TRAIN_CONFIG" \
     algorithm.update_epoch=8 \
     algorithm.train_actor=false \
     runner.resume_dir=/path/to/checkpoints/global_step_1 \
     runner.max_epochs=2 \
     runner.save_interval=1

``runner.max_epochs`` is a cumulative iteration limit and must exceed the loaded
``global_step``. Increase it accordingly when resuming a later checkpoint. This
entry point still initializes the real environment and collects rollouts each
iteration before learning from replay. Set ``algorithm.train_actor=true``
explicitly when you are ready to train the actor as well.

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
