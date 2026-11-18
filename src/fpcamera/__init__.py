"""fpcamera: trajectory-controlled camera-motion adapter for the frozen Wan2.2-TI2V-5B image-to-video model.

Sub-modules (see ``docs/CODEBASE_GUIDE.md`` for the full map):
geometry/data  ``flow_geometry``, ``motion_sampling``, ``flow_cache`` / ``flow_cache_v2``;
model          ``control_lora``, ``flow_control`` (adapter + guide), ``wan`` (pinned frozen loading);
training       ``flow_train_v2``, ``checkpointing``, ``session_deadline``;
inference      ``flow_infer``, ``flow_anchor``, ``flow_guidance``, ``strength``, ``trajectory_direction``;
evaluation     ``flow_validate_v2``, ``reveal_eval``, ``border_metrics``, ``visual_metrics``.
"""

__version__ = "0.1.0"
