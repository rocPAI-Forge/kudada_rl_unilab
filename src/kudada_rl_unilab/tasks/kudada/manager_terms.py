"""Kudada aliases for UniLab's robot-agnostic G1 walk terms.

The upstream implementations operate only on named sensors, commands, and the
configured robot entity; this package supplies those names in its MJCF/YAML.
Keeping local aliases gives the external owner a stable import surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.tasks.locomotion.g1.manager_terms import (
    G1GaitPhase as KudadaGaitPhase,
    G1PenaltyCurriculum as KudadaPenaltyCurriculum,
    G1VelocityCommand as KudadaVelocityCommand,
    G1VelocityCommandCfg as KudadaVelocityCommandCfg,
    _SensorTerm,
    _state,
    feet_phase,
    g1_tilt_exceeded as tilt_exceeded,
    penalty_close_feet_xy,
    penalty_feet_ori,
    weighted_pose,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv

__all__ = [
    "HeightCommand",
    "HeightCommandCfg",
    "KudadaGaitPhase",
    "KudadaPenaltyCurriculum",
    "KudadaVelocityCommand",
    "KudadaVelocityCommandCfg",
    "command_gated_feet_phase",
    "feet_phase",
    "penalty_close_feet_xy",
    "penalty_stand_ang_vel",
    "penalty_stand_lin_vel",
    "penalty_stand_pose",
    "penalty_feet_ori",
    "penalty_base_height_error",
    "penalty_wide_feet_xy",
    "tilt_exceeded",
    "track_base_height",
    "track_crouch_pose",
    "penalty_crouch_pose_error",
    "weighted_pose",
]


@dataclass(kw_only=True)
class HeightCommandCfg(CommandTermCfg):
    """Scalar base-height command resampled with the rest of the gait MDP."""

    ranges: tuple[float, float] = (0.18, 0.28)

    def build(self, env: ManagerBasedRlEnv) -> HeightCommand:
        return HeightCommand(self, env)


class HeightCommand(CommandTerm):
    cfg: HeightCommandCfg

    def __init__(self, cfg: HeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        lo, hi = (float(cfg.ranges[0]), float(cfg.ranges[1]))
        if lo > hi:
            raise ValueError(f"height ranges lower {lo} exceeds upper {hi}")
        self._lo = lo
        self._hi = hi
        self._command = np.zeros((self.num_envs, 1), dtype=get_global_dtype())

    @property
    def command(self) -> np.ndarray:
        return self._command

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._command[env_ids, 0] = self._env.rng.uniform(self._lo, self._hi, size=len(env_ids))

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        del env_ids

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids


class track_base_height(ManagerTermBase):
    """exp(-(z - h*)^2 / sigma) using the robot root height and a height command."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name", "height")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("track_base_height command_name must be a non-empty string")
        sigma = float(cfg.params.get("tracking_sigma", 0.01))
        if sigma <= 0.0:
            raise ValueError("track_base_height tracking_sigma must be positive")
        self._command_name = command_name
        self._sigma = sigma
        self._robot = env.scene["robot"]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        height = np.asarray(self._robot.data.root_link_pos_w[:, 2], dtype=get_global_dtype())
        target = env.command_manager.get_command(self._command_name).reshape(env.num_envs)
        return np.exp(-np.square(height - target) / self._sigma)


class penalty_base_height_error(ManagerTermBase):
    """Squared base-height error; pair with a negative reward weight."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._command_name = str(cfg.params.get("command_name", "height"))
        self._robot = env.scene["robot"]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        height = np.asarray(self._robot.data.root_link_pos_w[:, 2], dtype=get_global_dtype())
        target = env.command_manager.get_command(self._command_name).reshape(env.num_envs)
        return np.square(height - target)


def _command_rows(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = np.asarray(env.command_manager.get_command(command_name), dtype=get_global_dtype())
    return command.reshape(env.num_envs, -1)


def _command_norm(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = _command_rows(env, command_name)
    width = min(3, command.shape[1])
    return np.linalg.norm(command[:, :width], axis=1)


class command_gated_feet_phase(feet_phase):
    """Swing-foot reward only while the twist command asks for motion.

    The upstream ``feet_phase`` gate uses measured forward speed. Under a zero
    command that lets a small drift turn the gait reward on and keep the robot
    walking. This gate follows the command norm instead.
    """

    _allowed_params = feet_phase._allowed_params | frozenset({"command_threshold"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        self._command_threshold = float(cfg.params.get("command_threshold", 0.1))
        if self._command_threshold < 0.0:
            raise ValueError("command_gated_feet_phase command_threshold must be non-negative")
        super().__init__(cfg, env)

    def _gate(self, env: ManagerBasedRlEnv) -> np.ndarray:
        moving = _command_norm(env, self._command_name) >= self._command_threshold
        return np.asarray(moving, dtype=get_global_dtype())


class _StandingCommandTerm(_SensorTerm):
    """Sensor reward that is active only while the twist command is near zero."""

    _allowed_params = frozenset({"command_name", "command_threshold"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        command_name = cfg.params.get("command_name", "twist")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError(f"{self.name} command_name must be a non-empty string")
        self._command_name = command_name
        self._command_threshold = float(cfg.params.get("command_threshold", 0.1))
        if self._command_threshold < 0.0:
            raise ValueError(f"{self.name} command_threshold must be non-negative")

    def _standing_mask(self, env: ManagerBasedRlEnv) -> np.ndarray:
        standing = _command_norm(env, self._command_name) < self._command_threshold
        return np.asarray(standing, dtype=get_global_dtype())


class penalty_stand_lin_vel(_StandingCommandTerm):
    """Squared horizontal base speed while the command asks the robot to stand."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._linvel = self._bind(("pelvis_local_linvel",))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = _state(
            self.name,
            "pelvis linear velocity",
            self._read(self._linvel, self.name),
            (env.num_envs, 3),
        )
        speed_sq = np.square(values[:, 0]) + np.square(values[:, 1])
        return np.asarray(speed_sq * self._standing_mask(env), dtype=get_global_dtype())


class penalty_stand_ang_vel(_StandingCommandTerm):
    """Squared yaw rate while the command asks the robot to stand."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gyro = self._bind(("torso_gyro",))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = _state(
            self.name, "torso gyro", self._read(self._gyro, self.name), (env.num_envs, 3)
        )
        return np.asarray(np.square(values[:, 2]) * self._standing_mask(env), dtype=get_global_dtype())


class penalty_stand_pose(ManagerTermBase):
    """Joint-pose error from the default pose, active only for a near-zero command."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        command_name = cfg.params.get("command_name", "twist")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("penalty_stand_pose command_name must be a non-empty string")
        self._command_name = command_name
        self._command_threshold = float(cfg.params.get("command_threshold", 0.1))
        if self._command_threshold < 0.0:
            raise ValueError("penalty_stand_pose command_threshold must be non-negative")
        weights = cfg.params.get("pose_weights", [1.0] * 10)
        self._pose_weights = np.asarray(weights, dtype=get_global_dtype())
        if self._pose_weights.shape != (10,):
            raise ValueError("penalty_stand_pose pose_weights must have length 10")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        pose_error = weighted_pose(env, self._pose_weights)
        standing = _command_norm(env, self._command_name) < self._command_threshold
        return np.asarray(pose_error * standing, dtype=get_global_dtype())


class penalty_wide_feet_xy(_SensorTerm):
    """Penalty for excessive lateral foot separation.

    Forward/back separation is deliberately ignored so normal stride length is
    not mistaken for a wide stance.
    """

    _allowed_params = frozenset({"max_width"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._max_width = float(cfg.params.get("max_width", 0.19))
        if self._max_width <= 0.0:
            raise ValueError("penalty_wide_feet_xy max_width must be positive")
        self._feet_pos = self._bind(("left_foot_pos", "right_foot_pos"))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        values = _state(
            self.name, "foot position", self._read(self._feet_pos, self.name), (env.num_envs, 6)
        )
        feet_dist = np.abs(values[:, 1] - values[:, 4])
        return np.asarray(
            np.square(np.maximum(feet_dist - self._max_width, 0.0)),
            dtype=get_global_dtype(),
        )


class track_crouch_pose(ManagerTermBase):
    """Reward bent-knee / neutral-roll crouch joint targets instead of abducted legs."""

    _allowed_params = frozenset({"joint_targets", "tracking_sigma"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        targets = cfg.params.get(
            "joint_targets",
            [0.0, 0.0, -0.55, 1.05, -0.25, 0.0, 0.0, -0.55, 1.05, -0.25],
        )
        self._targets = np.asarray(targets, dtype=get_global_dtype())
        if self._targets.shape != (10,):
            raise ValueError(f"track_crouch_pose joint_targets must have length 10, got {self._targets.shape}")
        sigma = float(cfg.params.get("tracking_sigma", 0.12))
        if sigma <= 0.0:
            raise ValueError("track_crouch_pose tracking_sigma must be positive")
        self._sigma = sigma
        self._robot = env.scene["robot"]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        joint_pos = np.asarray(self._robot.data.joint_pos, dtype=get_global_dtype())
        err = np.sum(np.square(joint_pos - self._targets[None, :]), axis=1)
        return np.exp(-err / self._sigma)


class penalty_crouch_pose_error(ManagerTermBase):
    """Dense weighted joint error for learning a crouch without reward saturation."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        targets = cfg.params.get(
            "joint_targets",
            [0.0, 0.0, -0.8, 1.35, -0.55, 0.0, 0.0, -0.8, 1.35, -0.55],
        )
        weights = cfg.params.get(
            "joint_weights",
            [1.0, 10.0, 3.0, 3.0, 2.0, 1.0, 10.0, 3.0, 3.0, 2.0],
        )
        self._targets = np.asarray(targets, dtype=get_global_dtype())
        self._weights = np.asarray(weights, dtype=get_global_dtype())
        if self._targets.shape != (10,) or self._weights.shape != (10,):
            raise ValueError("penalty_crouch_pose_error targets and weights must have length 10")
        self._robot = env.scene["robot"]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        joint_pos = np.asarray(self._robot.data.joint_pos, dtype=get_global_dtype())
        return np.sum(self._weights[None, :] * np.square(joint_pos - self._targets[None, :]), axis=1)
