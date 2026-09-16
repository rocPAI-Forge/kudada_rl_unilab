"""Tinker aliases for UniLab's robot-agnostic G1 walk terms.

The upstream implementations operate only on named sensors, commands, and the
configured robot entity; this package supplies those names in its MJCF/YAML.
Keeping local aliases gives the external owner a stable import surface.
"""
from unilab.tasks.locomotion.g1.manager_terms import (
    G1GaitPhase as TinkerGaitPhase,
    G1PenaltyCurriculum as TinkerPenaltyCurriculum,
    G1VelocityCommand as TinkerVelocityCommand,
    G1VelocityCommandCfg as TinkerVelocityCommandCfg,
    feet_phase,
    g1_tilt_exceeded as tilt_exceeded,
    penalty_close_feet_xy,
    penalty_feet_ori,
    weighted_pose,
)

__all__ = [
    "TinkerGaitPhase",
    "TinkerPenaltyCurriculum",
    "TinkerVelocityCommand",
    "TinkerVelocityCommandCfg",
    "feet_phase",
    "penalty_close_feet_xy",
    "penalty_feet_ori",
    "tilt_exceeded",
    "weighted_pose",
]
