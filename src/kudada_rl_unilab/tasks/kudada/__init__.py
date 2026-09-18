"""Register Kudada locomotion tasks with UniLab."""
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env

from kudada_rl_unilab.assets import ensure_assets

TASK_NAME = "KudadaWalkFlat"
TASK_NAMES = (
    "KudadaWalkFlat",
    "KudadaCrouchWalk",
    "KudadaLateralWalk",
    "KudadaMarchRecover",
)
LEGACY_TASK_NAMES = {
    "TinkerWalkFlat": "KudadaWalkFlat",
    "TinkerCrouchWalk": "KudadaCrouchWalk",
    "TinkerLateralWalk": "KudadaLateralWalk",
    "TinkerMarchRecover": "KudadaMarchRecover",
}


def make_kudada_walk_flat_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    ensure_assets()
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


for _name in TASK_NAMES:
    registry.register_env_config(_name, ManagerBasedRlEnvCfg)
    registry.register_env(_name, make_kudada_walk_flat_env, sim_backend="mujoco")
for _legacy in LEGACY_TASK_NAMES:
    registry.register_env_config(_legacy, ManagerBasedRlEnvCfg)
    registry.register_env(_legacy, make_kudada_walk_flat_env, sim_backend="mujoco")

make_tinker_walk_flat_env = make_kudada_walk_flat_env

__all__ = [
    "TASK_NAME",
    "TASK_NAMES",
    "LEGACY_TASK_NAMES",
    "make_kudada_walk_flat_env",
    "make_tinker_walk_flat_env",
]
