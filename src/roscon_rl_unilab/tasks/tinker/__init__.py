"""Register the Tinker Lingzu walk-flat task with UniLab."""
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env

from roscon_rl_unilab.assets import ensure_assets

TASK_NAME = "TinkerWalkFlat"


def make_tinker_walk_flat_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    ensure_assets()
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
registry.register_env(TASK_NAME, make_tinker_walk_flat_env, sim_backend="mujoco")

__all__ = ["TASK_NAME", "make_tinker_walk_flat_env"]
