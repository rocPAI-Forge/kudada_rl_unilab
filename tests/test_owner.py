from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.cli import package_root
from unilab.envs import ManagerBasedRlEnvCfg

ROOT = Path(__file__).resolve().parents[1]
JOINTS = (
    "left_leg_hip_yaw_joint", "left_leg_hip_roll_joint", "left_leg_hip_pitch_joint",
    "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint",
    "right_leg_hip_yaw_joint", "right_leg_hip_roll_joint", "right_leg_hip_pitch_joint",
    "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint",
)


def materialize():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "sac"), version_base="1.3"):
        cfg = compose("config", overrides=["task=walk-flat/mujoco"])
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="sac").build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("KudadaWalkFlat")
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return cfg, env_cfg


def test_owner_materializes():
    cfg, env_cfg = materialize()
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    assert cfg.training.task_name == "KudadaWalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 10000
    assert cfg.algo.batch_size == 2048
    assert cfg.algo.updates_per_step == 4
    assert env_cfg.sim_dt == 0.005
    assert env_cfg.ctrl_dt == 0.02
    assert env_cfg.max_episode_seconds == 20.0
    assert env_cfg.scene.model_file.endswith("assets/robots/kudada/mjcf/scene_flat.xml")
    robot = env_cfg.scene.entities["robot"]
    assert tuple(robot.joint_names) == JOINTS
    assert tuple(robot.actuator_names) == JOINTS
    assert env_cfg.actions["joint_pos"].scale == 0.5
    assert tuple(env_cfg.commands["twist"].ranges.lin_vel_x) == (-1.0, 1.0)
    assert set(env_cfg.observations) == {"policy", "critic"}
    assert set(env_cfg.rewards) >= {"tracking_lin_vel", "feet_phase", "alive"}


def test_registry_exposes_mujoco_task():
    registry.ensure_registries()
    registered = registry.list_registered_envs()
    assert "KudadaWalkFlat" in registered
    assert "mujoco" in registered["KudadaWalkFlat"]["available_backends"]
    for name in ("KudadaCrouchWalk", "KudadaLateralWalk", "KudadaMarchRecover"):
        assert name in registered
        assert "mujoco" in registered[name]["available_backends"]
    for name in ("TinkerWalkFlat", "TinkerCrouchWalk", "TinkerLateralWalk", "TinkerMarchRecover"):
        assert name in registered


def _materialize_task(task: str, env_name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "sac"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={task}/mujoco"])
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="sac").build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config(env_name)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    return cfg, env_cfg


def test_crouch_walk_adds_height_command():
    cfg, env_cfg = _materialize_task("crouch-walk", "KudadaCrouchWalk")
    assert cfg.training.task_name == "KudadaCrouchWalk"
    assert "height" in env_cfg.commands
    assert "tracking_height" in env_cfg.rewards
    assert "track_crouch_pose" in env_cfg.rewards
    assert "penalty_wide_feet" in env_cfg.rewards
    assert env_cfg.rewards["pose"].params["pose_weights"][1] >= 5.0


def test_lateral_walk_narrows_forward_speed():
    cfg, env_cfg = _materialize_task("lateral-walk", "KudadaLateralWalk")
    assert cfg.training.task_name == "KudadaLateralWalk"
    assert tuple(env_cfg.commands["twist"].ranges.lin_vel_y) == (-0.4, 0.4)
    assert env_cfg.commands["twist"].ranges.lin_vel_x[1] <= 0.15


def test_march_recover_pushes_during_episode():
    cfg, env_cfg = _materialize_task("march-recover", "KudadaMarchRecover")
    assert cfg.training.task_name == "KudadaMarchRecover"
    assert env_cfg.events["push_robot"].mode == "interval"
    assert env_cfg.commands["twist"].rel_standing_envs == 0.0
    assert tuple(env_cfg.commands["twist"].ranges.lin_vel_x) == (0.0, 0.0)
    assert env_cfg.rewards["feet_phase"].params["swing_height"] >= 0.1
