#!/usr/bin/env python3
"""Roll out the trained FastSAC policy and software-render a gait video.

UniLab's built-in recorder needs a working MuJoCo GL context at env-build
time. On this MI210 host that path is skipped, so we cache physics snapshots
during policy rollout and render them afterwards with OSMesa.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("UNILAB_EXTRA_REGISTRY_PACKAGES", "kudada_rl_unilab.tasks")

import numpy as np

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base.config_adapter import create_env
from unilab.cli import package_root
from unilab.scripts.train_offpolicy import build_offpolicy_play_env_cfg_override
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_sac_playback_session,
)
from kudada_rl_unilab.conf_searchpath import register_conf_search_path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "roscon/train/2026-09-17_11-54-30_mujoco"
DEFAULT_STEPS = 400
FPS = 50


def _env(name: str, default: str | None = None) -> str | None:
    for prefix in ("KUDADA_", "TINKER_"):
        key = prefix + name
        if key in os.environ:
            return os.environ[key]
    return default


def compose_cfg(run_dir: Path, task: str):
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "sac"), version_base="1.3"):
        overrides = [
            f"task={task}/mujoco",
            f"algo.load_run={run_dir}",
            "training.play_env_num=1",
            "training.export_onnx=false",
            "env.events.reset_root_state_uniform.params.pose_range.yaw=[0,0]",
            "env.events.reset_root_state_uniform.params.velocity_range.x=[0,0]",
            "env.events.reset_root_state_uniform.params.velocity_range.y=[0,0]",
            "env.events.reset_root_state_uniform.params.velocity_range.z=[0,0]",
            "env.events.reset_root_state_uniform.params.velocity_range.roll=[0,0]",
            "env.events.reset_root_state_uniform.params.velocity_range.pitch=[0,0]",
            "env.events.reset_root_state_uniform.params.velocity_range.yaw=[0,0]",
            "env.events.pd_gains.params.kp_range=[1.0,1.0]",
            "env.events.pd_gains.params.kd_range=[1.0,1.0]",
        ]
        if task == "march-recover":
            overrides.append("env.events.push_robot.interval_range_s=[100,100]")
        return compose(
            "config",
            overrides=overrides,
        )


def _force_commands(session, vx: float, vy: float, yaw: float, standing: bool, height: float | None) -> None:
    twist = session.env.command_manager.get_term("twist")
    twist.vel_command_b[:] = (vx, vy, yaw)
    twist.is_standing_env[:] = standing
    if height is not None:
        session.env.command_manager.get_term("height").command[:] = height


def rollout(
    run_dir: Path,
    steps: int,
    task: str,
    cmd,
    standing: bool,
    height: float | None,
    push_step: int | None,
    push_y: float,
) -> np.ndarray:
    cfg = compose_cfg(run_dir, task)
    override = build_offpolicy_play_env_cfg_override("sac", cfg)
    session, _, ckpt = create_sac_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task=str(cfg.training.task_name),
            load_run=str(run_dir),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name=str(cfg.algo.algo_log_name),
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        env_factory=lambda n: create_env(cfg, num_envs=n, env_cfg_override=override),
        root_dir=REPO,
        device="cuda",
        algo_name="sac",
    )
    if ckpt is None:
        raise SystemExit("checkpoint was not loaded")
    session.reset()
    vx, vy, yaw = cmd
    _force_commands(session, vx, vy, yaw, standing, height)
    print(f"forced command vx={vx} vy={vy} yaw={yaw} standing={standing} height={height}")
    snaps = [session.physics_state().copy()]
    for step in range(steps):
        _force_commands(session, vx, vy, yaw, standing, height)
        if push_step is not None and step == push_step:
            robot = session.env.scene["robot"]
            robot.apply_root_velocity_delta_to_sim(
                np.array([[0.0, push_y, 0.0]], dtype=np.float32),
                None,
            )
            print(f"fixed lateral push at t={step * 0.02:.2f}s: dv_y={push_y} m/s")
        session.step_once()
        snaps.append(session.physics_state().copy())
    arr = np.stack(snaps, axis=0)
    print(f"captured {arr.shape} snapshots from {ckpt}")
    return arr


def _apply_snapshot(model, data, snapshot: np.ndarray) -> None:
    import mujoco

    flat = np.asarray(snapshot, dtype=np.float64).reshape(-1)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    expected = mujoco.mj_stateSize(model, spec)
    if flat.size != expected:
        raise ValueError(f"snapshot size {flat.size} != FULLPHYSICS {expected}")
    mujoco.mj_setState(model, data, flat, spec)
    mujoco.mj_forward(model, data)


def render_video(snapshots: np.ndarray, out_path: Path) -> None:
    import imageio.v2 as imageio
    import mujoco

    xml = REPO / "assets/robots/kudada/mjcf/scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=960)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 140.0
    cam.elevation = -20.0
    cam.distance = 1.6
    frames = []
    for i, snap in enumerate(snapshots):
        _apply_snapshot(model, data, snap)
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
        if i % 50 == 0:
            print(f"rendered {i}/{len(snapshots)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"wrote {out_path} ({len(frames)} frames, {FPS} fps)")


def main() -> None:
    run_dir = Path(_env("RUN_DIR", str(DEFAULT_RUN))).resolve()
    steps = int(_env("PLAY_STEPS", str(DEFAULT_STEPS)))
    mode = _env("RECORD_MODE", "all")
    task = _env("TASK", "walk-flat")
    vx = float(_env("VX", "0.6"))
    vy = float(_env("VY", "0.0"))
    yaw = float(_env("YAW", "0.0"))
    standing = _env("STANDING", "0") in {"1", "true", "True"}
    height_raw = _env("HEIGHT")
    height = float(height_raw) if height_raw else None
    push_step_raw = _env("PUSH_STEP")
    push_step = int(push_step_raw) if push_step_raw else None
    push_y = float(_env("PUSH_Y", "0.6"))
    snap_path = run_dir / "play_qpos.npy"
    video_path = Path(_env("VIDEO", str(run_dir / "play_video.mp4")))
    if mode in {"rollout", "all"}:
        snapshots = rollout(
            run_dir, steps, task, (vx, vy, yaw), standing, height, push_step, push_y
        )
        np.save(snap_path, snapshots)
        print(f"saved {snap_path}")
    else:
        snapshots = np.load(snap_path)
    if mode in {"render", "all"}:
        render_video(snapshots, video_path)


if __name__ == "__main__":
    main()
