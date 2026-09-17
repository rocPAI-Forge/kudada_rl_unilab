# roscon_rl_unilab

Tinker Lingzu (Nano1) 10-DoF 双足机器人平地速度跟踪训练仓库，基于
[UniLab](https://github.com/Motphys/UniLab) 的外部任务接入机制构建。

本仓库不复制 UniLab 的 runner/learner/collector：

- `UNILAB_EXTRA_REGISTRY_PACKAGES` 注入 `TinkerWalkFlat` 环境注册；
- Hydra `--config-dir` 注入 `task=walk-flat/mujoco` owner 配置；
- `unilab` 的 `sac` 入口运行 FastSAC 双缓冲异步训练；
- 物理后端固定为 MuJoCo。

## 资产

两套模型都随仓库保存：

- `assets/robots/tinker_lingzu/urdf/`：原始 SolidWorks URDF 与 11 个 STL；
- `assets/robots/tinker_lingzu/mjcf/`：修正后的 MuJoCo MJCF、平地 scene 与 STL。

MJCF 显式保留了 URDF 的质量/惯量、10 个关节限位，并补齐：

- 基座 free joint 和 IMU site；
- 双脚 position/orientation site；
- `kp=40, kv=2` 的位置执行器及 12/20 Nm 力矩限制；
- 默认蹲姿 `home` keyframe；
- 足底与膝盖碰撞几何。

## 安装（AMD ROCm）

要求 Python 3.10–3.13；当前 MI210 环境建议：

```bash
uv sync --extra rocm
```

## 训练

默认配方复刻原 `tinkerbot_rl` 的规模：2048 env、10000 iter、batch 2048、
每步 4 次更新、bf16 AMP、每 1000 iter 保存 checkpoint。

```bash
export HIP_VISIBLE_DEVICES=0
uv run --extra rocm roscon-train --algo sac --task walk-flat --sim mujoco
```

短程冒烟：

```bash
export HIP_VISIBLE_DEVICES=0
uv run --extra rocm roscon-train --algo sac --task walk-flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

回放最近一次 run：

```bash
uv run --extra rocm roscon-eval --algo sac --task walk-flat --sim mujoco --load-run -1
```

从已有 checkpoint 续训（默认加载归档的最终策略）：

```bash
export HIP_VISIBLE_DEVICES=0
uv run --extra rocm roscon-continue --algo sac --task walk-flat --sim mujoco \
  --load-run roscon/train/2026-09-17_11-54-30_mujoco \
  algo.max_iterations=10000 training.log_root=runs
```

步态视频（OSMesa，无显示）：

```bash
export HIP_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa
TINKER_RUN_DIR=roscon/train/2026-09-17_11-54-30_mujoco \
  uv run --extra rocm python scripts/record_gait_video.py
```

归档的训练资产、步态视频、以及与 MotrixLab/`tinkerbot_rl` 的吞吐/延迟/精度对比见
[`roscon/COMPARISON.md`](roscon/COMPARISON.md)。

## 测试

```bash
uv run --extra rocm pytest -q
```

任务目标与原 MotrixLab 版保持一致：跟踪 `[vx, vy, yaw_rate]` 指令，包含站立指令、
左右脚相位、姿态与动作平滑奖励，以及逐步增加惩罚权重的课程学习。由于仿真后端从
MotrixSim 改为 MuJoCo，训练结果不保证逐轨迹数值一致。
