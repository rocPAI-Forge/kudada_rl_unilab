# 合格步态资产怎么放

每个**已验收**步态占 `roscon/gaits/<task>/` 一个目录，五类东西都在里面：

| 你要的东西 | 目录里的文件 | 说明 |
| --- | --- | --- |
| 训练策略 / 任务设计 | `task.yaml`（walk-flat 另有 `task_base.yaml`） | Hydra owner 配置：指令范围、观测、事件、奖励权重 |
| 训练参数快照 | `run_config.json` | 该次训练实际展开后的完整 Hydra 配置（含 algo / env / reward） |
| 训练结果摘要 | `run_summary.json` | 迭代数、回报、吞吐、seed、墙钟时间 |
| 训练好的 policy | `model_10000.pt` | FastSAC 最终权重（10k iter） |
| 录制视频 | `play.mp4` | OSMesa 回放，无版本号 |

奖励函数的**实现**不按步态复制多份，统一在：

- `src/roscon_rl_unilab/tasks/tinker/manager_terms.py`（高度指令、蹲姿、横向脚距等）
- `src/roscon_rl_unilab/conf/sac/task/<task>/mujoco.yaml`（权重与参数，是 `task.yaml` 的源文件）

训练时 `runs/` 是工作目录，gitignore，不入库。只有验收后的上述文件拷进 `roscon/gaits/`。

## 四个合格步态

| 目录 | 任务名 | 策略要点 | 视频也拷了一份到 |
| --- | --- | --- | --- |
| `gaits/walk-flat/` | `TinkerWalkFlat` | 平地跟踪 `[vx, vy, yaw]`；最终是 continue-2（lin/ang 跟踪加强） | `roscon/renders/tinker_walk.mp4` |
| `gaits/crouch-walk/` | `TinkerCrouchWalk` | 屈膝降重心：蹲姿关节误差 + 横向脚距惩罚 + action scale 0.8 | `roscon/renders/tinker_crouch_walk.mp4` |
| `gaits/lateral-walk/` | `TinkerLateralWalk` | 主指令是 `vy`，限制 `vx`/`yaw` | `roscon/renders/tinker_lateral_walk.mp4` |
| `gaits/march-recover/` | `TinkerMarchRecover` | 原地高抬腿 + 训练中随机推，回放固定侧向推力 | `roscon/renders/tinker_march_recover.mp4` |

walk-flat 的对比实验（第一轮 10k、continue-1、continue-2 全量 checkpoint）仍放在 `roscon/train/`，见 `COMPARISON.md`。`gaits/walk-flat/` 只收**最终合格**那一套。

外形截图保存在 `roscon/renders/`：

- `tinker_views.png`：机器人多视角外形
- `tinker_walk_unilab_r2_end.png`：walk-flat 合格策略回放末帧（外形 + 步态姿态）

## 怎么从归档复现

```bash
# 例如蹲走
uv run --extra rocm roscon-eval --algo sac --task crouch-walk --sim mujoco \
  --load-run roscon/gaits/crouch-walk

export HIP_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa
TINKER_RUN_DIR=roscon/gaits/crouch-walk TINKER_TASK=crouch-walk \
  TINKER_VX=0.35 TINKER_HEIGHT=0.20 \
  uv run --extra rocm python scripts/record_gait_video.py
```
