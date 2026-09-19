# 合格步态资产怎么放

每个**已验收**步态占 `gaits/<task>/` 一个目录，五类东西都在里面：

| 你要的东西 | 目录里的文件 | 说明 |
| --- | --- | --- |
| 训练策略 / 任务设计 | `task.yaml`（walk-flat 另有 `task_base.yaml`） | Hydra owner 配置：指令范围、观测、事件、奖励权重 |
| 训练参数快照 | `run_config.json` | 该次训练实际展开后的完整 Hydra 配置（含 algo / env / reward） |
| 训练结果摘要 | `run_summary.json` | 迭代数、回报、吞吐、seed、墙钟时间 |
| 训练好的 policy | `model_10000.pt` | FastSAC 最终权重（10k iter） |
| 录制视频 | `play.mp4` | OSMesa 回放；单步态视频只在这里留一份 |

奖励函数的**实现**不按步态复制多份，统一在：

- `src/kudada_rl_unilab/tasks/kudada/manager_terms.py`（高度指令、蹲姿、横向脚距等）
- `src/kudada_rl_unilab/conf/sac/task/<task>/mujoco.yaml`（权重与参数，是 `task.yaml` 的源文件）

训练时 `runs/` 是工作目录，gitignore，不入库。验收后从
`runs/<TaskName>/<stamp>_mujoco/` 覆盖拷进本目录（见仓库根 README「验收入库」）。

**每个步态只保留当前验收的一份权重。** 新结果覆盖 `model_10000.pt`，不要并列
`model_v7.pt` / `model_20000.pt`。仓库随已发布步态数线性增长（约 16 MB/套），
不随训练次数增长。步态明显变多之后再考虑 Git LFS。

README 用的外形图和四步态同屏演示在 `renders/`，不在本目录重复存放单步态 mp4。

## 四个合格步态

| 目录 | 任务名 | 策略要点 | 回放 |
| --- | --- | --- | --- |
| `gaits/walk-flat/` | `KudadaWalkFlat` | 平地跟踪 `[vx, vy, yaw]`；从零训练后续训一轮 | `play.mp4` |
| `gaits/crouch-walk/` | `KudadaCrouchWalk` | 屈膝降重心：蹲姿关节误差 + 横向脚距惩罚 + action scale 0.8 | `play.mp4` |
| `gaits/lateral-walk/` | `KudadaLateralWalk` | 主指令是 `vy`，限制 `vx`/`yaw` | `play.mp4` |
| `gaits/march-recover/` | `KudadaMarchRecover` | 原地高抬腿 + 训练中随机推，回放固定侧向推力 | `play.mp4` |

旧 Tinker 命名下的 checkpoint 已移除；这里四套均由 `Kudada*` 任务重新训练产生。
历史对比在 [`docs/COMPARISON.md`](../docs/COMPARISON.md)。

外形：[`renders/kudada_views.png`](../renders/kudada_views.png)。
同屏演示：[`renders/kudada_gaits_2x2.mp4`](../renders/kudada_gaits_2x2.mp4)。

## 从归档复现

`kudada-eval --load-run -1` 指向本机最近一次 `runs/`，不是这里。评估入库策略：

```bash
export HIP_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa
uv run --extra rocm kudada-eval --algo sac --task crouch-walk --sim mujoco \
  --load-run gaits/crouch-walk
```

无头录视频（默认文件名是 `play_video.mp4`，覆盖归档时要设 `KUDADA_VIDEO`）：

```bash
export HIP_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa
KUDADA_RUN_DIR=gaits/crouch-walk KUDADA_TASK=crouch-walk \
  KUDADA_VX=0.35 KUDADA_HEIGHT=0.20 \
  KUDADA_VIDEO=gaits/crouch-walk/play.mp4 \
  uv run --extra rocm python scripts/record_gait_video.py
```

其余三个任务的完整命令在仓库根 README「录视频」一节。
