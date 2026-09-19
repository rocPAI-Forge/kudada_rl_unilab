# 合格步态资产怎么放

每个**已验收**步态占 `gaits/<task>/` 一个目录，五类东西都在里面：

| 你要的东西 | 目录里的文件 | 说明 |
| --- | --- | --- |
| 训练策略 / 任务设计 | `task.yaml`（walk-flat 另有 `task_base.yaml`） | Hydra owner 配置：指令范围、观测、事件、奖励权重 |
| 训练参数快照 | `run_config.json` | 该次训练实际展开后的完整 Hydra 配置（含 algo / env / reward） |
| 训练结果摘要 | `run_summary.json` | 迭代数、回报、吞吐、seed、墙钟时间 |
| 训练好的 policy | `model_10000.pt` | FastSAC 最终权重（10k iter） |
| 录制视频 | `play.mp4` | OSMesa 回放，无版本号；单步态视频只在这里留一份 |

奖励函数的**实现**不按步态复制多份，统一在：

- `src/kudada_rl_unilab/tasks/kudada/manager_terms.py`（高度指令、蹲姿、横向脚距等）
- `src/kudada_rl_unilab/conf/sac/task/<task>/mujoco.yaml`（权重与参数，是 `task.yaml` 的源文件）

训练时 `runs/` 是工作目录，gitignore，不入库。只有验收后的上述文件拷进 `gaits/`。

README 用的外形图和四步态同屏演示在 `renders/`，不在本目录重复存放单步态 mp4。

## 四个合格步态

| 目录 | 任务名 | 策略要点 | 回放 |
| --- | --- | --- | --- |
| `gaits/walk-flat/` | `KudadaWalkFlat` | 平地跟踪 `[vx, vy, yaw]`；Kudada 从零训练后续训一轮 | `play.mp4` |
| `gaits/crouch-walk/` | `KudadaCrouchWalk` | 屈膝降重心：蹲姿关节误差 + 横向脚距惩罚 + action scale 0.8 | `play.mp4` |
| `gaits/lateral-walk/` | `KudadaLateralWalk` | 主指令是 `vy`，限制 `vx`/`yaw` | `play.mp4` |
| `gaits/march-recover/` | `KudadaMarchRecover` | 原地高抬腿 + 训练中随机推，回放固定侧向推力 | `play.mp4` |

旧 Tinker 命名下的训练目录与 checkpoint 已移除；`gaits/` 里四套均由 `Kudada*` 任务重新训练产生。历史性能对比数据在 [`docs/COMPARISON.md`](../docs/COMPARISON.md)。

外形截图：[`renders/kudada_views.png`](../renders/kudada_views.png)（正/侧/背三视图，项目 README 引用）。
一次同屏演示：[`renders/kudada_gaits_2x2.mp4`](../renders/kudada_gaits_2x2.mp4)（四段 `play.mp4` 2×2 拼接）。

## 怎么从归档复现

```bash
# 例如蹲走
uv run --extra rocm kudada-eval --algo sac --task crouch-walk --sim mujoco \
  --load-run gaits/crouch-walk

export HIP_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa
KUDADA_RUN_DIR=gaits/crouch-walk KUDADA_TASK=crouch-walk \
  KUDADA_VX=0.35 KUDADA_HEIGHT=0.20 \
  uv run --extra rocm python scripts/record_gait_video.py
```
