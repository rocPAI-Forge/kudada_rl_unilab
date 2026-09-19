# Kudada walk-flat: MotrixLab vs UniLab

Same 10-DoF Kudada velocity-tracking task, same FastSAC budget
(2048 envs × 10k iter, batch 2048, 4 updates/step), same MI210, seed 1,
`sim_dt=5 ms`, `ctrl_dt=20 ms`. One run each for the matched budget;
UniLab then got two continue-train rounds on top of the first checkpoint.

Throughput numbers use the first UniLab run (`2026-09-17_09-29-01_mujoco`).
Tracking numbers use a shared eval: 16 envs × 500 steps (10 s), command
`vx=0.6 / vy=0 / yaw=0`, yaw reset = 0, deterministic actor, body-frame
velocity, no command resample.

MDP is **not** identical: UniLab is MuJoCo 3.11 (MJCF) vs MotrixSim (URDF),
41 vs 43 actor dims (UniLab policy has no lin-vel), tilt/height terminate
vs base/knee contact. Training episode return is not an accuracy metric.

## Stack

| Layer | tinkerbot_rl | kudada_rl_unilab |
|---|---|---|
| Framework | MotrixLab FastSAC async | UniLab 1.3 + unilab-rl 1.2 FastSAC |
| Physics | MotrixSim | MuJoCo 3.11 mjbatch |
| Collector inference | GPU cuda:0, AMP fp16 | learner-owned, no collector CUDA |
| Learner | GPU, compile, AMP bf16 | same |
| Actor / critic | 512 / 768, C51 101 atoms | same |
| Action | joint pos residual, scale 0.5, kp=40 kv=2 | same |
| Commands | ±1 m/s, ±1 rad/s, 20% stand, 10 s resample | same ranges / stand / resample |

## Throughput (matched 10k-iter budget)

| Metric | tinkerbot_rl | kudada_rl_unilab (first run) |
|---|---|---|
| Wall clock | 232 s (3m52s) | 318 s (5m17s) |
| Steady env-steps/s | 109,222 | 77,096 (Motrix 1.42×) |
| Learner samples/s | ~437 k | 308 k |
| Policy infer / env step / learner / iter wall | 1.14 / 15.43 / 15.82 / 18.4 ms | 1.18 / 11.0 / 25.43 / 27.03 ms |

UniLab physics step is faster (~11 ms vs ~15 ms); the overlapped cycle is
slower because the learner update is ~25 ms vs ~16 ms and collector
inference stays on the learner process.

## Aligned tracking eval

| Metric | Motrix (10k) | UniLab first 10k | UniLab continue 1 (speed) | UniLab continue 2 (yaw) |
|---|---|---|---|---|
| mean vx (cmd 0.6) | 0.573 m/s (96%) | 0.175 m/s (29%) | 0.505 m/s (84%) | **0.490 m/s (82%)** |
| RMSE (vx, vy) | 0.182 m/s | 0.466 m/s | 0.189 m/s | 0.192 m/s |
| tracking_lin exp | 0.883 | 0.439 | 0.875 | 0.870 |
| tracking_ang exp | 0.885 | 0.876 | 0.717 | **0.903** |
| mean \|wz\| | 0.154 rad/s | 0.153 | 0.286 | **0.129** |
| fall rate / 10 s | 0 | 0 | 0 | 0 |
| train ep_len | 976 | 981 | 1000 | 1000 |
| train return | 229.6 | 251.2 | 264.5 | 282.2 |

Continue 1: load first `model_10000.pt`, `tracking_lin_vel` 2→4, gait phase
gated at `vx≥0.15`. Continue 2: load that checkpoint, `tracking_ang_vel` 1.5→3.

## Gait video (forced vx=0.6, 8 s, yaw fixed)

| | Motrix | UniLab first | UniLab continue 2 |
|---|---|---|---|
| Clip | `tinkerbot_rl/roscon/renders/tinker_walk.mp4` | first-run (not archived as accepted) | `gaits/walk-flat/play.mp4` |
| Net displacement | ~4.8 m at vx=0.8 cmd | 1.51 m | **3.90 m** |
| Path / mean speed | ~0.60 / 0.8 m/s (75%) | ~0.19 m/s (31%) | **4.07 m / 0.51 m/s (82%)** |

## Archived policy (this repo)

The pre-rename Tinker checkpoints were removed after the Kudada-only retrain.
The accepted Kudada walk-flat policy, frozen configuration, summary, and video
are all under `gaits/walk-flat/`. Historical benchmark numbers above are
kept for comparison only.

Motrix counterpart: `tinkerbot_rl` branch `roscon`,
`roscon/train/26-09-16_11-00-26-017005/`.
