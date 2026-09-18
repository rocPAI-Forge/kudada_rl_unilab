"""Resume FastSAC from a checkpoint; UniLab's train_sac.py always starts cold."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from unilab.cli import package_root
from unilab.scripts import train_offpolicy as offpolicy
from unilab.utils.checkpoint import resolve_offpolicy_checkpoint_path

from kudada_rl_unilab.conf_searchpath import register_conf_search_path

_ORIG_BUILD_RUNNER = offpolicy.build_runner


def _latest_model(run_dir: Path) -> Path | None:
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def build_runner(algo_name, cfg: DictConfig, log_dir: str | None = None):
    runner = _ORIG_BUILD_RUNNER(algo_name, cfg, log_dir=log_dir)
    load_run = str(getattr(cfg.algo, "load_run", "-1"))
    ckpt_path = None
    candidate = Path(load_run)
    if candidate.is_file() and candidate.suffix == ".pt":
        ckpt_path = candidate
    elif candidate.is_dir():
        ckpt_path = _latest_model(candidate)
    else:
        resolved, _ = resolve_offpolicy_checkpoint_path(
            Path.cwd(),
            cfg.algo.algo_log_name,
            cfg.training.task_name,
            load_run,
        )
        if resolved:
            ckpt_path = Path(resolved)
    if ckpt_path is None or not ckpt_path.is_file():
        raise SystemExit(f"continue-train needs a checkpoint; algo.load_run={load_run!r}")
    state = torch.load(ckpt_path, map_location=runner.learner.device, weights_only=False)
    runner.learner.load_state_dict(state)
    print(f"[continue-train] loaded weights from {ckpt_path}")
    return runner


offpolicy.build_runner = build_runner
register_conf_search_path()


@hydra.main(version_base="1.3", config_path=str(package_root() / "conf" / "sac"), config_name="config")
def main(cfg: DictConfig) -> None:
    offpolicy.main(cfg)


if __name__ == "__main__":
    main()
