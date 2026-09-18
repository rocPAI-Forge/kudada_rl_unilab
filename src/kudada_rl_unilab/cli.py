"""Thin train/eval wrappers around the published UniLab package."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from unilab.cli import package_root as unilab_package_root

CONF_ROOT = Path(__file__).resolve().parent / "conf"
REGISTRY_PACKAGE = "kudada_rl_unilab.tasks"
EXTRA_PACKAGES_ENV = "UNILAB_EXTRA_REGISTRY_PACKAGES"
SCRIPT_NAMES = {"sac": "train_sac.py"}
SUPPORTED_SIMS = ("mujoco",)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTINUE_RUN = REPO_ROOT / "roscon/gaits/walk-flat"


def ensure_registry_env() -> None:
    packages = [p.strip() for p in os.environ.get(EXTRA_PACKAGES_ENV, "").split(",") if p.strip()]
    if REGISTRY_PACKAGE not in packages:
        packages.append(REGISTRY_PACKAGE)
    os.environ[EXTRA_PACKAGES_ENV] = ",".join(packages)


def build_command(mode: str, argv: Sequence[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog=f"kudada-{mode}")
    parser.add_argument("--algo", default="sac", choices=tuple(SCRIPT_NAMES))
    parser.add_argument("--task", default="walk-flat")
    parser.add_argument("--sim", default="mujoco", choices=SUPPORTED_SIMS)
    if mode in {"eval", "continue"}:
        parser.add_argument("--load-run", default=None)
    args, overrides = parser.parse_known_args(argv)
    conf_dir = CONF_ROOT / args.algo
    owner = conf_dir / "task" / args.task / f"{args.sim}.yaml"
    if not owner.is_file():
        raise SystemExit(f"Owner config not found: {owner}")
    generated = [f"task={args.task}/{args.sim}"]
    if mode == "continue":
        script = Path(__file__).resolve().parent / "continue_train.py"
        load_run = args.load_run or str(DEFAULT_CONTINUE_RUN)
        generated.extend(
            [
                f"algo.load_run={load_run}",
                "training.no_play=true",
            ]
        )
        return [sys.executable, str(script), "--config-dir", str(conf_dir), *generated, *overrides]
    script = unilab_package_root() / "scripts" / SCRIPT_NAMES[args.algo]
    if not script.is_file():
        raise SystemExit(f"UniLab entrypoint script not found: {script}")
    if mode == "eval":
        generated.append("training.play_only=true")
        if args.load_run is not None:
            generated.append(f"algo.load_run={args.load_run}")
    return [sys.executable, str(script), "--config-dir", str(conf_dir), *generated, *overrides]


def _run(mode: str, argv: Sequence[str] | None = None) -> int:
    ensure_registry_env()
    return subprocess.run(build_command(mode, argv), check=False).returncode


def train_main(argv: Sequence[str] | None = None) -> int:
    return _run("train", argv)


def eval_main(argv: Sequence[str] | None = None) -> int:
    return _run("eval", argv)


def continue_main(argv: Sequence[str] | None = None) -> int:
    return _run("continue", argv)
