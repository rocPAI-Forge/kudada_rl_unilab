from roscon_rl_unilab.cli import build_command, ensure_registry_env


def test_continue_command_loads_checkpoint():
    command = build_command(
        "continue",
        ["--algo", "sac", "--task", "walk-flat", "--sim", "mujoco", "algo.max_iterations=2"],
    )
    assert command[1].endswith("continue_train.py")
    assert "task=walk-flat/mujoco" in command
    assert any(item.startswith("algo.load_run=") for item in command)
    assert "training.no_play=true" in command
    assert "algo.max_iterations=2" in command


def test_train_command_routes_to_sac_owner():
    command = build_command("train", ["--algo", "sac", "--task", "walk-flat", "--sim", "mujoco", "algo.num_envs=4"])
    assert command[-2:] == ["task=walk-flat/mujoco", "algo.num_envs=4"]
    assert command[1].endswith("train_sac.py")


def test_registry_env_is_idempotent(monkeypatch):
    monkeypatch.setenv("UNILAB_EXTRA_REGISTRY_PACKAGES", "other.tasks")
    ensure_registry_env(); ensure_registry_env()
    assert __import__("os").environ["UNILAB_EXTRA_REGISTRY_PACKAGES"] == "other.tasks,roscon_rl_unilab.tasks"
