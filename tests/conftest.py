import os
from pathlib import Path

os.environ["UNILAB_EXTRA_REGISTRY_PACKAGES"] = "kudada_rl_unilab.tasks"

from kudada_rl_unilab.conf_searchpath import register_conf_search_path  # noqa: E402

register_conf_search_path()
REPO_ROOT = Path(__file__).resolve().parents[1]
