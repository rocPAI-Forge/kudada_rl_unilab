"""Register external task owner configs for programmatic Hydra composition."""
from pathlib import Path

CONF_ROOT = Path(__file__).resolve().parent / "conf"
_registered = False


def register_conf_search_path() -> None:
    global _registered
    if _registered:
        return
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    algo_dirs = tuple(f"file://{p}" for p in sorted(CONF_ROOT.iterdir()) if p.is_dir())

    class KudadaConfSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path):
            for path in algo_dirs:
                search_path.append("kudada_rl_unilab", path)

    Plugins.instance().register(KudadaConfSearchPathPlugin)
    _registered = True
