"""Repository-local Tinker Lingzu assets."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "assets" / "robots" / "tinker_lingzu"
MJCF_SCENE = ASSET_ROOT / "mjcf" / "scene_flat.xml"
URDF_MODEL = ASSET_ROOT / "urdf" / "tinker_lingzu.urdf"


def ensure_assets() -> Path:
    missing = [path for path in (MJCF_SCENE, URDF_MODEL) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Tinker Lingzu assets: {missing}")
    return ASSET_ROOT
