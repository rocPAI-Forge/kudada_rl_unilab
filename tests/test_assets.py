import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "robots" / "tinker_lingzu"
JOINTS = (
    "left_leg_hip_yaw_joint", "left_leg_hip_roll_joint", "left_leg_hip_pitch_joint",
    "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint",
    "right_leg_hip_yaw_joint", "right_leg_hip_roll_joint", "right_leg_hip_pitch_joint",
    "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint",
)


def test_dual_asset_trees_present():
    assert (ASSETS / "urdf" / "tinker_lingzu.urdf").is_file()
    assert (ASSETS / "mjcf" / "tinker_lingzu.xml").is_file()
    assert (ASSETS / "mjcf" / "scene_flat.xml").is_file()
    assert len(list((ASSETS / "urdf" / "meshes").glob("*.STL"))) == 11
    assert len(list((ASSETS / "mjcf" / "assets").glob("*.STL"))) == 11
    assert ET.parse(ASSETS / "urdf" / "tinker_lingzu.urdf").getroot().tag == "robot"


def test_mjcf_compiles_and_preserves_contract():
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "mjcf" / "scene_flat.xml"))
    assert (model.nq, model.nv, model.nu) == (17, 16, 10)
    assert model.nkey == 1
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu))
    assert names == JOINTS
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert model.body_mass[base_id] == pytest.approx(6.88358184)
    assert model.body_mass.sum() == pytest.approx(10.0)
    for name in ("left_foot", "right_foot", "imu_in_base"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    for name in ("base_link_collision", "left_leg_knee_pitch_collision", "right_leg_knee_pitch_collision", "left_foot_collision", "right_foot_collision"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


def test_asset_manifest_matches():
    manifest = ASSETS / "assets.sha256"
    assert manifest.is_file()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        path = ASSETS / relative.lstrip("./")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
