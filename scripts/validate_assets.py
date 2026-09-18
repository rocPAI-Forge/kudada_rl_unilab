#!/usr/bin/env python3
"""Compile the shipped MJCF and print its model contract."""
from pathlib import Path
import mujoco

root = Path(__file__).resolve().parents[1] / "assets/robots/kudada/mjcf"
model = mujoco.MjModel.from_xml_path(str(root / "scene_flat.xml"))
print(f"nq={model.nq} nv={model.nv} nu={model.nu} bodies={model.nbody} geoms={model.ngeom} sensors={model.nsensor}")
print(f"total_mass={model.body_mass.sum():.6f} kg")
