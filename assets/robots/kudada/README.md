# Kudada assets

- `urdf/`: source URDF and meshes from the original `tinkerbot_rl` repository.
- `mjcf/`: MuJoCo conversion used by UniLab.

The MJCF was generated from the URDF and then corrected for root inertia, named
collision geoms, foot/IMU sites, position actuators, force limits, and the home
keyframe. `assets.sha256` covers both trees.
