# Asset Code and Resources

This directory contains the asset code and serves as the default resource root:

- `__init__.py`: asset-root configuration and path helpers.
- `object_spawner.py`: object USD configurations selected from motion filenames.
- `robots/omnicontact_rubberhand.py`: G1 robot, ghost robot, and action-scale configurations.

The source package's `omnicontact/assets` is a relative symbolic link to this
directory. Import the code through `omnicontact.assets`; for example,
`from omnicontact.assets import asset_path`. The normal package installation
also includes the asset modules and the bundled ghost-robot URDF.

The training code expects an asset root containing at least:

```text
<asset-root>/
├── g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd
├── objects/<object-name>/<object-name>.usd
└── unitree_description/
    ├── meshes/g1/*.STL
    └── urdf/g1/simple.urdf
```

Large USD/STL assets and motion datasets are intentionally not bundled with
this repository. Place resource files here or set `OMNICONTACT_ASSET_DIR` to an
external asset root before training. This environment variable changes the
resource paths, while Python code continues to load from this directory.
The small ghost-robot URDF used by default is included at
`assets/unitree_description/urdf/g1/simple.urdf`.
