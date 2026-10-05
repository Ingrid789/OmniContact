"""Install the OmniContact Isaac Lab extension."""

from setuptools import find_packages, setup


setup(
    name="omnicontact",
    version="0.1.0",
    description="OmniContact rubber-hand reinforcement-learning tasks for Isaac Lab",
    packages=find_packages(),
    install_requires=[
        "rsl-rl-lib==2.3.3+omnicontact.1",
        "tensorboard",
        "psutil",
        "onnxscript",
        "wandb>=0.19",
        "huggingface-hub>=0.34",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    package_data={"omnicontact.assets": ["unitree_description/urdf/g1/*.urdf"]},
    zip_safe=False,
)
