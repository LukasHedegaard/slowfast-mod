#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="hags",
    version="0.1",
    author="Lukas Hedegaard",
    url="unknown",
    description="Human Action GCN Stitching",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "torchvision>=0.4.2",
        "sklearn",
        "unrar",
    ],
    extras_require={"dev": ["isort", "black", "flake8", "flake8-black",]},
    packages=find_packages(exclude=("configs", "tests")),
)
