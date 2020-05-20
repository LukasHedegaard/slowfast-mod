# HAGS 🧙‍♀️🧙🏻‍♀️🧙🏾‍♀️ Human Action GCN Stitching
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Description
Research repository for stitching together visual glimpses and superpixels using Graph Convolutional Network (GCNs) for Human Action Recognition and Detection tasks.


## Installation
- Install GCC
  ```bash
  conda install -c conda-forge gcc
  ```

- Install UnRAR (e.g. from the [website](https://www.rarlab.com/rar_add.htm))

- Clone this project and install it
  ```bash
  git clone https://github.com/LukasHedegaard/hags
  cd hags
  pip install -e .[dev]  
  pip install -r requirements.txt
  ```

### Datasets
You may follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets.


## Quick Start
See [GETTING_STARTED.md](GETTING_STARTED.md).


## Main Contribution
Coming up!

Novel contributions were added by
- [Lukas Hedegaard](https://www.linkedin.com/in/lukashedegaard/)


## Baselines  
Thanks to [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), who wrote and maintain [PySlowFast](https://github.com/facebookresearch/SlowFast), implementations of the following backbone network architectures are included:
- SlowFast 
- Slow
- C2D
- I3D
- Non-local Network


## Citation   
```
@article{tbd2020,
  title={TBD},
  author={Lukas Hedegaard},
  journal={TBD},
  year={2020}
}
```   