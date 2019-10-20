# PyTorch_ProxylessNAS

Unofficial PyTorch implementation of: [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332).

## Results

### CIFAR-10

|                      | Paper | Here |
| -------------------- | ----- | ---- |
| Model Parameters (M) | 5.7   | 5.3  |
| Top-1 Test Error (%) | 2.08  | 2.72 |

<!-- ### CIFAR-100

|                      | Paper | Here |
| -------------------- | ----- | ---- |
| Model Parameters (M) |  N/A  | 5.3  |
| Top-1 Test Error (%) |  N/A  | 2.72 | -->

## Requirements

- python 3
- pyyaml
- torch
- torchvision
- numpy
- graphviz

Optional packages:

- tensorboardX
- adabound

## Usage

### Arch Search

    python search.py -n <run_name> -c <config_path> -d <gpu_ids>

By default, checkpoints and genotypes are saved to 'chkpt/'

### Augment

    python augment.py -n <run_name> -c <config_path> -d <gpu_ids> -g <genotype_path>

## Config

Run configs are specified in config/default.yaml

- search: arch search run config
- augment: augment run config
- model: model specs.
- genotypes: candidate operations in arch search
- log: logging config

## Reference

- [official repo](https://github.com/mit-han-lab/ProxylessNAS)

- [PathLevel-EAS](https://github.com/han-cai/PathLevel-EAS)

- [darts](https://github.com/quark0/darts)

- [pt.darts](https://github.com/khanrc/pt.darts)

- [PyramidNet](https://github.com/dyhan0920/PyramidNet-PyTorch)