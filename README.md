# QH-DAC for Hybrid-Action UAV Control

This repository provides training and evaluation code for Q-guided Hybrid
Diffusion Actor-Critic (QH-DAC).

QH-DAC uses a conditional diffusion actor trained with denoising behavior
cloning and differentiable Q-guidance. Its Bellman target samples the next
action from a target diffusion actor and evaluates it with clipped double-Q
critics.

## Repository structure

```text
common/                    Shared environment, data, networks, and runners
data_collection/           Online hybrid SAC and dataset collection
QH-DAC/                    Proposed diffusion actor-critic
baselines/
  QH-MLP/                  Matched non-diffusion BC + Q control
  CQL/                     Conservative Q-Learning
  IQL/                     Implicit Q-Learning
  BCQ/                     Batch-Constrained Q-Learning
  ReBRAC/                  Revisited Behavior-Regularized Actor-Critic
  DTQL/                    Diffusion Trusted Q-Learning
data/                      Generated offline dataset
results/                   Generated checkpoints and JSON histories
```


## Installation

Python 3.8 or newer is supported. Install requirements:

```bash
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## 1. Train the online policy and collect data


```bash
python data_collection/collect_dataset.py --device cuda
```

The default run trains the online hybrid SAC behavior policy for 700 episodes
and saves 100,000 raw transitions to:

```text
data/uav_offline_dataset.npz
```

The state contains UAV coordinates, device-relative coordinates, AoI,
pre-decision SNR, cumulative energy, and remaining mission time. The reward
uses instantaneous mean AoI and current-step energy. Episodes terminate using
accumulated physical mission time.

Evaluate the selected online checkpoint:

```bash
python data_collection/test_online.py --device cuda --layouts 100
```

## 2. Train and test QH-DAC

```bash
python QH-DAC/train.py --device cuda
python QH-DAC/test.py --device cuda --layouts 100
```

Important QH-DAC options include:

```text
--diffusion-steps 50
--eta-bc 0.5
--subset-size 30000
--quality all
--epochs 150
```

## 3. Train and test baselines

The commands follow the same interface:

```bash
python baselines/QH-MLP/train.py --device cuda
python baselines/QH-MLP/test.py --device cuda --layouts 100

python baselines/CQL/train.py --device cuda
python baselines/CQL/test.py --device cuda --layouts 100

python baselines/IQL/train.py --device cuda
python baselines/IQL/test.py --device cuda --layouts 100

python baselines/BCQ/train.py --device cuda
python baselines/BCQ/test.py --device cuda --layouts 100

python baselines/ReBRAC/train.py --device cuda
python baselines/ReBRAC/test.py --device cuda --layouts 100

python baselines/DTQL/train.py --device cuda
python baselines/DTQL/test.py --device cuda --layouts 100
```

## Citation

If you find our codes useful, please cite our work: @misc{QH_DAC,
                                                    title={Diffusion Offline Reinforcement Learning for Fair and Energy-Efficient UAV-Assisted Wireless Networks}, 
                                                    author={Eslam Eldeeb and Hirley Alves},
                                                    year={2026},
                                                    eprint={2606.16331},
                                                    archivePrefix={arXiv},
                                                    primaryClass={cs.LG},
                                                    url={https://arxiv.org/abs/2606.16331}, 
                                              }


