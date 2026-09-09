# QH-DAC for Hybrid-Action UAV Control

This repository provides training and evaluation code for Q-guided Hybrid
Diffusion Actor-Critic (QH-DAC). The policy jointly controls a continuous
two-dimensional UAV displacement and a discrete device-scheduling decision.

QH-DAC uses a conditional diffusion actor trained with denoising behavior
cloning and differentiable Q-guidance. Its Bellman target samples the next
action from a target diffusion actor and evaluates it with clipped double-Q
critics. The proposed objective contains neither an SAC entropy term nor a CQL
regularizer. CQL is included only as an independent baseline.

This public repository demonstrates dataset collection, method training, and
held-out evaluation. It intentionally contains no notebooks or plotting code.

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

Each method folder contains `agent.py`, `train.py`, and `test.py`. Shared
training and evaluation behavior is kept in `common/` so all methods use the
same dataset subsets, action constraints, terminal masks, and test layouts.

## Installation

Python 3.8 or newer is supported. Create an environment and install the two
runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

For experiments on an existing CUDA server, use the installed PyTorch build
that matches the server CUDA driver instead of replacing it unnecessarily.

## 1. Train the online policy and collect data

From the repository root:

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

Interrupted collection resumes from `results/online_training_state.pt`.

Evaluate the selected online checkpoint on held-out layouts:

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

Training writes `results/qh_dac/checkpoint.pt` and
`results/qh_dac/history.json`. Rerunning the command resumes the checkpoint;
use `--no-resume` to start a new run after moving or deleting the old output.

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

QH-MLP is a matched actor ablation: it replaces the diffusion actor with an
MLP while retaining the same behavior-cloning/Q-guidance objective and critic.
It is not presented as a new standalone offline-RL algorithm.

## Common command-line options

All offline training scripts accept:

```text
--dataset PATH
--output-dir PATH
--epochs INTEGER
--batch-size INTEGER
--subset-size INTEGER
--subset-seed INTEGER
--quality {all,good,bad}
--reward-scale FLOAT
--eval-every INTEGER
--eval-layouts INTEGER
--seed INTEGER
--device {cpu,cuda}
--no-resume
```

For a short execution check after generating the dataset:

```bash
python QH-DAC/train.py \
  --device cpu --epochs 1 --subset-size 1000 --eval-layouts 2 \
  --output-dir results/smoke_qh_dac --no-resume
```

## Evaluation output

Testing prints a JSON object containing raw episodic return, time-weighted AoI,
total energy, throughput, transmission time, action latency, episode length,
idle fraction, served-device count, and boundary-rejection rate. Use `--output`
to save the JSON response:

```bash
python QH-DAC/test.py --output results/qh_dac/test_metrics.json
```

## Reproducibility

- Continuous displacement is projected onto the disk
  `||displacement||_2 <= 25 m`.
- Discrete scheduling uses one common `K+1` representation, with action zero
  reserved for idling.
- The QH-DAC actor uses straight-through softmax for scheduling during
  Q-guided training.
- The target uses the target actor and the minimum of two target critics; it
  does not use an unconstrained maximum-Q backup.
- Raw rewards are stored in the dataset. A positive scale of `0.01` is applied
  only to optimizer targets for numerical conditioning.
- Dataset subset selection and evaluation layouts are seeded.

## License

This project is released under the MIT License.

