# Shooter: High-Performance RL Environment

A high-throughput reinforcement-learning environment built in C and integrated into [PufferLib's](https://github.com/PufferAI/PufferLib) Ocean suite.

The environment trains an agent to control its aim and fire at moving targets. Its simulation loop runs natively in C, while a Gymnasium-compatible Python wrapper exposes batched observations, actions, rewards, and episode statistics for PPO training.

> **Fork attribution:** This repository is a fork of PufferAI/PufferLib. Most of the repository is upstream PufferLib code. My project contribution is the Shooter environment and its Ocean/PPO integration, centered in [`pufferlib/ocean/shooter/`](pufferlib/ocean/shooter/).

## Results

Development-machine measurements:

| Metric | Result |
| --- | ---: |
| Batched environment throughput | **>200,000 steps/second** |
| Speedup over the vectorized Python reference used during development | **>20x** |
| PPO learning speed after hyperparameter tuning | **>3x faster than the initial baseline** |

Throughput is hardware- and configuration-dependent. The included benchmark defaults to 512 parallel environments and runs for 10 seconds.

## Environment

Each episode lasts 1,000 steps. The agent receives a reward of +1 when a hitscan shot intersects the moving target; the target then respawns at a randomized position.

**Observation space:** seven normalized floats

- Agent x/y position
- Aim heading and angular velocity
- Target x/y position and radius

**Action space:** four discrete actions

- No-op
- Fire
- Increase angular velocity
- Decrease angular velocity

## Implementation

- **Native simulation core:** target movement, aim dynamics, hit detection, resets, rewards, and episode logging are implemented in C.
- **Batched execution:** PufferLib's native binding vectorizes many independent environments and writes directly into shared observation, action, reward, and terminal buffers.
- **Python interface:** a Gymnasium-compatible wrapper handles environment construction, stepping, logging, rendering, and benchmarking.
- **Training integration:** the environment is registered in Ocean and includes a PPO configuration plus a Protein hyperparameter sweep over learning rate, discount factor, entropy coefficient, and minibatch size.
- **Rendering:** Raylib visualizes the agent, target, aim direction, and shots.

## Project Map

- [`shooter.h`](pufferlib/ocean/shooter/shooter.h) — C environment state and simulation loop
- [`binding.c`](pufferlib/ocean/shooter/binding.c) — native PufferLib binding
- [`shooter.py`](pufferlib/ocean/shooter/shooter.py) — Python wrapper and throughput benchmark
- [`shooter.ini`](pufferlib/config/ocean/shooter.ini) — PPO and sweep configuration
- [`environment.py`](pufferlib/ocean/environment.py) — Ocean registry integration

## Quick Start

PufferLib 3.0's build currently targets Linux and macOS. A C compiler and Python 3.9+ are required.

```bash
git clone https://github.com/micahtracht/PufferLib.git
cd PufferLib

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Benchmark Environment Throughput

```bash
python -m pufferlib.ocean.shooter.shooter
```

The command prints aggregate Shooter steps per second across the default 512-environment batch.

### Train with PPO

```bash
python -m pufferlib.pufferl train puffer_shooter
```

### Run the Hyperparameter Sweep

Configure Weights & Biases first, then run:

```bash
python -m pufferlib.pufferl sweep puffer_shooter --wandb
```

The training and sweep commands automatically load [`pufferlib/config/ocean/shooter.ini`](pufferlib/config/ocean/shooter.ini).

## Upstream Project

PufferLib is an open-source high-performance reinforcement-learning toolkit created by Joseph Suarez and contributors. See the [upstream repository](https://github.com/PufferAI/PufferLib), [documentation](https://puffer.ai), and this fork's [MIT license](LICENSE).
