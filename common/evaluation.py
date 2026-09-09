from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .data import Normalizer


def _agent_act(agent, policy_input: np.ndarray, deterministic: bool = True):
    if hasattr(agent, "act"):
        return agent.act(policy_input, deterministic=deterministic)
    if hasattr(agent, "select_action"):
        return agent.select_action(policy_input, deterministic=deterministic)
    raise AttributeError(
        f"{type(agent).__name__} provides neither act() nor select_action()"
    )


@torch.no_grad()
def evaluate_policy(
    env,
    agent,
    n_episodes: int = 5,
    state_normalizer: Optional[Normalizer] = None,
    seed: Optional[int] = 0,
    deterministic: bool = True,
) -> Dict[str, float]:
    metrics = {
        "return": [],
        "aoi": [],
        "total_energy_j": [],
        "mean_step_energy_j": [],
        "ep_len": [],
        "time_elapsed": [],
        "throughput_bits": [],
        "mean_transmission_time_s": [],
        "inference_time_ms": [],
        "out_of_bounds_fraction": [],
        "idle_fraction": [],
        "served_devices": [],
    }
    for episode in range(n_episodes):
        episode_seed = None if seed is None else seed + episode
        if episode_seed is not None:
            torch.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
        observation = env.reset(seed=episode_seed)
        done = False
        episode_return = 0.0
        time_weighted_aoi = 0.0
        energy = 0.0
        steps = 0
        throughput = 0.0
        transmission_times = []
        inference_times = []
        rejected_moves = 0
        schedule_counts = np.zeros(env.K + 1, dtype=np.int64)
        last_info = {}
        while not done:
            policy_input = (
                observation
                if state_normalizer is None
                else state_normalizer.normalize(observation)
            )
            started = time.perf_counter()
            action_cont, action_disc = _agent_act(
                agent, policy_input, deterministic=deterministic
            )
            inference_times.append((time.perf_counter() - started) * 1000.0)
            observation, reward, done, info = env.step(
                (float(action_cont[0]), float(action_cont[1]), int(action_disc))
            )
            episode_return += reward
            time_weighted_aoi += float(np.mean(env.A)) * float(info["T_total"])
            energy += float(info["E_step"])
            steps += 1
            rejected_moves += int(info["out_of_bounds"])
            schedule_counts[int(action_disc)] += 1
            if info["served_device"] is not None:
                throughput += env.cfg.Dk_bits
                transmission_times.append(float(info["T_com"]))
            last_info = info
        elapsed = float(last_info["time_elapsed"])
        metrics["return"].append(episode_return)
        metrics["aoi"].append(time_weighted_aoi / max(elapsed, 1e-12))
        metrics["total_energy_j"].append(energy)
        metrics["mean_step_energy_j"].append(energy / max(steps, 1))
        metrics["ep_len"].append(steps)
        metrics["time_elapsed"].append(elapsed)
        metrics["throughput_bits"].append(throughput)
        metrics["mean_transmission_time_s"].append(
            float(np.mean(transmission_times)) if transmission_times else 0.0
        )
        metrics["inference_time_ms"].append(float(np.mean(inference_times)))
        metrics["out_of_bounds_fraction"].append(rejected_moves / max(steps, 1))
        metrics["idle_fraction"].append(schedule_counts[0] / max(steps, 1))
        metrics["served_devices"].append(
            int(np.count_nonzero(schedule_counts[1:]))
        )
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def _synchronize_agent(agent) -> None:
    device = getattr(agent, "device", None)
    if torch.cuda.is_available() and device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@torch.no_grad()
def _single_action_latency_ms(
    agent,
    policy_input: np.ndarray,
    repeats: int = 10,
    deterministic: bool = True,
) -> float:
    # Warm up lazy CUDA kernels before timing deployable one-state latency.
    for _ in range(2):
        _agent_act(agent, policy_input, deterministic=deterministic)
    _synchronize_agent(agent)
    started = time.perf_counter()
    for _ in range(repeats):
        _agent_act(agent, policy_input, deterministic=deterministic)
    _synchronize_agent(agent)
    return (time.perf_counter() - started) * 1000.0 / max(repeats, 1)


@torch.no_grad()
def evaluate_policy_batch(
    envs: Sequence[object],
    agent,
    state_normalizer: Optional[Normalizer] = None,
    seeds: Optional[Sequence[int]] = None,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate independent layouts synchronously with batched policy calls.

    Environment dynamics remain independent. Only neural-network inference is
    batched, which preserves the 100-layout protocol while avoiding 100 serial
    diffusion chains at every environment step.
    """
    if not envs:
        raise ValueError("envs must not be empty")
    if seeds is None:
        seeds = list(range(len(envs)))
    if len(seeds) != len(envs):
        raise ValueError("seeds and envs must have the same length")

    torch.manual_seed(int(seeds[0]))
    torch.cuda.manual_seed_all(int(seeds[0]))
    observations = [env.reset(seed=int(seed)) for env, seed in zip(envs, seeds)]
    initial_policy_input = (
        observations[0]
        if state_normalizer is None
        else state_normalizer.normalize(observations[0])
    )
    count = len(envs)
    active = np.ones(count, dtype=bool)
    returns = np.zeros(count, dtype=np.float64)
    time_weighted_aoi = np.zeros(count, dtype=np.float64)
    energy = np.zeros(count, dtype=np.float64)
    steps = np.zeros(count, dtype=np.int64)
    throughput = np.zeros(count, dtype=np.float64)
    rejected = np.zeros(count, dtype=np.int64)
    schedule_counts = np.zeros((count, envs[0].K + 1), dtype=np.int64)
    transmission_times = [[] for _ in envs]
    elapsed = np.zeros(count, dtype=np.float64)

    while np.any(active):
        indices = np.flatnonzero(active)
        policy_input = np.stack([observations[index] for index in indices])
        if state_normalizer is not None:
            policy_input = state_normalizer.normalize(policy_input)
        action_cont, action_disc = agent.act_batch(
            policy_input, deterministic=deterministic
        )
        for row, index in enumerate(indices):
            observation, reward, done, info = envs[index].step(
                (
                    float(action_cont[row, 0]),
                    float(action_cont[row, 1]),
                    int(action_disc[row]),
                )
            )
            observations[index] = observation
            returns[index] += float(reward)
            time_weighted_aoi[index] += (
                float(np.mean(envs[index].A)) * float(info["T_total"])
            )
            energy[index] += float(info["E_step"])
            steps[index] += 1
            rejected[index] += int(info["out_of_bounds"])
            schedule_counts[index, int(action_disc[row])] += 1
            if info["served_device"] is not None:
                throughput[index] += envs[index].cfg.Dk_bits
                transmission_times[index].append(float(info["T_com"]))
            if done:
                active[index] = False
                elapsed[index] = float(info["time_elapsed"])

    latency_ms = _single_action_latency_ms(
        agent, initial_policy_input, deterministic=deterministic
    )
    mean_transmission = np.asarray(
        [float(np.mean(values)) if values else 0.0 for values in transmission_times]
    )
    return {
        "return": float(np.mean(returns)),
        "aoi": float(np.mean(time_weighted_aoi / np.maximum(elapsed, 1e-12))),
        "total_energy_j": float(np.mean(energy)),
        "mean_step_energy_j": float(np.mean(energy / np.maximum(steps, 1))),
        "ep_len": float(np.mean(steps)),
        "time_elapsed": float(np.mean(elapsed)),
        "throughput_bits": float(np.mean(throughput)),
        "mean_transmission_time_s": float(np.mean(mean_transmission)),
        "inference_time_ms": latency_ms,
        "out_of_bounds_fraction": float(
            np.mean(rejected / np.maximum(steps, 1))
        ),
        "idle_fraction": float(
            np.mean(schedule_counts[:, 0] / np.maximum(steps, 1))
        ),
        "served_devices": float(
            np.mean(np.count_nonzero(schedule_counts[:, 1:], axis=1))
        ),
    }



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

