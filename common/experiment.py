from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .env import UAVEnvConfig, UAVOfflineRLEnv, make_device_layout
from .evaluation import evaluate_policy, evaluate_policy_batch


def default_env_config() -> UAVEnvConfig:
    return UAVEnvConfig(
        area_size=1000.0,
        h=100.0,
        B=1e6,
        fc=2e9,
        Dk_bits=2e6,
        lam=0.5,
        T_th=400.0,
        H_max=None,
        d_max=25.0,
        fading="rayleigh",
    )


def build_env(
    num_devices: int,
    layout_seed: int,
    cfg: Optional[UAVEnvConfig] = None,
) -> UAVOfflineRLEnv:
    cfg = cfg or default_env_config()
    layout = make_device_layout(num_devices, cfg.area_size, layout_seed)
    return UAVOfflineRLEnv(layout, cfg)


def evaluate_across_layouts(
    agent,
    num_devices: int,
    layout_seeds: Sequence[int],
    state_normalizer=None,
    cfg: Optional[UAVEnvConfig] = None,
    fading_seed_offset: int = 100_000,
    deterministic: bool = True,
) -> Dict[str, float]:
    cfg = cfg or default_env_config()
    seeds = tuple(int(seed) for seed in layout_seeds)
    if hasattr(agent, "act_batch"):
        envs = [build_env(num_devices, seed, cfg) for seed in seeds]
        fading_seeds = [fading_seed_offset + seed for seed in seeds]
        return evaluate_policy_batch(
            envs,
            agent,
            state_normalizer=state_normalizer,
            seeds=fading_seeds,
            deterministic=deterministic,
        )
    metrics: Dict[str, List[float]] = defaultdict(list)
    for seed in seeds:
        result = evaluate_policy(
            build_env(num_devices, seed, cfg),
            agent,
            n_episodes=1,
            state_normalizer=state_normalizer,
            seed=fading_seed_offset + seed,
            deterministic=deterministic,
        )
        for name, value in result.items():
            metrics[name].append(value)
    return {name: float(np.mean(values)) for name, values in metrics.items()}

