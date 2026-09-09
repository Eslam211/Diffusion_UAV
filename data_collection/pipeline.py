from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from common.data import ReplayBuffer
from common.env import UAVEnvConfig
from common.experiment import build_env, default_env_config, evaluate_across_layouts
from common.evaluation import set_seed
from .online_agent import HybridSACAgent, SACConfig, random_hybrid_action


DATASET_PROTOCOL = "qh_dac_uav_v1"
TRAIN_LAYOUT_SEEDS = tuple(range(10_000, 10_020))
VALIDATION_LAYOUT_SEEDS = tuple(range(20_000, 20_020))
TEST_LAYOUT_SEEDS = tuple(range(100))


def save_history(history: Dict[str, Iterable[float]], path: str) -> None:
    serializable = {
        name: np.asarray(values).astype(float).tolist()
        for name, values in history.items()
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


class _PhysicalObservationNormalizer:
    def __init__(self, reference_env) -> None:
        self.reference_env = reference_env

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return self.reference_env.normalize_observation(values)


def _state_dict_cpu(module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _append_replay_buffer(
    destination: ReplayBuffer,
    source: ReplayBuffer,
    quality: int,
) -> None:
    """Append a circular buffer chronologically and close the stratum."""
    for index in source.chronological_indices():
        destination.add(
            source.s[index],
            source.a_cont[index],
            int(source.a_disc[index]),
            float(source.r[index, 0]),
            source.s2[index],
            bool(source.done[index, 0]),
            quality=quality,
        )
    if len(source):
        last = (destination._ptr - 1) % destination.capacity
        destination.done[last, 0] = 1.0


def train_online_and_collect(
    output_dataset: str,
    checkpoint_path: Optional[str] = None,
    history_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    seed: int = 0,
    K: int = 10,
    episodes: int = 700,
    capacity: int = 100_000,
    random_steps: int = 10_000,
    batch_size: int = 64,
    updates_per_step: int = 1,
    reward_scale: float = 0.01,
    online_dataset_fraction: float = 0.5,
    validation_every: int = 10,
    finalize_dataset: bool = True,
    training_layout_seeds: Sequence[int] = TRAIN_LAYOUT_SEEDS,
    validation_layout_seeds: Sequence[int] = VALIDATION_LAYOUT_SEEDS,
    validation_fading_seed_offset: int = 300_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    env_cfg: Optional[UAVEnvConfig] = None,
) -> Dict[str, List[float]]:
    set_seed(seed)
    cfg = env_cfg or default_env_config()
    training_layout_seeds = tuple(int(value) for value in training_layout_seeds)
    validation_layout_seeds = tuple(int(value) for value in validation_layout_seeds)
    if not training_layout_seeds or not validation_layout_seeds:
        raise ValueError("training and validation layout seed sets must be non-empty")
    if set(training_layout_seeds) & set(validation_layout_seeds):
        raise ValueError("training and validation layout seeds must be disjoint")
    if set(training_layout_seeds) & set(TEST_LAYOUT_SEEDS):
        raise ValueError("training and held-out test layout seeds must be disjoint")
    if set(validation_layout_seeds) & set(TEST_LAYOUT_SEEDS):
        raise ValueError("validation and held-out test layout seeds must be disjoint")
    if validation_every <= 0:
        raise ValueError("validation_every must be positive")

    reference_env = build_env(K, training_layout_seeds[0], cfg)
    online_normalizer = _PhysicalObservationNormalizer(reference_env)
    agent = HybridSACAgent(
        reference_env.observation_dim,
        K + 1,
        cfg.d_max,
        cfg=SACConfig(device=device),
    ).to(device)
    if not 0.0 < online_dataset_fraction < 1.0:
        raise ValueError("online_dataset_fraction must be strictly between 0 and 1")
    online_collection_limit = int(capacity * online_dataset_fraction)
    if random_steps <= 0 or online_collection_limit <= random_steps:
        raise ValueError(
            "The behavior portion must contain positive random and online strata"
        )
    online_capacity = online_collection_limit - random_steps
    nominal_good_capacity = capacity - online_collection_limit
    replay_buffer = ReplayBuffer(reference_env.observation_dim, capacity)
    random_buffer = ReplayBuffer(reference_env.observation_dim, random_steps)
    online_buffer = ReplayBuffer(reference_env.observation_dim, online_capacity)
    history: Dict[str, List[float]] = defaultdict(list)
    global_step = 0
    latest_logs: Dict[str, float] = {}
    best_episode = 0
    best_validation_return = -float("inf")
    best_validation_metrics: Dict[str, float] = {}
    best_greedy_validation_metrics: Dict[str, float] = {}
    best_agent_state: Optional[Dict[str, torch.Tensor]] = None
    start_episode = 1
    training_state_path = Path(resume_path) if resume_path is not None else None
    if training_state_path is not None and training_state_path.exists():
        try:
            training_state = torch.load(
                training_state_path, map_location=device, weights_only=False
            )
        except TypeError:
            training_state = torch.load(training_state_path, map_location=device)
        expected_run = {
            "protocol_version": DATASET_PROTOCOL,
            "K": K,
            "capacity": capacity,
            "random_steps": random_steps,
            "batch_size": batch_size,
            "updates_per_step": updates_per_step,
            "reward_scale": reward_scale,
            "online_dataset_fraction": online_dataset_fraction,
            "validation_every": validation_every,
            "validation_fading_seed_offset": validation_fading_seed_offset,
            "training_layout_seeds": list(training_layout_seeds),
            "validation_layout_seeds": list(validation_layout_seeds),
        }
        actual_run = training_state.get("run_config", {})
        if actual_run != expected_run:
            raise RuntimeError(
                "Online resume configuration does not match this run. "
                "Rename the old resume file before starting."
            )
        agent.load_state_dict(training_state["agent_state_dict"])
        agent.actor_opt.load_state_dict(training_state["actor_opt"])
        agent.critic_opt.load_state_dict(training_state["critic_opt"])
        agent.alpha_opt.load_state_dict(training_state["alpha_opt"])
        replay_buffer.load_state_dict(training_state["replay_buffer"])
        random_buffer.load_state_dict(training_state["random_buffer"])
        online_buffer.load_state_dict(training_state["online_buffer"])
        history.update(
            {
                name: list(values)
                for name, values in training_state["history"].items()
            }
        )
        global_step = int(training_state["global_step"])
        best_episode = int(training_state["best_episode"])
        best_validation_return = float(
            training_state["best_validation_return"]
        )
        best_validation_metrics = {
            name: float(value)
            for name, value in training_state["best_validation_metrics"].items()
        }
        best_greedy_validation_metrics = {
            name: float(value)
            for name, value in training_state.get(
                "best_greedy_validation_metrics", {}
            ).items()
        }
        best_agent_state = training_state["best_agent_state"]
        np.random.set_state(training_state["numpy_rng_state"])
        torch.random.set_rng_state(training_state["torch_rng_state"].cpu())
        if (
            torch.cuda.is_available()
            and training_state.get("cuda_rng_states") is not None
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in training_state["cuda_rng_states"]]
            )
        start_episode = int(training_state["episode"]) + 1
        print(
            f"online_resumed_from_episode={start_episode - 1:03d}",
            flush=True,
        )

    for episode in range(start_episode, episodes + 1):
        layout_seed = training_layout_seeds[
            (episode - 1) % len(training_layout_seeds)
        ]
        env = build_env(K, layout_seed, cfg)
        state = env.reset(seed + 1_000_000 + episode)
        done = False
        episode_return = 0.0
        time_weighted_aoi = 0.0
        schedule_counts = np.zeros(K + 1, dtype=np.int64)
        agent.train()
        while not done:
            global_step += 1
            if global_step <= random_steps:
                action_cont, action_disc = random_hybrid_action(cfg.d_max, K + 1)
                quality = 0
            else:
                action_cont, action_disc = agent.select_action(
                    env.normalize_observation(state), deterministic=False
                )
                quality = 1
            next_state, reward, done, info = env.step(
                (float(action_cont[0]), float(action_cont[1]), int(action_disc))
            )
            replay_buffer.add(
                env.normalize_observation(state),
                action_cont,
                action_disc,
                reward * reward_scale,
                env.normalize_observation(next_state),
                done,
                quality=quality,
            )
            export_buffer = (
                random_buffer if global_step <= random_steps else online_buffer
            )
            export_buffer.add(
                state,
                action_cont,
                action_disc,
                reward,
                next_state,
                done,
                quality=quality,
            )
            state = next_state
            schedule_counts[int(action_disc)] += 1
            episode_return += reward
            time_weighted_aoi += float(np.mean(env.A)) * float(info["T_total"])
            if global_step > random_steps and len(replay_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    latest_logs = agent.update(
                        replay_buffer.sample(batch_size, device)
                    )

        history["episode"].append(episode)
        history["return"].append(episode_return)
        history["aoi"].append(time_weighted_aoi / max(env.time_elapsed, 1e-12))
        history["total_energy_j"].append(env.cumulative_energy)
        history["time_elapsed"].append(env.time_elapsed)
        history["training_layout_seed"].append(layout_seed)
        history["alpha_continuous"].append(
            latest_logs.get("alpha_continuous", np.nan)
        )
        history["alpha_discrete"].append(
            latest_logs.get("alpha_discrete", np.nan)
        )
        history["discrete_entropy"].append(
            latest_logs.get("discrete_entropy", np.nan)
        )
        history["idle_fraction"].append(
            float(schedule_counts[0] / max(schedule_counts.sum(), 1))
        )
        history["served_devices"].append(
            int(np.count_nonzero(schedule_counts[1:]))
        )
        evaluate_now = (
            episode == 1
            or episode % validation_every == 0
            or episode == episodes
        )
        if evaluate_now:
            agent.eval()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                validation = evaluate_across_layouts(
                    agent,
                    K,
                    validation_layout_seeds,
                    state_normalizer=online_normalizer,
                    cfg=cfg,
                    fading_seed_offset=validation_fading_seed_offset,
                    deterministic=True,
                )
                greedy_validation = {}
                if global_step > random_steps:
                    sample_discrete = agent.cfg.sample_discrete_at_evaluation
                    agent.cfg.sample_discrete_at_evaluation = False
                    try:
                        greedy_validation = evaluate_across_layouts(
                            agent,
                            K,
                            validation_layout_seeds,
                            state_normalizer=online_normalizer,
                            cfg=cfg,
                            fading_seed_offset=validation_fading_seed_offset,
                            deterministic=True,
                        )
                    finally:
                        agent.cfg.sample_discrete_at_evaluation = sample_discrete
            finally:
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            history["eval_episode"].append(episode)
            for name, value in validation.items():
                history[f"eval_{name}"].append(value)
            for name, value in greedy_validation.items():
                history[f"eval_greedy_{name}"].append(value)

            if validation["return"] > best_validation_return:
                best_validation_return = float(validation["return"])
                best_episode = episode
                best_validation_metrics = {
                    name: float(value) for name, value in validation.items()
                }
                best_greedy_validation_metrics = {
                    name: float(value)
                    for name, value in greedy_validation.items()
                }
                best_agent_state = _state_dict_cpu(agent)
                if checkpoint_path is not None:
                    checkpoint = Path(checkpoint_path)
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "agent_state_dict": best_agent_state,
                            "best_episode": best_episode,
                            "best_validation_metrics": best_validation_metrics,
                            "best_greedy_validation_metrics": (
                                best_greedy_validation_metrics
                            ),
                            "protocol_version": DATASET_PROTOCOL,
                        },
                        checkpoint,
                    )

            diagnostics = ""
            if latest_logs:
                diagnostics = (
                    f" alpha_cont={latest_logs['alpha_continuous']:.6f} "
                    f"alpha_disc={latest_logs['alpha_discrete']:.6f} "
                    f"disc_entropy={latest_logs['discrete_entropy']:.6f}"
                )
            if greedy_validation:
                diagnostics += (
                    f" greedy_aoi_s={greedy_validation['aoi']:.6f}"
                )
            print(
                f"episode={episode:03d} train_return={episode_return:.6f} "
                f"train_aoi_s={history['aoi'][-1]:.6f} "
                f"train_energy_kj={env.cumulative_energy / 1000.0:.6f} "
                f"validation_return={validation['return']:.6f} "
                f"validation_aoi_s={validation['aoi']:.6f} "
                f"validation_energy_kj={validation['total_energy_j'] / 1000.0:.6f} "
                f"validation_idle_frac={validation['idle_fraction']:.4f} "
                f"best_episode={best_episode:03d}"
                f"{diagnostics}",
                flush=True,
            )
            if history_path is not None:
                save_history(history, history_path)
            if training_state_path is not None:
                training_state_path.parent.mkdir(parents=True, exist_ok=True)
                run_config = {
                    "protocol_version": DATASET_PROTOCOL,
                    "K": K,
                    "capacity": capacity,
                    "random_steps": random_steps,
                    "batch_size": batch_size,
                    "updates_per_step": updates_per_step,
                    "reward_scale": reward_scale,
                    "online_dataset_fraction": online_dataset_fraction,
                    "validation_every": validation_every,
                    "validation_fading_seed_offset": validation_fading_seed_offset,
                    "training_layout_seeds": list(training_layout_seeds),
                    "validation_layout_seeds": list(validation_layout_seeds),
                }
                training_state = {
                    "run_config": run_config,
                    "episode": episode,
                    "global_step": global_step,
                    "agent_state_dict": agent.state_dict(),
                    "actor_opt": agent.actor_opt.state_dict(),
                    "critic_opt": agent.critic_opt.state_dict(),
                    "alpha_opt": agent.alpha_opt.state_dict(),
                    "replay_buffer": replay_buffer.state_dict(),
                    "random_buffer": random_buffer.state_dict(),
                    "online_buffer": online_buffer.state_dict(),
                    "history": dict(history),
                    "best_episode": best_episode,
                    "best_validation_return": best_validation_return,
                    "best_validation_metrics": best_validation_metrics,
                    "best_greedy_validation_metrics": (
                        best_greedy_validation_metrics
                    ),
                    "best_agent_state": best_agent_state,
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                }
                temporary = training_state_path.with_suffix(
                    training_state_path.suffix + ".tmp"
                )
                torch.save(training_state, temporary)
                temporary.replace(training_state_path)

    if best_agent_state is None:
        raise RuntimeError("No validation checkpoint was produced")

    if not finalize_dataset:
        pilot_history = {
            name: list(values) for name, values in history.items()
        }
        pilot_history["best_episode"] = [best_episode]
        pilot_history["best_validation_return"] = [
            best_validation_metrics["return"]
        ]
        pilot_history["best_validation_aoi"] = [
            best_validation_metrics["aoi"]
        ]
        pilot_history["best_validation_total_energy_j"] = [
            best_validation_metrics["total_energy_j"]
        ]
        pilot_history["best_greedy_validation_aoi"] = [
            best_greedy_validation_metrics.get("aoi", float("inf"))
        ]
        print(
            f"pilot_complete_episode={episodes:03d} "
            f"best_episode={best_episode:03d} "
            f"best_validation_return={best_validation_metrics['return']:.6f} "
            f"best_validation_aoi_s={best_validation_metrics['aoi']:.6f}",
            flush=True,
        )
        return pilot_history

    last_agent_state = _state_dict_cpu(agent)
    agent.load_state_dict(best_agent_state)
    agent.eval()
    good_target = capacity - len(random_buffer) - len(online_buffer)
    if good_target < nominal_good_capacity:
        raise RuntimeError("Export strata exceed dataset capacity")
    good_buffer = ReplayBuffer(reference_env.observation_dim, good_target)
    good_episode = 0
    good_episode_returns: List[float] = []
    good_episode_aois: List[float] = []
    good_episode_energies: List[float] = []
    while len(good_buffer) < good_target:
        good_episode += 1
        collection_seed = seed + 2_000_000 + good_episode
        torch.manual_seed(collection_seed)
        torch.cuda.manual_seed_all(collection_seed)
        layout_seed = training_layout_seeds[
            (good_episode - 1) % len(training_layout_seeds)
        ]
        env = build_env(K, layout_seed, cfg)
        state = env.reset(collection_seed)
        done = False
        episode_return = 0.0
        time_weighted_aoi = 0.0
        while not done and len(good_buffer) < good_target:
            action_cont, action_disc = agent.act(
                env.normalize_observation(state), deterministic=True
            )
            next_state, reward, done, info = env.step(
                (float(action_cont[0]), float(action_cont[1]), int(action_disc))
            )
            good_buffer.add(
                state,
                action_cont,
                action_disc,
                reward,
                next_state,
                done,
                quality=2,
            )
            state = next_state
            episode_return += reward
            time_weighted_aoi += float(np.mean(env.A)) * float(info["T_total"])
        good_episode_returns.append(episode_return)
        good_episode_aois.append(
            time_weighted_aoi / max(env.time_elapsed, 1e-12)
        )
        good_episode_energies.append(env.cumulative_energy)
    last = (good_buffer._ptr - 1) % good_buffer.capacity
    good_buffer.done[last, 0] = 1.0

    dataset_buffer = ReplayBuffer(reference_env.observation_dim, capacity)
    _append_replay_buffer(dataset_buffer, random_buffer, quality=0)
    _append_replay_buffer(dataset_buffer, online_buffer, quality=1)
    _append_replay_buffer(dataset_buffer, good_buffer, quality=2)
    if len(dataset_buffer) != capacity:
        raise RuntimeError(
            f"Expected {capacity} exported transitions, got {len(dataset_buffer)}"
        )

    history["best_episode"] = [best_episode]
    history["best_validation_return"] = [best_validation_metrics["return"]]
    history["best_validation_aoi"] = [best_validation_metrics["aoi"]]
    history["best_validation_total_energy_j"] = [
        best_validation_metrics["total_energy_j"]
    ]
    history["best_greedy_validation_aoi"] = [
        best_greedy_validation_metrics.get("aoi", float("inf"))
    ]
    history["good_collection_return"] = [
        float(np.mean(good_episode_returns))
    ]
    history["good_collection_aoi"] = [float(np.mean(good_episode_aois))]
    history["good_collection_total_energy_j"] = [
        float(np.mean(good_episode_energies))
    ]
    metadata = {
        "seed": seed,
        "K": K,
        "state_dim": reference_env.observation_dim,
        "num_discrete": K + 1,
        "d_max_m": cfg.d_max,
        "mission_time_s": cfg.T_th,
        "reward": "-[lambda * instantaneous_mean_AoI + (1-lambda) * step_energy_kJ]",
        "observation": "uav_xy, relative_xy, AoI, SNR_dB, cumulative_energy_J, remaining_time_s",
        "quality_labels": {
            "0": "random warm-up",
            "1": "rolling late online-SAC",
            "2": "best validated SAC (mean displacement, sampled schedule)",
        },
        "quality_counts": {
            "0": len(random_buffer),
            "1": len(online_buffer),
            "2": len(good_buffer),
        },
        "online_training_episodes": episodes,
        "online_training_steps": global_step,
        "online_reward_scale": reward_scale,
        "offline_reward_scale": reward_scale,
        "discount_gamma": 0.99,
        "online_entropy_temperatures": "separate continuous and categorical",
        "online_continuous_target_entropy": -2.0,
        "online_discrete_target_entropy_ratio": 0.1,
        "online_actor_critic": "permutation-equivariant shared device encoder",
        "online_aoi_scaling": "log1p(AoI)/log1p(A_max)",
        "reward_zscore_normalization": False,
        "online_state_scaling": "fixed physical scaling; exported states are raw",
        "online_dataset_fraction": online_dataset_fraction,
        "training_layout_seeds": list(training_layout_seeds),
        "validation_layout_seeds": list(validation_layout_seeds),
        "held_out_test_layout_seeds": list(TEST_LAYOUT_SEEDS),
        "validation_every_episodes": validation_every,
        "best_episode": best_episode,
        "best_validation_metrics": best_validation_metrics,
        "best_greedy_validation_metrics": best_greedy_validation_metrics,
        "good_collection_metrics": {
            "return": float(np.mean(good_episode_returns)),
            "aoi": float(np.mean(good_episode_aois)),
            "total_energy_j": float(np.mean(good_episode_energies)),
        },
        "protocol_version": DATASET_PROTOCOL,
    }
    dataset_buffer.save_npz(output_dataset, metadata)
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        last_checkpoint = checkpoint.with_name(
            f"{checkpoint.stem}_last{checkpoint.suffix}"
        )
        torch.save(
            {
                "agent_state_dict": last_agent_state,
                "metadata": metadata,
                "selected_for_dataset": False,
            },
            last_checkpoint,
        )
        torch.save(
            {
                "agent_state_dict": best_agent_state,
                "metadata": metadata,
                "best_episode": best_episode,
                "best_validation_metrics": best_validation_metrics,
                "best_greedy_validation_metrics": (
                    best_greedy_validation_metrics
                ),
                "selected_for_dataset": True,
            },
            checkpoint,
        )
    if history_path is not None:
        save_history(history, history_path)
    print(
        f"selected_best_episode={best_episode:03d} "
        f"validation_return={best_validation_metrics['return']:.6f} "
        f"validation_aoi_s={best_validation_metrics['aoi']:.6f} "
        f"validation_energy_kj={best_validation_metrics['total_energy_j'] / 1000.0:.6f} "
        f"good_collection_aoi_s={np.mean(good_episode_aois):.6f}",
        flush=True,
    )
    return dict(history)

