import argparse
import numpy as np
import pandas as pd
import os
import torch
import json
import matplotlib.pyplot as plt

from env import UAVEnvConfig, UAVOfflineRLEnv
from agent_Online import HybridSACAgent, SACConfig
from Replay_buffer import ReplayBuffer, BufferConfig


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_action(d_max: float, num_discrete: int):
    angle = np.random.uniform(0, 2*np.pi)
    rad = d_max * np.sqrt(np.random.uniform(0, 1))
    w_x = rad * np.cos(angle)
    w_y = rad * np.sin(angle)
    s = np.random.randint(0, num_discrete)
    return np.array([w_x, w_y], dtype=np.float32), int(s)

def save_dataset():
    seed = 0
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    K = 10
    dev_xy = np.random.uniform(0, 1000, size=(K, 2)).astype(np.float32)

    env_cfg = UAVEnvConfig(
        area_size=1000.0,
        h=100.0,
        B=1e6,
        fc=2e9,
        Dk_bits=2e6,
        lam=0.5,
        H_max=400,
        T_th=400.0,
        d_max=25.0, # bounded motion
        fading="rayleigh"
    )
    env = UAVOfflineRLEnv(dev_xy, env_cfg)

    obs_dim = 2 + K + 1
    num_discrete = K + 1
    d_max = env_cfg.d_max

    agent = HybridSACAgent(
        state_dim=obs_dim,
        num_discrete=num_discrete,
        d_max=d_max,
        device=device,
        hidden_dims=(256, 256, 256),
        cfg=SACConfig(
            device=device,
            gamma=0.99,
            tau=0.005,
            actor_lr=1e-4,
            critic_lr=1e-4,
            alpha_lr=3e-4,
            target_entropy_total=None,
            grad_clip_norm=None,
        )
    )

    buffer = ReplayBuffer(
        state_dim=obs_dim,
        cfg=BufferConfig(capacity=300_000, device=device)
    )

    # Training settings
    total_env_steps = 100_000
    start_random_steps = 10_000
    batch_size = 256
    updates_per_step = 1
    log_every = 1_000
    num_episodes = 500
    global_step = 0

    ep_rewards = []
    ep_lengths = []
    ep_times = []

    critic_losses = []
    actor_losses = []
    alpha_vals = []

    ep_aois = []
    ep_energies = []

    # Training loop
    for ep in range(1, num_episodes + 1):
        obs = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_len = 0
        ep_aoi = 0.0
        ep_energy = 0.0

        done = False
        for t in range(1, total_env_steps + 1):
            global_step += 1

            if global_step <= start_random_steps:
                a_cont, a_disc = random_action(d_max, num_discrete)
            else:
                a_cont, a_disc = agent.select_action(obs, deterministic=False)

            next_obs, reward, done, info = env.step((float(a_cont[0]), float(a_cont[1]), int(a_disc)))
            ep_aoi += np.sum(env.A)
            ep_energy += info["E_total"]

            buffer.add(obs, a_cont, a_disc, reward, next_obs, done)

            obs = next_obs
            ep_return += reward
            ep_len += 1

            if global_step > start_random_steps and len(buffer) >= batch_size:
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)
                    logs = agent.update(batch)
                    critic_losses.append(logs["critic_loss"])
                    actor_losses.append(logs["actor_loss"])
                    alpha_vals.append(logs["alpha"])

            if done:
                break

        ep_rewards.append(ep_return)
        ep_lengths.append(ep_len)
        ep_times.append(info["time_elapsed"])

        ep_avg_aoi = ep_aoi / max(ep_len, 1)
        ep_avg_energy = ep_energy / max(ep_len, 1)
        ep_aois.append(ep_avg_aoi)
        ep_energies.append(ep_avg_energy)

        avg_r = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 1 else 0.0
        avg_len = np.mean(ep_lengths[-10:]) if len(ep_lengths) >= 1 else 0.0
        avg_time = np.mean(ep_times[-10:]) if len(ep_times) >= 1 else 0.0
        cl = np.mean(critic_losses[-200:]) if len(critic_losses) >= 1 else 0.0
        al = np.mean(actor_losses[-200:]) if len(actor_losses) >= 1 else 0.0
        av = np.mean(alpha_vals[-200:]) if len(alpha_vals) >= 1 else 0.0

        avg_aoi = np.mean(ep_aois[-10:]) if len(ep_aois) >= 1 else 0.0
        avg_energy = np.mean(ep_energies[-10:]) if len(ep_energies) >= 1 else 0.0

        print(
            f"Episode {ep:>5d} | "
            f"AvgEpReturn(10) {avg_r:>9.2f} | "
            f"AoI {avg_aoi:>8.2f} | "
            f"Energy {avg_energy:>8.2f} | "
            f"AvgEpLen(10) {avg_len:>6.1f} | "
            f"AvgEpTime(10) {avg_time:>8.2f}s | "
            f"CriticLoss {cl:>8.2f} | "
            f"ActorLoss {al:>8.2f} | "
            f"alpha {av:>6.3f}"
        )

    buffer.save_npz("uav_offline_dataset.npz") # Save online dataset for offline RL
    print(f"Saved dataset with {len(buffer)} transitions")

