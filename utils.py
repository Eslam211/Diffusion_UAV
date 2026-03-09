
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    @staticmethod
    def from_data(x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return Normalizer(mean=mean, std=std, eps=eps)


@dataclass
class OfflineNormPack:
    state_norm: Normalizer
    reward_norm: Optional[Normalizer] = None

    def norm_state(self, s: np.ndarray) -> np.ndarray:
        return self.state_norm.normalize(s)

    def norm_reward(self, r: np.ndarray) -> np.ndarray:
        if self.reward_norm is None:
            return r
        return self.reward_norm.normalize(r)

class OfflineHybridDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        device: str = "cpu",
        normalize: bool = True,
        normalize_reward: bool = True,
        stats: Optional[OfflineNormPack] = None,
    ) -> None:
        super().__init__()
        data = np.load(npz_path)

        self.s = data["s"].astype(np.float32)
        self.a_cont = data["a_cont"].astype(np.float32)
        self.a_disc = data["a_disc"].astype(np.int64)
        self.r = data["r"].astype(np.float32)
        self.s2 = data["s2"].astype(np.float32)
        self.done = data["done"].astype(np.float32)
        
        self.s       = self.s[10000:]
        self.a_cont = self.a_cont[10000:]
        self.a_disc = self.a_disc[10000:]
        self.r      = self.r[10000:]
        self.s2  = self.s2[10000:]
        self.done        = self.done[10000:]
        
        N = 10000
        total_size = self.s.shape[0]

        indices = np.random.choice(total_size, size=N, replace=False)

        self.s       = self.s[indices]
        self.a_cont = self.a_cont[indices]
        self.a_disc = self.a_disc[indices]
        self.r      = self.r[indices]
        self.s2  = self.s2[indices]
        self.done        = self.done[indices]
        
        np.savez(
            "uav_offline_dataset_filtered_10k.npz",
            s=self.s,
            a_cont=self.a_cont,
            a_disc=self.a_disc,
            r=self.r,
            s2=self.s2,
            done=self.done
        )

        self.N = self.s.shape[0]
        self.device = torch.device(device)

        self.normalize = bool(normalize)
        if self.normalize:
            if stats is None:
                state_norm = Normalizer.from_data(self.s)
                reward_norm = Normalizer.from_data(self.r) if normalize_reward else None
                self.stats = OfflineNormPack(state_norm=state_norm, reward_norm=reward_norm)
            else:
                self.stats = stats

            self.s = self.stats.norm_state(self.s).astype(np.float32)
            self.s2 = self.stats.norm_state(self.s2).astype(np.float32)
            if normalize_reward and self.stats.reward_norm is not None:
                self.r = self.stats.norm_reward(self.r).astype(np.float32)
        else:
            self.stats = stats

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "s": torch.as_tensor(self.s[idx], device=self.device),
            "a_cont": torch.as_tensor(self.a_cont[idx], device=self.device),
            "a_disc": torch.as_tensor(self.a_disc[idx], device=self.device),
            "r": torch.as_tensor(self.r[idx], device=self.device),
            "s2": torch.as_tensor(self.s2[idx], device=self.device),
            "done": torch.as_tensor(self.done[idx], device=self.device),
        }


def make_dataloader(
    npz_path: str,
    batch_size: int,
    device: str,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True,
    normalize_reward: bool = True,
    stats: Optional[OfflineNormPack] = None,
) -> Tuple[DataLoader, OfflineHybridDataset]:
    ds = OfflineHybridDataset(
        npz_path=npz_path,
        device=device,
        normalize=normalize,
        normalize_reward=normalize_reward,
        stats=stats,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return dl, ds


@torch.no_grad()
def evaluate_policy(
    env,
    agent,
    n_episodes: int = 5,
    deterministic: bool = True,
    state_normalizer: Optional[Normalizer] = None,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    
    returns = []
    avg_aoi_steps = []
    avg_energy_steps = []
    lens = []
    times = []

    for ep in range(n_episodes):
        obs = env.reset(seed=None if seed is None else (seed + ep))
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_aoi_sum = 0.0
        ep_energy_sum = 0.0
        last_info = {}

        while not done:
            obs_in = obs
            if state_normalizer is not None:
                obs_in = state_normalizer.normalize(obs_in)

            a_cont, a_disc = agent.act(obs_in, deterministic=deterministic)
            obs, r, done, info = env.step((float(a_cont[0]), float(a_cont[1]), int(a_disc)))

            ep_ret += float(r)
            ep_len += 1

            ep_aoi_sum += float(np.sum(env.A)) / float(env.K)
            ep_energy_sum += float(info["E_total"])
            last_info = info

        returns.append(ep_ret)
        avg_aoi_steps.append(ep_aoi_sum / max(ep_len, 1))
        avg_energy_steps.append(ep_energy_sum / max(ep_len, 1))
        lens.append(ep_len)
        times.append(float(last_info.get("time_elapsed", 0.0)))

    return {
        "return": float(np.mean(returns)),
        "aoi": float(np.mean(avg_aoi_steps)),
        "energy": float(np.mean(avg_energy_steps)),
        "ep_len": float(np.mean(lens)),
        "time_elapsed": float(np.mean(times)),
    }

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
