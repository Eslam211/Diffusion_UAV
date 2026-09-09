from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (self.std + self.eps)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * (self.std + self.eps) + self.mean

    @staticmethod
    def from_data(values: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        return Normalizer(mean, np.where(std < 1e-6, 1.0, std), eps)

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }


@dataclass
class OfflineNormPack:
    state_norm: Normalizer
    reward_norm: Optional[Normalizer] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "state_norm": self.state_norm.to_dict(),
            "reward_norm": None
            if self.reward_norm is None
            else self.reward_norm.to_dict(),
        }


def infer_next_actions(
    action_cont: np.ndarray, action_disc: np.ndarray, done: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    next_cont = action_cont.copy()
    next_disc = action_disc.copy()
    if len(action_cont) > 1:
        nonterminal = done[:-1].reshape(-1) < 0.5
        next_cont[:-1][nonterminal] = action_cont[1:][nonterminal]
        next_disc[:-1][nonterminal] = action_disc[1:][nonterminal]
    return next_cont, next_disc


class OfflineHybridDataset(Dataset):
    """Read-only transition dataset with explicit, reproducible subsetting."""

    def __init__(
        self,
        npz_path: str,
        normalize: bool = True,
        normalize_reward: bool = False,
        reward_scale: float = 0.01,
        stats: Optional[OfflineNormPack] = None,
        subset_size: Optional[int] = None,
        subset_seed: int = 0,
        quality: str = "all",
    ) -> None:
        super().__init__()
        data = np.load(npz_path, allow_pickle=False)
        arrays = {
            "s": data["s"].astype(np.float32),
            "a_cont": data["a_cont"].astype(np.float32),
            "a_disc": data["a_disc"].astype(np.int64),
            "r": data["r"].astype(np.float32).reshape(-1, 1),
            "s2": data["s2"].astype(np.float32),
            "done": data["done"].astype(np.float32).reshape(-1, 1),
        }
        arrays["r"] *= float(reward_scale)
        if "next_a_cont" in data and "next_a_disc" in data:
            arrays["next_a_cont"] = data["next_a_cont"].astype(np.float32)
            arrays["next_a_disc"] = data["next_a_disc"].astype(np.int64)
        else:
            arrays["next_a_cont"], arrays["next_a_disc"] = infer_next_actions(
                arrays["a_cont"], arrays["a_disc"], arrays["done"]
            )
        quality_label = (
            data["quality"].astype(np.int64).reshape(-1)
            if "quality" in data
            else np.ones(len(arrays["s"]), dtype=np.int64)
        )
        if quality not in {"all", "bad", "good"}:
            raise ValueError("quality must be one of: all, bad, good")
        eligible = np.arange(len(quality_label))
        if quality == "bad":
            eligible = eligible[quality_label == quality_label.min()]
        elif quality == "good":
            eligible = eligible[quality_label == quality_label.max()]
        if subset_size is not None:
            if subset_size > len(eligible):
                raise ValueError(
                    f"Requested {subset_size} samples but only {len(eligible)} are eligible"
                )
            rng = np.random.default_rng(subset_seed)
            eligible = np.sort(rng.permutation(eligible)[:subset_size])
        for name in arrays:
            arrays[name] = arrays[name][eligible]

        self.raw_s = arrays["s"].copy()
        self.raw_s2 = arrays["s2"].copy()
        self.stats = stats
        if normalize:
            if self.stats is None:
                self.stats = OfflineNormPack(
                    state_norm=Normalizer.from_data(
                        np.concatenate([arrays["s"], arrays["s2"]], axis=0)
                    ),
                    reward_norm=Normalizer.from_data(arrays["r"])
                    if normalize_reward
                    else None,
                )
            arrays["s"] = self.stats.state_norm.normalize(arrays["s"]).astype(
                np.float32
            )
            arrays["s2"] = self.stats.state_norm.normalize(arrays["s2"]).astype(
                np.float32
            )
            if normalize_reward and self.stats.reward_norm is not None:
                arrays["r"] = self.stats.reward_norm.normalize(arrays["r"]).astype(
                    np.float32
                )
        self.arrays = arrays
        self.N = len(arrays["s"])
        self.metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(values[index]) for name, values in self.arrays.items()
        }


def make_dataloader(
    npz_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True,
    normalize_reward: bool = False,
    reward_scale: float = 0.01,
    stats: Optional[OfflineNormPack] = None,
    subset_size: Optional[int] = None,
    subset_seed: int = 0,
    quality: str = "all",
) -> Tuple[DataLoader, OfflineHybridDataset]:
    dataset = OfflineHybridDataset(
        npz_path,
        normalize=normalize,
        normalize_reward=normalize_reward,
        reward_scale=reward_scale,
        stats=stats,
        subset_size=subset_size,
        subset_seed=subset_seed,
        quality=quality,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return loader, dataset


class ReplayBuffer:
    def __init__(self, state_dim: int, capacity: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a_cont = np.zeros((capacity, 2), dtype=np.float32)
        self.a_disc = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.quality = np.zeros(capacity, dtype=np.int64)
        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def chronological_indices(self) -> np.ndarray:
        """Indices from oldest to newest, including after a circular wrap."""
        if self._size < self.capacity:
            return np.arange(self._size, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(self._ptr, self.capacity, dtype=np.int64),
                np.arange(0, self._ptr, dtype=np.int64),
            )
        )

    def state_dict(self) -> Dict[str, object]:
        """Serializable training state for interruption-safe online runs."""
        return {
            "capacity": self.capacity,
            "s": self.s,
            "a_cont": self.a_cont,
            "a_disc": self.a_disc,
            "r": self.r,
            "s2": self.s2,
            "done": self.done,
            "quality": self.quality,
            "_size": self._size,
            "_ptr": self._ptr,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"Replay capacity mismatch: {state['capacity']} != {self.capacity}"
            )
        for name in ("s", "a_cont", "a_disc", "r", "s2", "done", "quality"):
            source = np.asarray(state[name])
            destination = getattr(self, name)
            if source.shape != destination.shape:
                raise ValueError(
                    f"Replay field {name} shape mismatch: "
                    f"{source.shape} != {destination.shape}"
                )
            destination[...] = source
        self._size = int(state["_size"])
        self._ptr = int(state["_ptr"])

    def add(
        self,
        state: np.ndarray,
        action_cont: np.ndarray,
        action_disc: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        quality: int = 1,
    ) -> None:
        index = self._ptr
        self.s[index] = state
        self.a_cont[index] = action_cont
        self.a_disc[index] = int(action_disc)
        self.r[index, 0] = float(reward)
        self.s2[index] = next_state
        self.done[index, 0] = float(done)
        self.quality[index] = int(quality)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: str, include_next_actions: bool = False
    ) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self._size, size=batch_size)
        arrays = {
            "s": self.s[indices],
            "a_cont": self.a_cont[indices],
            "a_disc": self.a_disc[indices],
            "r": self.r[indices],
            "s2": self.s2[indices],
            "done": self.done[indices],
        }
        if include_next_actions:
            next_cont, next_disc = infer_next_actions(
                self.a_cont[: self._size],
                self.a_disc[: self._size],
                self.done[: self._size],
            )
            arrays["next_a_cont"] = next_cont[indices]
            arrays["next_a_disc"] = next_disc[indices]
        return {
            name: torch.as_tensor(values, device=device)
            for name, values in arrays.items()
        }

    def save_npz(self, path: str, metadata: Optional[Dict[str, object]] = None) -> None:
        indices = self.chronological_indices()
        action_cont = self.a_cont[indices]
        action_disc = self.a_disc[indices]
        done = self.done[indices]
        next_cont, next_disc = infer_next_actions(
            action_cont, action_disc, done
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            s=self.s[indices],
            a_cont=action_cont,
            a_disc=action_disc,
            r=self.r[indices],
            s2=self.s2[indices],
            done=done,
            next_a_cont=next_cont,
            next_a_disc=next_disc,
            quality=self.quality[indices],
            metadata_json=json.dumps(metadata or {}, sort_keys=True),
        )


def save_normalization(stats: OfflineNormPack, path: str) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
