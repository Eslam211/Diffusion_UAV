from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class BufferConfig:
    capacity: int = 1_000_000
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, state_dim: int, cfg: Optional[BufferConfig] = None) -> None:
        self.cfg = cfg or BufferConfig()
        self.device = torch.device(self.cfg.device)

        self.state_dim = int(state_dim)
        self.capacity = int(self.cfg.capacity)

        self._ptr = 0
        self._size = 0

        self.s = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.a_cont = np.zeros((self.capacity, 2), dtype=np.float32)
        self.a_disc = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        s: np.ndarray,
        a_cont: np.ndarray,
        a_disc: int,
        r: float,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr

        self.s[i] = s
        self.a_cont[i] = a_cont
        self.a_disc[i] = int(a_disc)
        self.r[i, 0] = float(r)
        self.s2[i] = s2
        self.done[i, 0] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        assert self._size > 0, "ReplayBuffer is empty."
        batch_size = int(batch_size)
        idx = np.random.randint(0, self._size, size=batch_size)

        batch = {
            "s": torch.as_tensor(self.s[idx], device=self.device),
            "a_cont": torch.as_tensor(self.a_cont[idx], device=self.device),
            "a_disc": torch.as_tensor(self.a_disc[idx], device=self.device),
            "r": torch.as_tensor(self.r[idx], device=self.device),
            "s2": torch.as_tensor(self.s2[idx], device=self.device),
            "done": torch.as_tensor(self.done[idx], device=self.device),
        }
        return batch

    def save_npz(self, path: str) -> None:
        n = self._size
        np.savez_compressed(
            path,
            s=self.s[:n],
            a_cont=self.a_cont[:n],
            a_disc=self.a_disc[:n],
            r=self.r[:n],
            s2=self.s2[:n],
            done=self.done[:n],
        )

    def load_npz(self, path: str) -> None:
        data = np.load(path)
        s = data["s"].astype(np.float32)
        a_cont = data["a_cont"].astype(np.float32)
        a_disc = data["a_disc"].astype(np.int64)
        r = data["r"].astype(np.float32)
        s2 = data["s2"].astype(np.float32)
        done = data["done"].astype(np.float32)

        n = s.shape[0]
        if n > self.capacity:
            raise ValueError(f"Dataset size {n} exceeds buffer capacity {self.capacity}.")

        self.s[:n] = s
        self.a_cont[:n] = a_cont
        self.a_disc[:n] = a_disc
        self.r[:n] = r
        self.s2[:n] = s2
        self.done[:n] = done

        self._size = n
        self._ptr = n % self.capacity
