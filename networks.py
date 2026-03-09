
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, activation=nn.Mish) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.float32 and t.dtype != torch.float64:
            t = t.float()
        device = t.device
        half = self.dim // 2
        emb_scale = math.log(10000.0) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * (-emb_scale))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class HybridDenoiser(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden_dim: int = 256,
        time_dim: int = 32,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        act = nn.Mish if activation.lower() == "mish" else nn.ReLU

        self.state_dim = state_dim
        self.num_discrete = num_discrete
        self.action_dim_total = 2 + num_discrete

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            act(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(state_dim + self.action_dim_total + time_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, self.action_dim_total),
        )

        self.apply(init_weights)

    def forward(self, a_noisy: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        s = s.view(s.size(0), -1)
        x = torch.cat([a_noisy, t_emb, s], dim=-1)
        return self.net(x)

class QNetworkHybrid(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.Mish
        in_dim = state_dim + 2 + num_discrete
        self.net = mlp(in_dim, hidden, 1, activation=act)
        self.apply(init_weights)

    def forward(self, s: torch.Tensor, a_cont_norm: torch.Tensor, a_disc_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a_cont_norm, a_disc_onehot], dim=-1)
        return self.net(x)


class DoubleQCriticHybrid(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.q1 = QNetworkHybrid(state_dim, num_discrete, hidden, activation)
        self.q2 = QNetworkHybrid(state_dim, num_discrete, hidden, activation)

    def forward(self, s: torch.Tensor, a_cont_norm: torch.Tensor, a_disc_onehot: torch.Tensor):
        return self.q1(s, a_cont_norm, a_disc_onehot), self.q2(s, a_cont_norm, a_disc_onehot)
