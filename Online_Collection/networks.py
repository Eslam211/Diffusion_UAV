from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class HybridActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        d_max: float = 25.0,
        log_std_bounds: Tuple[float, float] = (LOG_STD_MIN, LOG_STD_MAX),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_discrete = num_discrete
        self.d_max = float(d_max)
        self.log_std_min, self.log_std_max = log_std_bounds

        self.trunk = mlp(state_dim, hidden_dims, hidden_dims[-1])
        trunk_out = hidden_dims[-1]

        self.mu = nn.Linear(trunk_out, 2)
        self.log_std = nn.Linear(trunk_out, 2)

        self.logits = nn.Linear(trunk_out, num_discrete)

        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        logits = self.logits(h)
        return {"mu": mu, "log_std": log_std, "logits": logits}

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(state)
        if deterministic:
            a_u = torch.tanh(out["mu"])
            a_cont = a_u * self.d_max
            a_disc = torch.argmax(out["logits"], dim=-1)
            return a_cont, a_disc

        a_cont, a_disc, _, _ = self.sample(state)
        return a_cont, a_disc

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(state)
        mu, log_std, logits = out["mu"], out["log_std"], out["logits"]
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample() # (B,2)
        a_u = torch.tanh(z)
        a_cont = a_u * self.d_max # scale to meters

        logp_cont = normal.log_prob(z) # (B,2)
        logp_cont = logp_cont.sum(dim=-1, keepdim=True)
        corr = torch.log(1.0 - a_u.pow(2) + EPS).sum(dim=-1, keepdim=True)
        logp_cont = logp_cont - corr

        cat = torch.distributions.Categorical(logits=logits)
        a_disc = cat.sample()
        logp_disc = cat.log_prob(a_disc).unsqueeze(-1)

        logp_total = logp_cont + logp_disc

        extra = {
            "mu": mu,
            "log_std": log_std,
            "logits": logits,
            "logp_cont": logp_cont,
            "logp_disc": logp_disc,
        }
        return a_cont, a_disc, logp_total, extra

    def log_prob_given_action(
        self, state: torch.Tensor, a_cont: torch.Tensor, a_disc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(state)
        mu, log_std, logits = out["mu"], out["log_std"], out["logits"]
        std = torch.exp(log_std)

        a_u = torch.clamp(a_cont / (self.d_max + EPS), -1 + 1e-6, 1 - 1e-6)
        z = atanh(a_u)

        normal = torch.distributions.Normal(mu, std)
        logp_cont = normal.log_prob(z).sum(dim=-1, keepdim=True)
        corr = torch.log(1.0 - a_u.pow(2) + EPS).sum(dim=-1, keepdim=True)
        logp_cont = logp_cont - corr

        cat = torch.distributions.Categorical(logits=logits)
        logp_disc = cat.log_prob(a_disc).unsqueeze(-1)

        return logp_cont + logp_disc, logp_cont, logp_disc


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_discrete = num_discrete
        in_dim = state_dim + 2 + num_discrete
        self.net = mlp(in_dim, hidden_dims, 1)
        self.apply(init_weights)

    def forward(self, state: torch.Tensor, a_cont: torch.Tensor, a_disc: torch.Tensor) -> torch.Tensor:
        a_disc = a_disc.long()
        one_hot = F.one_hot(a_disc, num_classes=self.num_discrete).float()
        x = torch.cat([state, a_cont, one_hot], dim=-1)
        return self.net(x)


class DoubleQCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, num_discrete, hidden_dims)
        self.q2 = QNetwork(state_dim, num_discrete, hidden_dims)

    def forward(self, state: torch.Tensor, a_cont: torch.Tensor, a_disc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, a_cont, a_disc), self.q2(state, a_cont, a_disc)
