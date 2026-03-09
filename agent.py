from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import DoubleQCriticHybrid
from diffusion import DiffusionHybrid


@dataclass
class OfflineAgentConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005

    critic_lr: float = 5e-4
    actor_lr: float = 5e-4

    eta_bc: float = 0.7

    cql_alpha: float = 1.0
    cql_num_random: int = 10
    cql_temp: float = 1.0

    grad_clip_norm: Optional[float] = 1.0


class DiffusionCQLHybridAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        actor: DiffusionHybrid,
        critic_hidden=(256, 256, 256),
        cfg: Optional[OfflineAgentConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or OfflineAgentConfig()
        self.device = torch.device(self.cfg.device)

        self.state_dim = int(state_dim)
        self.num_discrete = int(num_discrete)
        self.d_max = float(d_max)

        self.actor = actor.to(self.device)

        self.critic = DoubleQCriticHybrid(
            state_dim=state_dim,
            num_discrete=num_discrete,
            hidden=critic_hidden,
            activation="relu",
        ).to(self.device)

        self.critic_target = DoubleQCriticHybrid(
            state_dim=state_dim,
            num_discrete=num_discrete,
            hidden=critic_hidden,
            activation="relu",
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1.0 - tau)
                p_t.data.add_(tau * p.data)

    def _to_onehot(self, a_disc: torch.Tensor) -> torch.Tensor:
        return F.one_hot(a_disc.long(), num_classes=self.num_discrete).float()

    def _cont_to_norm(self, a_cont_m: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a_cont_m / (self.d_max + 1e-8), -1.0, 1.0)

    def _pack_action_vec0(self, a_cont_m: torch.Tensor, a_disc: torch.Tensor) -> torch.Tensor:
        a_cont_norm = self._cont_to_norm(a_cont_m)
        a_onehot = self._to_onehot(a_disc)
        a_onehot_pm = a_onehot * 2.0 - 1.0
        return torch.cat([a_cont_norm, a_onehot_pm], dim=-1)

    def _critic_q(self, s: torch.Tensor, a_cont_m: torch.Tensor, a_disc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_cont_norm = self._cont_to_norm(a_cont_m)
        a_onehot = self._to_onehot(a_disc)
        return self.critic(s, a_cont_norm, a_onehot)

    @torch.no_grad()
    def act(self, s_np: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, int]:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.actor.sample(s, return_probs=False)
        a_cont = out["a_cont"].squeeze(0).cpu().numpy()
        a_disc = int(out["a_disc"].item())
        return a_cont, a_disc

    def _critic_bellman_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        s = batch["s"].to(self.device)
        a_cont = batch["a_cont"].to(self.device)
        a_disc = batch["a_disc"].to(self.device).long()
        r = batch["r"].to(self.device)
        s2 = batch["s2"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            out2 = self.actor.sample(s2, return_probs=False)
            a2_cont = out2["a_cont"]
            a2_disc = out2["a_disc"]

            q1_t, q2_t = self._critic_target_q(s2, a2_cont, a2_disc)
            q_t = torch.min(q1_t, q2_t)
            q_t = torch.clamp(q_t, -100.0, 100.0)
            y = r + cfg.gamma * (1.0 - done) * q_t

        q1, q2 = self._critic_q(s, a_cont, a_disc)
        bellman = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
        aux = {"s": s, "a_cont": a_cont, "a_disc": a_disc, "y": y, "q1": q1, "q2": q2}
        return bellman, aux

    def _critic_target_q(self, s: torch.Tensor, a_cont_m: torch.Tensor, a_disc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_cont_norm = self._cont_to_norm(a_cont_m)
        a_onehot = self._to_onehot(a_disc)
        return self.critic_target(s, a_cont_norm, a_onehot)

    def _cql_regularizer(self, s: torch.Tensor, q1_data: torch.Tensor, q2_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        b = s.size(0)
        n = int(cfg.cql_num_random)

        rand_cont = (torch.rand((b * n, 2), device=self.device) * 2.0 - 1.0) * self.d_max
        rand_disc = torch.randint(0, self.num_discrete, (b * n,), device=self.device)
        rand_onehot = self._to_onehot(rand_disc)
        rand_cont_norm = self._cont_to_norm(rand_cont)

        s_rep = s.unsqueeze(1).repeat(1, n, 1).view(b * n, -1)

        q1_rand = self.critic.q1(s_rep, rand_cont_norm, rand_onehot).view(b, n, 1)
        q2_rand = self.critic.q2(s_rep, rand_cont_norm, rand_onehot).view(b, n, 1)

        with torch.no_grad():
            out_pi = self.actor.sample(s, return_probs=False)
            pi_cont = out_pi["a_cont"]
            pi_disc = out_pi["a_disc"]
        pi_onehot = self._to_onehot(pi_disc)
        pi_cont_norm = self._cont_to_norm(pi_cont)

        q1_pi = self.critic.q1(s, pi_cont_norm, pi_onehot)
        q2_pi = self.critic.q2(s, pi_cont_norm, pi_onehot)

        q1_cat = torch.cat([q1_rand, q1_pi.unsqueeze(1)], dim=1)
        q2_cat = torch.cat([q2_rand, q2_pi.unsqueeze(1)], dim=1)

        cql1 = (torch.logsumexp(q1_cat / cfg.cql_temp, dim=1) * cfg.cql_temp).mean() - q1_data.mean()
        cql2 = (torch.logsumexp(q2_cat / cfg.cql_temp, dim=1) * cfg.cql_temp).mean() - q2_data.mean()

        return cql1, cql2

    def _actor_losses(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        s = batch["s"].to(self.device)
        a_cont = batch["a_cont"].to(self.device)
        a_disc = batch["a_disc"].to(self.device).long()

        a_vec0 = self._pack_action_vec0(a_cont, a_disc)
        bc_loss = self.actor.loss(a_vec0, s)

        out_pi = self.actor.sample(s, return_probs=False)
        pi_cont = out_pi["a_cont"]
        pi_disc = out_pi["a_disc"]

        q1_pi, q2_pi = self._critic_q(s, pi_cont, pi_disc)
        q_pi = torch.min(q1_pi, q2_pi)
        g_loss = -q_pi.mean()

        actor_loss = cfg.eta_bc * bc_loss + (1.0 - cfg.eta_bc) * g_loss

        logs = {
            "bc_loss": float(bc_loss.item()),
            "q_guidance_loss": float(g_loss.item()),
            "mean_q_pi": float(q_pi.mean().item()),
        }
        return actor_loss, logs

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg

        # critic
        bellman_loss, aux = self._critic_bellman_loss(batch)
        s = aux["s"]
        q1_data, q2_data = aux["q1"], aux["q2"]

        cql1, cql2 = self._cql_regularizer(s, q1_data, q2_data)
        critic_loss = bellman_loss + cfg.cql_alpha * (cql1 + cql2)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        # actor
        actor_loss, actor_logs = self._actor_losses(batch)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_opt.step()

        self._soft_update()

        return {
            "critic_total": float(critic_loss.item()),
            "critic_bellman": float(bellman_loss.item()),
            "cql1": float(cql1.item()),
            "cql2": float(cql2.item()),
            "actor_total": float(actor_loss.item()),
            **actor_logs,
        }
