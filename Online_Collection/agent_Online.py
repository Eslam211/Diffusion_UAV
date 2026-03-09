
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import HybridActor, DoubleQCritic


@dataclass
class SACConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    target_entropy_total: Optional[float] = None
    grad_clip_norm: Optional[float] = 10.0


class HybridSACAgent:
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        device: str,
        hidden_dims=(256, 256, 256),
        cfg: Optional[SACConfig] = None,
    ) -> None:
        self.cfg = cfg or SACConfig()
        self.device = device

        self.actor = HybridActor(
            state_dim=state_dim,
            num_discrete=num_discrete,
            hidden_dims=hidden_dims,
            d_max=d_max,
        ).to(self.device)

        self.critic = DoubleQCritic(
            state_dim=state_dim,
            num_discrete=num_discrete,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.critic_target = DoubleQCritic(
            state_dim=state_dim,
            num_discrete=num_discrete,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.log_alpha = torch.tensor(0.2, device=self.device, requires_grad=False)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        if self.cfg.target_entropy_total is None:
            self.target_entropy = -(2.0 + float(np.log(num_discrete)))
        else:
            self.target_entropy = float(self.cfg.target_entropy_total)

        self.num_discrete = num_discrete
        self.state_dim = state_dim

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, int]:
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_cont, a_disc = self.actor.act(s, deterministic=deterministic)
        return a_cont.squeeze(0).cpu().numpy(), int(a_disc.item())

    def _soft_update(self, src: nn.Module, tgt: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p, p_t in zip(src.parameters(), tgt.parameters()):
                p_t.data.mul_(1.0 - tau)
                p_t.data.add_(tau * p.data)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        s = batch["s"].to(self.device)
        a_cont = batch["a_cont"].to(self.device)
        a_disc = batch["a_disc"].to(self.device).long()
        r = batch["r"].to(self.device)
        s2 = batch["s2"].to(self.device)
        done = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            a2_cont, a2_disc, logp2, _ = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, a2_cont, a2_disc)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            y = r + cfg.gamma * (1.0 - done) * q_t

        q1, q2 = self.critic(s, a_cont, a_disc)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        # Actor update
        a_new_cont, a_new_disc, logp, _ = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, a_new_cont, a_new_disc)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()

        self._soft_update(self.critic, self.critic_target, cfg.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_logp": float(logp.mean().item()),
            "mean_q_pi": float(q_pi.mean().item()),
        }
