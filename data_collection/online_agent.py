from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.actions import normalize_displacement, one_hot
from common.q_guided import frozen
from common.networks import DoubleQSetCriticHybrid, HybridSetActor


@dataclass
class SACConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    initial_alpha_continuous: float = 0.01
    initial_alpha_discrete: float = 0.01
    target_entropy_continuous: float = -2.0
    target_entropy_discrete: Optional[float] = None
    # A low nonzero categorical target permits decisive scheduling while still
    # recovering if the policy collapses prematurely to one action.
    target_entropy_discrete_ratio: float = 0.1
    min_alpha: float = 1e-4
    max_alpha: float = 1.0
    grad_clip_norm: Optional[float] = 10.0
    # Evaluation/deployment keeps the continuous mean but samples the learned
    # categorical scheduling distribution.
    sample_discrete_at_evaluation: bool = True


class HybridSACAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[SACConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SACConfig()
        self.device = torch.device(self.cfg.device)
        self.num_discrete = int(num_discrete)
        self.d_max = float(d_max)
        self.actor = HybridSetActor(
            state_dim, num_discrete, d_max, hidden
        ).to(self.device)
        self.critic = DoubleQSetCriticHybrid(
            state_dim, num_discrete, hidden
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr
        )
        self.log_alpha_continuous = nn.Parameter(
            torch.tensor(
                math.log(self.cfg.initial_alpha_continuous), device=self.device
            )
        )
        self.log_alpha_discrete = nn.Parameter(
            torch.tensor(
                math.log(self.cfg.initial_alpha_discrete), device=self.device
            )
        )
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha_continuous, self.log_alpha_discrete],
            lr=self.cfg.alpha_lr,
        )
        self.target_entropy_continuous = float(
            self.cfg.target_entropy_continuous
        )
        self.target_entropy_discrete = (
            self.cfg.target_entropy_discrete_ratio * math.log(num_discrete)
            if self.cfg.target_entropy_discrete is None
            else float(self.cfg.target_entropy_discrete)
        )
        if not 0.0 <= self.target_entropy_discrete <= math.log(num_discrete):
            raise ValueError("categorical target entropy must be in [0, log(N)]")
        if not 0.0 < self.cfg.min_alpha <= self.cfg.max_alpha:
            raise ValueError("alpha bounds must satisfy 0 < min_alpha <= max_alpha")

    @property
    def alpha_continuous(self) -> torch.Tensor:
        return self.log_alpha_continuous.exp()

    @property
    def alpha_discrete(self) -> torch.Tensor:
        return self.log_alpha_discrete.exp()

    def _q(
        self,
        critic: DoubleQSetCriticHybrid,
        state: torch.Tensor,
        action: Dict[str, torch.Tensor],
    ):
        return critic(
            state,
            normalize_displacement(action["a_cont"], self.d_max),
            action["a_onehot"],
        )

    def _q_all_discrete(
        self,
        critic: DoubleQSetCriticHybrid,
        state: torch.Tensor,
        action_cont_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every scheduling action for one sampled displacement.

        The actor factorizes the continuous and categorical distributions given
        the state, so summing over the small categorical branch is exact and has
        much lower variance than differentiating through one Gumbel sample.
        """
        batch_size = state.shape[0]
        state_all = state[:, None, :].expand(
            batch_size, self.num_discrete, state.shape[-1]
        )
        cont_norm = normalize_displacement(action_cont_m, self.d_max)
        cont_all = cont_norm[:, None, :].expand(
            batch_size, self.num_discrete, cont_norm.shape[-1]
        )
        disc_all = torch.eye(
            self.num_discrete, device=state.device, dtype=state.dtype
        )[None, :, :].expand(batch_size, -1, -1)
        q1, q2 = critic(
            state_all.reshape(batch_size * self.num_discrete, -1),
            cont_all.reshape(batch_size * self.num_discrete, -1),
            disc_all.reshape(batch_size * self.num_discrete, -1),
        )
        return q1.reshape(batch_size, self.num_discrete), q2.reshape(
            batch_size, self.num_discrete
        )

    @torch.no_grad()
    def select_action(
        self, state_np: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, int]:
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action = (
            self.actor.sample(state)
            if deterministic
            else self.actor.sample_train(
                state,
                stochastic_continuous=True,
                stochastic_discrete=True,
            )
        )
        return action["a_cont"][0].cpu().numpy(), int(action["a_disc"][0])

    @torch.no_grad()
    def act(
        self, state_np: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Evaluation-compatible action; input must already be normalized."""
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if deterministic and self.cfg.sample_discrete_at_evaluation:
            action = self.actor.sample_eval(state)
        elif deterministic:
            action = self.actor.sample(state)
        else:
            action = self.actor.sample_train(
                state,
                stochastic_continuous=True,
                stochastic_discrete=True,
            )
        return action["a_cont"][0].cpu().numpy(), int(action["a_disc"][0])

    @torch.no_grad()
    def act_batch(
        self, state_np: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched evaluation path; input must already be normalized."""
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        )
        if deterministic and self.cfg.sample_discrete_at_evaluation:
            action = self.actor.sample_eval(state)
        elif deterministic:
            action = self.actor.sample(state)
        else:
            action = self.actor.sample_train(
                state,
                stochastic_continuous=True,
                stochastic_discrete=True,
            )
        return action["a_cont"].cpu().numpy(), action["a_disc"].cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        reward = batch["r"].to(self.device)
        next_state = batch["s2"].to(self.device)
        done = batch["done"].to(self.device).float()
        with torch.no_grad():
            next_action = self.actor.sample_train(
                next_state,
                stochastic_continuous=True,
                stochastic_discrete=False,
            )
            tq1, tq2 = self._q_all_discrete(
                self.critic_target, next_state, next_action["a_cont"]
            )
            next_log_probs = F.log_softmax(next_action["logits"], dim=-1)
            next_expected_q = (
                next_action["a_probs"] * torch.minimum(tq1, tq2)
            ).sum(dim=-1, keepdim=True)
            next_expected_logp_discrete = (
                next_action["a_probs"] * next_log_probs
            ).sum(dim=-1, keepdim=True)
            next_soft_value = (
                next_expected_q
                - self.alpha_continuous.detach() * next_action["logp_cont"]
                - self.alpha_discrete.detach() * next_expected_logp_discrete
            )
            target = reward + self.cfg.gamma * (1.0 - done) * (
                next_soft_value
            )
        data_onehot = one_hot(disc, self.num_discrete)
        cont_norm = normalize_displacement(cont, self.d_max)
        q1, q2 = self.critic(state, cont_norm, data_onehot)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.grad_clip_norm
            )
        self.critic_opt.step()

        action = self.actor.sample_train(
            state,
            stochastic_continuous=True,
            stochastic_discrete=False,
        )
        with frozen(self.critic):
            pq1, pq2 = self._q_all_discrete(
                self.critic, state, action["a_cont"]
            )
            log_probs = F.log_softmax(action["logits"], dim=-1)
            expected_q = (
                action["a_probs"] * torch.minimum(pq1, pq2)
            ).sum(dim=-1, keepdim=True)
            expected_logp_discrete = (
                action["a_probs"] * log_probs
            ).sum(dim=-1, keepdim=True)
            actor_loss = (
                self.alpha_continuous.detach() * action["logp_cont"]
                + self.alpha_discrete.detach() * expected_logp_discrete
                - expected_q
            ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.grad_clip_norm
            )
        self.actor_opt.step()

        discrete_entropy = -expected_logp_discrete
        continuous_alpha_loss = -(
            self.log_alpha_continuous
            * (
                action["logp_cont"].detach()
                + self.target_entropy_continuous
            )
        ).mean()
        discrete_alpha_loss = (
            self.log_alpha_discrete
            * (discrete_entropy.detach() - self.target_entropy_discrete)
        ).mean()
        alpha_loss = continuous_alpha_loss + discrete_alpha_loss
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            lower = math.log(self.cfg.min_alpha)
            upper = math.log(self.cfg.max_alpha)
            self.log_alpha_continuous.clamp_(lower, upper)
            self.log_alpha_discrete.clamp_(lower, upper)

        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_parameter.data.lerp_(parameter.data, self.cfg.tau)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha_continuous": float(self.alpha_continuous.detach()),
            "alpha_discrete": float(self.alpha_discrete.detach()),
            "continuous_logp": float(action["logp_cont"].mean().detach()),
            "discrete_entropy": float(discrete_entropy.mean().detach()),
        }


def random_hybrid_action(d_max: float, num_discrete: int) -> Tuple[np.ndarray, int]:
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    radius = d_max * np.sqrt(np.random.uniform(0.0, 1.0))
    action = np.array(
        [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
    )
    return action, int(np.random.randint(num_discrete))


