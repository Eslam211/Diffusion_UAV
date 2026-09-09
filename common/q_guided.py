from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import normalize_displacement, one_hot, pack_diffusion_action
from .diffusion import DiffusionHybrid
from .networks import DoubleQCriticHybrid, HybridMLPActor


@dataclass
class QGuidedConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    eta_bc: float = 0.5
    grad_clip_norm: Optional[float] = 1.0
    normalize_q_guidance: bool = True
    target_q_clip: Optional[float] = 500.0


@contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    flags = [parameter.requires_grad for parameter in module.parameters()]
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class QGuidedHybridAgent(nn.Module):
    """Double-Q actor-critic with behavior cloning and direct Q guidance."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        actor: nn.Module,
        critic_hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[QGuidedConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or QGuidedConfig()
        self.device = torch.device(self.cfg.device)
        self.num_discrete = int(num_discrete)
        self.d_max = float(d_max)
        self.actor = actor.to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = DoubleQCriticHybrid(
            state_dim, num_discrete, critic_hidden
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for target in (self.actor_target, self.critic_target):
            target.eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr
        )

    def _soft_update(self) -> None:
        with torch.no_grad():
            for source, target in (
                (self.critic, self.critic_target),
                (self.actor, self.actor_target),
            ):
                for parameter, target_parameter in zip(
                    source.parameters(), target.parameters()
                ):
                    target_parameter.data.lerp_(parameter.data, self.cfg.tau)

    def _critic_q_onehot(
        self,
        critic: DoubleQCriticHybrid,
        state: torch.Tensor,
        action_cont_m: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return critic(
            state,
            normalize_displacement(action_cont_m, self.d_max),
            action_onehot,
        )

    def _critic_q(
        self,
        critic: DoubleQCriticHybrid,
        state: torch.Tensor,
        action_cont_m: torch.Tensor,
        action_disc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._critic_q_onehot(
            critic,
            state,
            action_cont_m,
            one_hot(action_disc, self.num_discrete),
        )

    def _critic_q_all_discrete(
        self,
        critic: DoubleQCriticHybrid,
        state: torch.Tensor,
        action_cont_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return (
            q1.reshape(batch_size, self.num_discrete),
            q2.reshape(batch_size, self.num_discrete),
        )

    @torch.no_grad()
    def act(
        self, state_np: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, int]:
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action = (
            self.actor.sample_eval(state)
            if isinstance(self.actor, HybridMLPActor)
            else self.actor.sample(state)
        )
        return action["a_cont"][0].cpu().numpy(), int(action["a_disc"][0])

    @torch.no_grad()
    def act_batch(
        self, state_np: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        )
        action = (
            self.actor.sample_eval(state)
            if isinstance(self.actor, HybridMLPActor)
            else self.actor.sample(state)
        )
        return action["a_cont"].cpu().numpy(), action["a_disc"].cpu().numpy()

    def _behavior_loss(
        self,
        state: torch.Tensor,
        action_cont: torch.Tensor,
        action_disc: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.actor, DiffusionHybrid):
            packed = pack_diffusion_action(
                action_cont, action_disc, self.d_max, self.num_discrete
            )
            return self.actor.loss(packed, state)
        if isinstance(self.actor, HybridMLPActor):
            return self.actor.bc_loss(state, action_cont, action_disc)
        raise TypeError(f"Unsupported actor type: {type(self.actor).__name__}")

    def _critic_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        state = batch["s"].to(self.device)
        action_cont = batch["a_cont"].to(self.device)
        action_disc = batch["a_disc"].to(self.device).long()
        reward = batch["r"].to(self.device)
        next_state = batch["s2"].to(self.device)
        done = batch["done"].to(self.device).float()
        with torch.no_grad():
            next_action = self.actor_target.sample(next_state)
            q1_target, q2_target = self._critic_q_onehot(
                self.critic_target,
                next_state,
                next_action["a_cont"],
                next_action["a_onehot"],
            )
            next_q = torch.minimum(q1_target, q2_target)
            if self.cfg.target_q_clip is not None:
                next_q = next_q.clamp(
                    -self.cfg.target_q_clip, self.cfg.target_q_clip
                )
            target = reward + self.cfg.gamma * (1.0 - done) * next_q
        q1, q2 = self._critic_q(
            self.critic, state, action_cont, action_disc
        )
        return F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)

    def _actor_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        state = batch["s"].to(self.device)
        action_cont = batch["a_cont"].to(self.device)
        action_disc = batch["a_disc"].to(self.device).long()
        if not 0.0 <= self.cfg.eta_bc <= 1.0:
            raise ValueError("eta_bc must be in [0, 1]")
        behavior_loss = (
            self._behavior_loss(state, action_cont, action_disc)
            if self.cfg.eta_bc > 0.0
            else torch.zeros((), device=self.device)
        )
        mean_q = torch.zeros((), device=self.device)
        q_scale = torch.ones((), device=self.device)
        q_guidance = torch.zeros((), device=self.device)
        if self.cfg.eta_bc < 1.0:
            policy_action = self.actor.sample_train(state)
            with frozen(self.critic):
                q1, q2 = self._critic_q_onehot(
                    self.critic,
                    state,
                    policy_action["a_cont"],
                    policy_action["a_onehot"],
                )
                q = torch.minimum(q1, q2)
                mean_q = q.mean()
                if self.cfg.normalize_q_guidance:
                    q_scale = 1.0 / q.abs().mean().detach().clamp_min(1e-6)
                q_guidance = -q_scale * mean_q
        loss = (
            self.cfg.eta_bc * behavior_loss
            + (1.0 - self.cfg.eta_bc) * q_guidance
        )
        return loss, {
            "behavior_loss": float(behavior_loss.detach()),
            "q_guidance_loss": float(q_guidance.detach()),
            "mean_q": float(mean_q.detach()),
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        critic_loss = self._critic_loss(batch)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.grad_clip_norm
            )
        self.critic_opt.step()

        actor_loss, logs = self._actor_loss(batch)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.grad_clip_norm
            )
        self.actor_opt.step()
        self._soft_update()
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            **logs,
        }

