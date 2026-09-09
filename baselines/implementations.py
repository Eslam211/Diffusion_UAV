"""Hybrid-action implementations of the comparison methods."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.actions import (
    hybrid_bc_distance,
    normalize_displacement,
    one_hot,
    pack_diffusion_action,
    project_displacement_torch,
    straight_through_softmax,
)
from common.q_guided import QGuidedConfig, QGuidedHybridAgent, frozen
from common.diffusion import DiffusionHybrid
from common.networks import (
    DoubleQCriticHybrid,
    HybridMLPActor,
    HybridPerturbation,
    HybridVAE,
    ValueNetwork,
)


def expectile_loss(difference: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(difference > 0, expectile, 1.0 - expectile)
    return weight * difference.pow(2)


@dataclass
class CQLConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    cql_alpha: float = 1.0
    entropy_alpha: float = 0.2
    cql_num_random: int = 10


class CQLHybridAgent(QGuidedHybridAgent):
    """Maximum-entropy CQL with differentiable hybrid action sampling."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[CQLConfig] = None,
    ) -> None:
        self.cql_cfg = cfg or CQLConfig()
        actor = HybridMLPActor(state_dim, num_discrete, d_max, hidden)
        super().__init__(
            state_dim,
            num_discrete,
            d_max,
            actor,
            hidden,
            QGuidedConfig(
                device=self.cql_cfg.device,
                gamma=self.cql_cfg.gamma,
                tau=self.cql_cfg.tau,
                actor_lr=self.cql_cfg.actor_lr,
                critic_lr=self.cql_cfg.critic_lr,
                eta_bc=0.0,
            ),
        )

    def _critic_bellman_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        reward = batch["r"].to(self.device)
        next_state = batch["s2"].to(self.device)
        done = batch["done"].to(self.device).float()
        with torch.no_grad():
            next_action = self.actor_target.sample_train(
                next_state,
                stochastic_continuous=True,
                stochastic_discrete=False,
            )
            tq1, tq2 = self._critic_q_all_discrete(
                self.critic_target,
                next_state,
                next_action["a_cont"],
            )
            next_log_probs = F.log_softmax(next_action["logits"], dim=-1)
            next_soft_value = (
                next_action["a_probs"]
                * (
                    torch.minimum(tq1, tq2)
                    - self.cql_cfg.entropy_alpha
                    * (next_action["logp_cont"] + next_log_probs)
                )
            ).sum(dim=-1, keepdim=True)
            if self.cfg.target_q_clip is not None:
                next_soft_value = next_soft_value.clamp(
                    -self.cfg.target_q_clip, self.cfg.target_q_clip
                )
            target = reward + self.cfg.gamma * (1.0 - done) * (
                next_soft_value
            )
        q1, q2 = self._critic_q(self.critic, state, cont, disc)
        bellman = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        return bellman, {"state": state, "q1": q1, "q2": q2, "target": target}

    def _actor_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        state = batch["s"].to(self.device)
        policy = self.actor.sample_train(
            state,
            stochastic_continuous=True,
            stochastic_discrete=False,
        )
        with frozen(self.critic):
            q1, q2 = self._critic_q_all_discrete(
                self.critic,
                state,
                policy["a_cont"],
            )
            log_probs = F.log_softmax(policy["logits"], dim=-1)
            actor_loss = (
                policy["a_probs"]
                * (
                    self.cql_cfg.entropy_alpha
                    * (policy["logp_cont"] + log_probs)
                    - torch.minimum(q1, q2)
                )
            ).sum(dim=-1).mean()
        return actor_loss, {
            "bc_loss": 0.0,
            "q_guidance_loss": float(actor_loss.detach()),
            "mean_q_pi": float(
                (policy["a_probs"] * torch.minimum(q1, q2))
                .sum(dim=-1)
                .mean()
                .detach()
            ),
        }

    def _random_actions(
        self, count: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = 2.0 * torch.pi * torch.rand(count, 1, device=self.device)
        radius = (
            torch.sqrt(torch.rand(count, 1, device=self.device)) * self.d_max
        )
        action_cont = (
            torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1) * radius
        )
        action_disc = torch.randint(
            self.num_discrete, (count,), device=self.device
        )
        return action_cont, action_disc

    def _cql_regularizer(
        self,
        state: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        samples = self.cql_cfg.cql_num_random
        random_cont, random_disc = self._random_actions(batch_size * samples)
        repeated_state = state[:, None, :].expand(-1, samples, -1).reshape(
            batch_size * samples, -1
        )
        random_onehot = one_hot(random_disc, self.num_discrete)
        random_cont_norm = normalize_displacement(random_cont, self.d_max)
        q1_random = self.critic.q1(
            repeated_state, random_cont_norm, random_onehot
        ).view(batch_size, samples, 1)
        q2_random = self.critic.q2(
            repeated_state, random_cont_norm, random_onehot
        ).view(batch_size, samples, 1)
        with torch.no_grad():
            policy = self.actor.sample_train(
                state,
                stochastic_continuous=True,
                stochastic_discrete=True,
            )
        q1_policy, q2_policy = self._critic_q_onehot(
            self.critic,
            state,
            policy["a_cont"],
            policy["a_onehot"],
        )
        q1_ood = torch.cat([q1_random, q1_policy[:, None, :]], dim=1)
        q2_ood = torch.cat([q2_random, q2_policy[:, None, :]], dim=1)
        cql1 = torch.logsumexp(q1_ood, dim=1).mean() - q1_data.mean()
        cql2 = torch.logsumexp(q2_ood, dim=1).mean() - q2_data.mean()
        return cql1, cql2

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        bellman, values = self._critic_bellman_loss(batch)
        cql1, cql2 = self._cql_regularizer(
            values["state"], values["q1"], values["q2"]
        )
        critic_loss = bellman + self.cql_cfg.cql_alpha * (cql1 + cql2)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.grad_clip_norm
            )
        self.critic_opt.step()

        actor_loss, actor_logs = self._actor_loss(batch)
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
            "bellman_loss": float(bellman.detach()),
            "cql1": float(cql1.detach()),
            "cql2": float(cql2.detach()),
            "actor_loss": float(actor_loss.detach()),
            **actor_logs,
        }


class _HybridAgent(nn.Module):
    def __init__(self, num_discrete: int, d_max: float, device: str):
        super().__init__()
        self.num_discrete = int(num_discrete)
        self.d_max = float(d_max)
        self.device = torch.device(device)

    def _q(
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

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.data.lerp_(parameter.data, tau)


@dataclass
class IQLConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    max_advantage_weight: float = 100.0
    lr: float = 3e-4


class IQLHybridAgent(_HybridAgent):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[IQLConfig] = None,
    ) -> None:
        self.cfg = cfg or IQLConfig()
        super().__init__(num_discrete, d_max, self.cfg.device)
        self.actor = HybridMLPActor(
            state_dim, num_discrete, d_max, hidden
        ).to(self.device)
        self.critic = DoubleQCriticHybrid(
            state_dim, num_discrete, hidden
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.value = ValueNetwork(state_dim, hidden).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def act(self, state_np: np.ndarray, deterministic: bool = True):
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action = self.actor.sample_eval(state)
        return action["a_cont"][0].cpu().numpy(), int(action["a_disc"][0])

    @torch.no_grad()
    def act_batch(self, state_np: np.ndarray, deterministic: bool = True):
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        )
        action = self.actor.sample_eval(state)
        return action["a_cont"].cpu().numpy(), action["a_disc"].cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        reward = batch["r"].to(self.device)
        next_state = batch["s2"].to(self.device)
        done = batch["done"].to(self.device).float()
        disc_onehot = one_hot(disc, self.num_discrete)

        with torch.no_grad():
            q1_target, q2_target = self._q(
                self.critic_target, state, cont, disc_onehot
            )
            q_data = torch.minimum(q1_target, q2_target)
        value_prediction = self.value(state)
        value_loss = expectile_loss(
            q_data - value_prediction, self.cfg.expectile
        ).mean()
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            target = reward + self.cfg.gamma * (1.0 - done) * self.value(next_state)
        q1, q2 = self._q(self.critic, state, cont, disc_onehot)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        with torch.no_grad():
            advantage = q_data - self.value(state)
            weights = torch.exp(
                self.cfg.advantage_temperature * advantage
            ).clamp(max=self.cfg.max_advantage_weight)
        actor_loss = -(
            weights * self.actor.log_prob_data(state, cont, disc)
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self._soft_update(self.critic, self.critic_target, self.cfg.tau)
        return {
            "critic_total": float(critic_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "actor_total": float(actor_loss.detach()),
        }


@dataclass
class BCQConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    phi: float = 0.05
    target_candidates: int = 10
    eval_candidates: int = 100


class BCQHybridAgent(_HybridAgent):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[BCQConfig] = None,
    ) -> None:
        self.cfg = cfg or BCQConfig()
        super().__init__(num_discrete, d_max, self.cfg.device)
        self.vae = HybridVAE(state_dim, num_discrete).to(self.device)
        self.perturbation = HybridPerturbation(
            state_dim, num_discrete, self.cfg.phi
        ).to(self.device)
        self.perturbation_target = copy.deepcopy(self.perturbation).to(self.device)
        self.critic = DoubleQCriticHybrid(
            state_dim, num_discrete, hidden
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for target in (self.perturbation_target, self.critic_target):
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.cfg.lr)
        self.perturb_opt = torch.optim.Adam(
            self.perturbation.parameters(), lr=self.cfg.lr
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    def _decode_behavior(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoded = self.vae.decode(state)
        cont_norm = project_displacement_torch(torch.tanh(decoded[:, :2]), 1.0)
        return cont_norm, decoded[:, 2:]

    def _candidates(
        self,
        state: torch.Tensor,
        count: int,
        target: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        repeated = state[:, None, :].expand(-1, count, -1).reshape(
            batch_size * count, -1
        )
        base_cont, base_logits = self._decode_behavior(repeated)
        module = self.perturbation_target if target else self.perturbation
        output = module(repeated, base_cont, base_logits)
        return repeated, output["a_cont_norm"], output["a_onehot"]

    @torch.no_grad()
    def act(self, state_np: np.ndarray, deterministic: bool = True):
        cont, disc = self.act_batch(np.asarray(state_np)[None, :], deterministic)
        return cont[0], int(disc[0])

    @torch.no_grad()
    def act_batch(self, state_np: np.ndarray, deterministic: bool = True):
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        )
        batch_size = state.shape[0]
        repeated, cont_norm, onehot = self._candidates(
            state, self.cfg.eval_candidates, target=False
        )
        q1, q2 = self.critic(repeated, cont_norm, onehot)
        candidate_q = torch.minimum(q1, q2).reshape(
            batch_size, self.cfg.eval_candidates
        )
        choice = candidate_q.argmax(dim=1)
        flat_choice = (
            torch.arange(batch_size, device=self.device)
            * self.cfg.eval_candidates
            + choice
        )
        return (
            (cont_norm[flat_choice] * self.d_max).cpu().numpy(),
            onehot[flat_choice].argmax(dim=-1).cpu().numpy(),
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        reward = batch["r"].to(self.device)
        next_state = batch["s2"].to(self.device)
        done = batch["done"].to(self.device).float()
        packed = pack_diffusion_action(
            cont, disc, self.d_max, self.num_discrete
        )
        vae_loss = self.vae.loss(state, packed)
        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()

        with torch.no_grad():
            repeated, next_cont, next_onehot = self._candidates(
                next_state, self.cfg.target_candidates, target=True
            )
            tq1, tq2 = self.critic_target(repeated, next_cont, next_onehot)
            candidate_q = torch.minimum(tq1, tq2).view(
                state.shape[0], self.cfg.target_candidates, 1
            )
            max_q = candidate_q.max(dim=1).values
            target_q = reward + self.cfg.gamma * (1.0 - done) * max_q
        data_cont_norm = normalize_displacement(cont, self.d_max)
        data_onehot = one_hot(disc, self.num_discrete)
        q1, q2 = self.critic(state, data_cont_norm, data_onehot)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        base_cont, base_logits = self._decode_behavior(state)
        base_cont, base_logits = base_cont.detach(), base_logits.detach()
        policy = self.perturbation(state, base_cont, base_logits)
        with frozen(self.critic):
            pq1, pq2 = self.critic(
                state, policy["a_cont_norm"], policy["a_onehot"]
            )
            perturb_loss = -torch.minimum(pq1, pq2).mean()
        self.perturb_opt.zero_grad(set_to_none=True)
        perturb_loss.backward()
        self.perturb_opt.step()
        self._soft_update(self.critic, self.critic_target, self.cfg.tau)
        self._soft_update(
            self.perturbation, self.perturbation_target, self.cfg.tau
        )
        return {
            "critic_total": float(critic_loss.detach()),
            "vae_loss": float(vae_loss.detach()),
            "actor_total": float(perturb_loss.detach()),
        }


@dataclass
class ReBRACConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True


class ReBRACHybridAgent(_HybridAgent):
    """ReBRAC with MSE penalties applied to the full hybrid action vector."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[ReBRACConfig] = None,
    ) -> None:
        self.cfg = cfg or ReBRACConfig()
        super().__init__(num_discrete, d_max, self.cfg.device)
        self.actor = HybridMLPActor(
            state_dim, num_discrete, d_max, hidden
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = DoubleQCriticHybrid(
            state_dim, num_discrete, hidden, layer_norm=True
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for target in (self.actor_target, self.critic_target):
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr
        )
        self.update_count = 0

    @torch.no_grad()
    def act(self, state_np: np.ndarray, deterministic: bool = True):
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action = self.actor.sample_eval(state)
        return action["a_cont"][0].cpu().numpy(), int(action["a_disc"][0])

    @torch.no_grad()
    def act_batch(self, state_np: np.ndarray, deterministic: bool = True):
        del deterministic
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        )
        action = self.actor.sample_eval(state)
        return action["a_cont"].cpu().numpy(), action["a_disc"].cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        required = {"next_a_cont", "next_a_disc"}
        if not required.issubset(batch):
            raise KeyError("ReBRAC requires next behavior actions in each batch")
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        reward = batch["r"].to(self.device)
        next_state = batch["s2"].to(self.device)
        done = batch["done"].to(self.device).float()
        next_cont_data = batch["next_a_cont"].to(self.device)
        next_disc_data = batch["next_a_disc"].to(self.device).long()
        with torch.no_grad():
            next_action = self.actor_target.sample_train(next_state)
            noise = torch.randn_like(next_action["a_cont_norm"]) * self.cfg.policy_noise
            noise = noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_cont_norm = project_displacement_torch(
                next_action["a_cont_norm"] + noise, 1.0
            )
            tq1, tq2 = self.critic_target(
                next_state, next_cont_norm, next_action["a_onehot"]
            )
            penalty = hybrid_bc_distance(
                next_cont_norm * self.d_max,
                next_action["a_onehot"],
                next_cont_data,
                next_disc_data,
                self.d_max,
            )
            target = reward + self.cfg.gamma * (1.0 - done) * (
                torch.minimum(tq1, tq2) - self.cfg.critic_bc_coef * penalty
            )
        data_onehot = one_hot(disc, self.num_discrete)
        q1, q2 = self._q(self.critic, state, cont, data_onehot)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = torch.zeros((), device=self.device)
        bc_loss = torch.zeros((), device=self.device)
        if self.update_count % self.cfg.policy_freq == 0:
            policy = self.actor.sample_train(state)
            with frozen(self.critic):
                pq1, pq2 = self._q(
                    self.critic,
                    state,
                    policy["a_cont"],
                    policy["a_onehot"],
                )
                q = torch.minimum(pq1, pq2)
                q_scale = (
                    1.0 / q.abs().mean().detach().clamp_min(1e-6)
                    if self.cfg.normalize_q
                    else 1.0
                )
                bc_loss = hybrid_bc_distance(
                    policy["a_cont"],
                    policy["a_onehot"],
                    cont,
                    disc,
                    self.d_max,
                ).mean()
                actor_loss = -q_scale * q.mean() + self.cfg.actor_bc_coef * bc_loss
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target, self.cfg.tau)
        self._soft_update(self.critic, self.critic_target, self.cfg.tau)
        self.update_count += 1
        return {
            "critic_total": float(critic_loss.detach()),
            "actor_total": float(actor_loss.detach()),
            "bc_loss": float(bc_loss.detach()),
        }


@dataclass
class DTQLConfig:
    device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    lr: float = 3e-4
    trust_weight: float = 1.0
    q_weight: float = 1.0
    direct_bc_weight: float = 1.0
    eval_candidates: int = 64
    normalize_q: bool = True


class DTQLHybridAgent(_HybridAgent):
    """DTQL: behavior diffusion model plus a deployable one-step policy."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        behavior_actor: DiffusionHybrid,
        hidden: Tuple[int, ...] = (256, 256, 256),
        cfg: Optional[DTQLConfig] = None,
    ) -> None:
        self.cfg = cfg or DTQLConfig()
        super().__init__(num_discrete, d_max, self.cfg.device)
        self.behavior_actor = behavior_actor.to(self.device)
        self.actor = HybridMLPActor(
            state_dim, num_discrete, d_max, hidden
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = DoubleQCriticHybrid(
            state_dim, num_discrete, hidden
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.value = ValueNetwork(state_dim, hidden).to(self.device)
        for target in (self.actor_target, self.critic_target):
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self.behavior_opt = torch.optim.Adam(
            self.behavior_actor.parameters(), lr=self.cfg.lr
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def act(self, state_np: np.ndarray, deterministic: bool = True):
        cont, disc = self.act_batch(np.asarray(state_np)[None, :], deterministic)
        return cont[0], int(disc[0])

    @torch.no_grad()
    def act_batch(self, state_np: np.ndarray, deterministic: bool = True):
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        )
        batch_size = state.shape[0]
        repeated = state[:, None, :].expand(
            batch_size, self.cfg.eval_candidates, state.shape[-1]
        ).reshape(batch_size * self.cfg.eval_candidates, -1)
        action = self.actor_target.sample_train(
            repeated,
            stochastic_continuous=True,
            stochastic_discrete=True,
        )
        q1, q2 = self._q(
            self.critic_target,
            repeated,
            action["a_cont"],
            action["a_onehot"],
        )
        candidate_q = torch.minimum(q1, q2).reshape(
            batch_size, self.cfg.eval_candidates
        )
        if deterministic:
            choice = candidate_q.argmax(dim=1)
        else:
            probabilities = F.softmax(candidate_q, dim=1)
            choice = torch.multinomial(probabilities, 1).squeeze(1)
        flat_choice = (
            torch.arange(batch_size, device=self.device)
            * self.cfg.eval_candidates
            + choice
        )
        return (
            action["a_cont"][flat_choice].cpu().numpy(),
            action["a_disc"][flat_choice].cpu().numpy(),
        )

    def _critic_value_update(
        self,
        state: torch.Tensor,
        cont: torch.Tensor,
        disc: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_onehot = one_hot(disc, self.num_discrete)
        with torch.no_grad():
            tq1, tq2 = self._q(
                self.critic_target, state, cont, data_onehot
            )
            q_data = torch.minimum(tq1, tq2)
        value_loss = expectile_loss(
            q_data - self.value(state), self.cfg.expectile
        ).mean()
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()
        with torch.no_grad():
            target = reward + self.cfg.gamma * (1.0 - done) * self.value(next_state)
        q1, q2 = self._q(self.critic, state, cont, data_onehot)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self._soft_update(self.critic, self.critic_target, self.cfg.tau)
        return critic_loss, value_loss

    def pretrain_update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        packed = pack_diffusion_action(
            cont, disc, self.d_max, self.num_discrete
        )
        behavior_loss = self.behavior_actor.loss(packed, state)
        self.behavior_opt.zero_grad(set_to_none=True)
        behavior_loss.backward()
        self.behavior_opt.step()
        critic_loss, value_loss = self._critic_value_update(
            state,
            cont,
            disc,
            batch["r"].to(self.device),
            batch["s2"].to(self.device),
            batch["done"].to(self.device).float(),
        )
        return {
            "behavior_loss": float(behavior_loss.detach()),
            "critic_total": float(critic_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        state = batch["s"].to(self.device)
        cont = batch["a_cont"].to(self.device)
        disc = batch["a_disc"].to(self.device).long()
        packed = pack_diffusion_action(
            cont, disc, self.d_max, self.num_discrete
        )
        behavior_loss = self.behavior_actor.loss(packed, state)
        self.behavior_opt.zero_grad(set_to_none=True)
        behavior_loss.backward()
        self.behavior_opt.step()
        critic_loss, value_loss = self._critic_value_update(
            state,
            cont,
            disc,
            batch["r"].to(self.device),
            batch["s2"].to(self.device),
            batch["done"].to(self.device).float(),
        )

        policy = self.actor.sample_train(state)
        policy_vector = torch.cat(
            [policy["a_cont_norm"], policy["a_onehot"] * 2.0 - 1.0], dim=-1
        )
        # Freeze the score network so this loss moves only the one-step actor.
        with frozen(self.behavior_actor):
            trust_loss = self.behavior_actor.loss(policy_vector, state)
        with frozen(self.critic):
            q1, q2 = self._q(
                self.critic,
                state,
                policy["a_cont"],
                policy["a_onehot"],
            )
            q = torch.minimum(q1, q2)
            q_scale = (
                1.0 / q.abs().mean().detach().clamp_min(1e-6)
                if self.cfg.normalize_q
                else 1.0
            )
            q_loss = -q_scale * q.mean()
        direct_bc = self.actor.bc_loss(state, cont, disc)
        actor_loss = (
            self.cfg.trust_weight * trust_loss
            + self.cfg.q_weight * q_loss
            + self.cfg.direct_bc_weight * direct_bc
        )
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self._soft_update(self.actor, self.actor_target, self.cfg.tau)
        return {
            "behavior_loss": float(behavior_loss.detach()),
            "critic_total": float(critic_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "trust_loss": float(trust_loss.detach()),
            "q_guidance_loss": float(q_loss.detach()),
            "q_guidance_scale": float(
                q_scale.detach() if torch.is_tensor(q_scale) else q_scale
            ),
            "bc_loss": float(direct_bc.detach()),
            "actor_total": float(actor_loss.detach()),
        }
