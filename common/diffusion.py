from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import project_displacement_torch, straight_through_softmax


def linear_beta_schedule(
    n_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> torch.Tensor:
    # The canonical endpoints assume 1000 diffusion steps. Without rescaling,
    # 50 steps leave alpha_bar_T around 0.60, so training never reaches the
    # N(0, I) terminal distribution used to start inference.
    scale = 1000.0 / float(n_timesteps)
    return torch.linspace(
        scale * beta_start,
        scale * beta_end,
        n_timesteps,
        dtype=torch.float32,
    ).clamp(max=0.999)


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / n_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return torch.clip((1.0 - alpha_bar[1:] / alpha_bar[:-1]).float(), 1e-8, 0.999)


def vp_beta_schedule(n_timesteps: int) -> torch.Tensor:
    time = torch.linspace(0, 1, n_timesteps, dtype=torch.float32)
    beta_min, beta_max = 0.1, 20.0
    betas = 1.0 - torch.exp(
        -0.5 * (beta_min + (beta_max - beta_min) * time) / n_timesteps
    )
    return torch.clip(betas, 1e-8, 0.999)


def extract(values: torch.Tensor, timestep: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, timestep)
    return gathered.view(timestep.shape[0], *([1] * (len(shape) - 1)))


@dataclass
class DiffusionConfig:
    n_timesteps: int = 50
    beta_schedule: str = "linear"
    predict_epsilon: bool = True
    clip_denoised: bool = True
    bc_loss: str = "mse"
    ddim: bool = False
    temperature: float = 1.0
    discrete_temperature: float = 1.0


class DiffusionHybrid(nn.Module):
    """DDPM actor over a shared continuous-plus-categorical action vector."""

    reverse_variance_name = "beta_tilde"

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        model: nn.Module,
        d_max: float,
        cfg: Optional[DiffusionConfig] = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_discrete = int(num_discrete)
        self.d_max = float(d_max)
        self.action_dim_total = 2 + self.num_discrete
        self.model = model
        self.cfg = cfg or DiffusionConfig()

        if self.cfg.beta_schedule == "linear":
            betas = linear_beta_schedule(self.cfg.n_timesteps)
        elif self.cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.cfg.n_timesteps)
        elif self.cfg.beta_schedule == "vp":
            betas = vp_beta_schedule(self.cfg.n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule={self.cfg.beta_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alpha_bar)
        self.register_buffer("alphas_cumprod_prev", alpha_bar_previous)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_bar)
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_bar - 1.0)
        )
        # This is beta_tilde from the DDPM posterior, not beta_t.
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_bar_previous) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bar_previous) * torch.sqrt(alphas) / (1.0 - alpha_bar),
        )

    def q_sample(
        self,
        action_start: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_start) if noise is None else noise
        return extract(self.sqrt_alphas_cumprod, timestep, action_start.shape) * action_start + extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, action_start.shape
        ) * noise

    def predict_x0_from_eps(
        self, action_t: torch.Tensor, timestep: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, timestep, action_t.shape) * action_t - extract(
            self.sqrt_recipm1_alphas_cumprod, timestep, action_t.shape
        ) * epsilon

    def q_posterior(
        self,
        action_start: torch.Tensor,
        action_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.posterior_mean_coef1, timestep, action_t.shape) * action_start + extract(
            self.posterior_mean_coef2, timestep, action_t.shape
        ) * action_t
        variance = extract(self.posterior_variance, timestep, action_t.shape)
        log_variance = extract(
            self.posterior_log_variance_clipped, timestep, action_t.shape
        )
        return mean, variance, log_variance

    def p_mean_variance(
        self, action_t: torch.Tensor, timestep: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = self.model(action_t, timestep, state)
        action_start = (
            self.predict_x0_from_eps(action_t, timestep, model_output)
            if self.cfg.predict_epsilon
            else model_output
        )
        if self.cfg.clip_denoised:
            action_start = action_start.clamp(-1.0, 1.0)
        mean, variance, log_variance = self.q_posterior(
            action_start, action_t, timestep
        )
        return mean, variance, log_variance, action_start, model_output

    def p_sample(
        self, action_t: torch.Tensor, timestep: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        batch_size = action_t.shape[0]
        mean, _, log_variance, action_start, model_output = self.p_mean_variance(
            action_t, timestep, state
        )
        nonzero = (timestep != 0).float().view(
            batch_size, *([1] * (action_t.ndim - 1))
        )
        if self.cfg.ddim:
            alpha_previous = extract(
                self.alphas_cumprod_prev, timestep, action_t.shape
            )
            epsilon = (
                model_output
                if self.cfg.predict_epsilon
                else (
                    action_t
                    - extract(self.sqrt_alphas_cumprod, timestep, action_t.shape)
                    * action_start
                )
                / extract(
                    self.sqrt_one_minus_alphas_cumprod, timestep, action_t.shape
                ).clamp_min(1e-8)
            )
            deterministic = torch.sqrt(alpha_previous) * action_start + torch.sqrt(
                1.0 - alpha_previous
            ) * epsilon
            return nonzero * deterministic + (1.0 - nonzero) * action_start
        noise = torch.randn_like(action_t) * float(self.cfg.temperature)
        return mean + nonzero * torch.exp(0.5 * log_variance) * noise

    def sample_action_vector_train(self, state: torch.Tensor) -> torch.Tensor:
        """Differentiable reverse chain used only for actor Q-guidance."""
        action = torch.randn(
            state.shape[0], self.action_dim_total, device=state.device
        )
        for index in reversed(range(self.cfg.n_timesteps)):
            timestep = torch.full(
                (state.shape[0],), index, device=state.device, dtype=torch.long
            )
            action = self.p_sample(action, timestep, state)
        return action

    @torch.no_grad()
    def sample_action_vector(self, state: torch.Tensor) -> torch.Tensor:
        return self.sample_action_vector_train(state)

    def _decode(
        self, action_vector: torch.Tensor, straight_through: bool
    ) -> Dict[str, torch.Tensor]:
        cont_norm = project_displacement_torch(action_vector[:, :2], 1.0)
        logits = action_vector[:, 2:]
        if straight_through:
            onehot, disc, probs = straight_through_softmax(
                logits, self.cfg.discrete_temperature
            )
        else:
            probs = F.softmax(logits / self.cfg.discrete_temperature, dim=-1)
            disc = probs.argmax(dim=-1)
            onehot = F.one_hot(disc, self.num_discrete).to(action_vector.dtype)
        return {
            "a_cont": cont_norm * self.d_max,
            "a_cont_norm": cont_norm,
            "a_disc": disc.long(),
            "a_onehot": onehot,
            "a_probs": probs,
            "a_vec": action_vector,
        }

    def sample_train(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._decode(self.sample_action_vector_train(state), straight_through=True)

    @torch.no_grad()
    def sample(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._decode(self.sample_action_vector(state), straight_through=False)

    def loss(
        self, action_vector_start: torch.Tensor, state: torch.Tensor, weights: float = 1.0
    ) -> torch.Tensor:
        if self.cfg.bc_loss != "mse":
            raise ValueError(f"Unknown BC loss: {self.cfg.bc_loss}")
        batch_size = action_vector_start.shape[0]
        timestep = torch.randint(
            0,
            self.cfg.n_timesteps,
            (batch_size,),
            device=action_vector_start.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(action_vector_start)
        noisy_action = self.q_sample(action_vector_start, timestep, noise)
        prediction = self.model(noisy_action, timestep, state)
        target = noise if self.cfg.predict_epsilon else action_vector_start
        per_sample = F.mse_loss(prediction, target, reduction="none").mean(-1)
        return (per_sample * weights).mean()

