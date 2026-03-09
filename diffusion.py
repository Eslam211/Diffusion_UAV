
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(n_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)

def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas.float(), 1e-8, 0.999)

def vp_beta_schedule(n_timesteps: int) -> torch.Tensor:
    t = torch.linspace(0, 1, n_timesteps, dtype=torch.float32)
    beta_min, beta_max = 0.1, 20.0
    betas = 1.0 - torch.exp(-0.5 * (beta_min + (beta_max - beta_min) * t) / n_timesteps)
    return torch.clip(betas, 1e-8, 0.999)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


@dataclass
class DiffusionConfig:
    n_timesteps: int = 50
    beta_schedule: str = "linear"
    predict_epsilon: bool = True
    clip_denoised: bool = True
    bc_loss: str = "mse"
    ddim: bool = False
    temperature: float = 1.0


class DiffusionHybrid(nn.Module):
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

        # betas
        if self.cfg.beta_schedule == "linear":
            betas = linear_beta_schedule(self.cfg.n_timesteps)
        elif self.cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.cfg.n_timesteps)
        elif self.cfg.beta_schedule == "vp":
            betas = vp_beta_schedule(self.cfg.n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule={self.cfg.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise


    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
               extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps_pred = self.model(x_t, t, state)
        if self.cfg.predict_epsilon:
            x0 = self.predict_x0_from_eps(x_t, t, eps_pred)
        else:
            x0 = eps_pred

        if self.cfg.clip_denoised:
            x0 = torch.clamp(x0, -1.0, 1.0)

        model_mean, var, log_var = self.q_posterior(x_start=x0, x_t=x_t, t=t)
        return model_mean, var, log_var, x0

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = x_t.shape[0]
        model_mean, _, model_log_var, _ = self.p_mean_variance(x_t, t, state)

        if self.cfg.ddim:
            return model_mean

        noise = torch.randn_like(x_t) * float(self.cfg.temperature)
        nonzero_mask = (1 - (t == 0).float()).view(b, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

    @torch.no_grad()
    def sample_action_vector(
        self,
        state: torch.Tensor,
        return_chain: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = self.betas.device
        b = state.shape[0]
        x = torch.randn((b, self.action_dim_total), device=device)
        chain = [x] if return_chain else None

        for i in reversed(range(self.cfg.n_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)
            if return_chain:
                chain.append(x)

        if return_chain:
            return x, torch.stack(chain, dim=1)
        return x, None

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        return_probs: bool = True,
    ) -> Dict[str, torch.Tensor]:

        a_vec, _ = self.sample_action_vector(state, return_chain=False)

        a_cont_norm = torch.clamp(a_vec[:, :2], -1.0, 1.0)
        a_cont = a_cont_norm * self.d_max

        logits_disc = a_vec[:, 2:] 
        probs = F.softmax(logits_disc, dim=-1)
        a_disc = torch.argmax(probs, dim=-1)

        a_onehot = F.one_hot(a_disc, num_classes=self.num_discrete).float()

        out = {
            "a_cont": a_cont,
            "a_disc": a_disc.long(),
            "a_onehot": a_onehot,
            "a_vec": a_vec,
        }
        if return_probs:
            out["a_probs"] = probs
        return out


    def loss(self, a_vec0: torch.Tensor, state: torch.Tensor, weights: float = 1.0) -> torch.Tensor:
        b = a_vec0.shape[0]
        t = torch.randint(0, self.cfg.n_timesteps, (b,), device=a_vec0.device, dtype=torch.long)
        noise = torch.randn_like(a_vec0)
        a_noisy = self.q_sample(a_vec0, t, noise=noise)
        eps_pred = self.model(a_noisy, t, state)

        if self.cfg.bc_loss == "mse":
            loss = F.mse_loss(eps_pred, noise, reduction="none")
            loss = loss.mean(dim=-1)  # (B,)
            return (loss * weights).mean()
        else:
            raise ValueError("Unknown loss.")
