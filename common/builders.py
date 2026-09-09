from __future__ import annotations

from .diffusion import DiffusionConfig, DiffusionHybrid
from .networks import HybridDenoiser


def build_diffusion_actor(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    diffusion_steps: int = 50,
    beta_schedule: str = "linear",
) -> DiffusionHybrid:
    denoiser = HybridDenoiser(
        state_dim,
        num_discrete,
        hidden_dim=256,
        time_dim=32,
        activation="mish",
    )
    actor = DiffusionHybrid(
        state_dim,
        num_discrete,
        denoiser,
        d_max,
        DiffusionConfig(
            n_timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
            predict_epsilon=True,
            clip_denoised=True,
            bc_loss="mse",
            ddim=False,
            temperature=1.0,
        ),
    )
    return actor.to(device)

