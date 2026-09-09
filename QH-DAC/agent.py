from __future__ import annotations

from typing import Tuple

from common.builders import build_diffusion_actor
from common.q_guided import QGuidedConfig, QGuidedHybridAgent


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    diffusion_steps: int = 50,
    eta_bc: float = 0.5,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    hidden: Tuple[int, ...] = (256, 256, 256),
) -> QGuidedHybridAgent:
    actor = build_diffusion_actor(
        state_dim,
        num_discrete,
        d_max,
        device,
        diffusion_steps=diffusion_steps,
    )
    return QGuidedHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        actor,
        hidden,
        QGuidedConfig(
            device=device,
            eta_bc=eta_bc,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )

