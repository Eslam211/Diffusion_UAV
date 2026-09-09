from __future__ import annotations

from common.networks import HybridMLPActor
from common.q_guided import QGuidedConfig, QGuidedHybridAgent


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    eta_bc: float = 0.5,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-4,
) -> QGuidedHybridAgent:
    actor = HybridMLPActor(state_dim, num_discrete, d_max)
    return QGuidedHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        actor,
        cfg=QGuidedConfig(
            device=device,
            eta_bc=eta_bc,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )

