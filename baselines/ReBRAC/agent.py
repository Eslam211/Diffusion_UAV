from __future__ import annotations

from baselines.implementations import ReBRACConfig, ReBRACHybridAgent


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    actor_bc_coef: float = 1.0,
    critic_bc_coef: float = 1.0,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
) -> ReBRACHybridAgent:
    return ReBRACHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        cfg=ReBRACConfig(
            device=device,
            actor_bc_coef=actor_bc_coef,
            critic_bc_coef=critic_bc_coef,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        ),
    )

