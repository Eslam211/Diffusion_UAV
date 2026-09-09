from __future__ import annotations

from baselines.implementations import DTQLConfig, DTQLHybridAgent
from common.builders import build_diffusion_actor


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    diffusion_steps: int = 50,
    pretrain_epochs: int = 10,
    expectile: float = 0.7,
    trust_weight: float = 1.0,
    q_weight: float = 1.0,
    direct_bc_weight: float = 1.0,
    lr: float = 3e-4,
) -> DTQLHybridAgent:
    del pretrain_epochs
    behavior_actor = build_diffusion_actor(
        state_dim,
        num_discrete,
        d_max,
        device,
        diffusion_steps=diffusion_steps,
    )
    return DTQLHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        behavior_actor,
        cfg=DTQLConfig(
            device=device,
            expectile=expectile,
            trust_weight=trust_weight,
            q_weight=q_weight,
            direct_bc_weight=direct_bc_weight,
            lr=lr,
        ),
    )

