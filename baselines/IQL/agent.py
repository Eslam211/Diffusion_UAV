from __future__ import annotations

from baselines.implementations import IQLConfig, IQLHybridAgent


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    expectile: float = 0.7,
    advantage_temperature: float = 3.0,
    max_advantage_weight: float = 100.0,
    lr: float = 3e-4,
) -> IQLHybridAgent:
    return IQLHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        cfg=IQLConfig(
            device=device,
            expectile=expectile,
            advantage_temperature=advantage_temperature,
            max_advantage_weight=max_advantage_weight,
            lr=lr,
        ),
    )

