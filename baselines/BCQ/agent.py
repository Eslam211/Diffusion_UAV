from __future__ import annotations

from baselines.implementations import BCQConfig, BCQHybridAgent


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    phi: float = 0.05,
    target_candidates: int = 10,
    eval_candidates: int = 100,
    lr: float = 3e-4,
) -> BCQHybridAgent:
    return BCQHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        cfg=BCQConfig(
            device=device,
            phi=phi,
            target_candidates=target_candidates,
            eval_candidates=eval_candidates,
            lr=lr,
        ),
    )

