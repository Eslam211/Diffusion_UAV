from __future__ import annotations

from baselines.implementations import CQLConfig, CQLHybridAgent


def build_agent(
    state_dim: int,
    num_discrete: int,
    d_max: float,
    device: str,
    cql_alpha: float = 1.0,
    entropy_alpha: float = 0.2,
    cql_num_random: int = 10,
) -> CQLHybridAgent:
    return CQLHybridAgent(
        state_dim,
        num_discrete,
        d_max,
        cfg=CQLConfig(
            device=device,
            cql_alpha=cql_alpha,
            entropy_alpha=entropy_alpha,
            cql_num_random=cql_num_random,
        ),
    )

