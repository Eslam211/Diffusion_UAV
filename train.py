import argparse
import numpy as np
import torch

from env import UAVEnvConfig, UAVOfflineRLEnv
from networks import HybridDenoiser
from diffusion import DiffusionHybrid, DiffusionConfig
from agent import DiffusionCQLHybridAgent, OfflineAgentConfig
from utils import make_dataloader, evaluate_policy, set_seed


def build_eval_env(state_dim: int, seed: int = 0) -> UAVOfflineRLEnv:
    K = int(state_dim - 3)
    rng = np.random.RandomState(seed)
    dev_xy = np.random.uniform(0, 1000, size=(K, 2)).astype(np.float32)
    env_cfg = UAVEnvConfig(
        area_size=1000.0,
        h=100.0,
        B=1e6,
        fc=2e9,
        Dk_bits=2e6,
        lam=0.5,
        H_max=400,
        T_th=400.0,
        d_max=25.0,
        fading="rayleigh",
    )
    return UAVOfflineRLEnv(dev_xy, env_cfg)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument("--dataset", type=str, default="uav_offline_dataset.npz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--diff_steps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--eta_bc", type=float, default=0.5)
    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--cql_num_random", type=int, default=10)

    args = parser.parse_args()
    set_seed(args.seed)

    # Data
    dl, ds = make_dataloader(
        npz_path=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
        shuffle=True,
        num_workers=0,
        normalize=True,
        normalize_reward=True,
        stats=None,
    )
    state_dim = ds.s.shape[1]
    K = state_dim - 3
    num_discrete = K + 1
    d_max = 25.0

    print(f"[DATA] N={len(ds)}  state_dim={state_dim}  K={K}  num_discrete={num_discrete}  device={args.device}")

    # Models
    denoiser = HybridDenoiser(
        state_dim=state_dim,
        num_discrete=num_discrete,
        hidden_dim=256,
        time_dim=32,
        activation="mish",
    )

    diff_cfg = DiffusionConfig(
        n_timesteps=args.diff_steps,
        beta_schedule=args.beta_schedule,
        predict_epsilon=True,
        clip_denoised=True,
        bc_loss="mse",
        ddim=False,
        temperature=1.0,
    )

    actor = DiffusionHybrid(
        state_dim=state_dim,
        num_discrete=num_discrete,
        model=denoiser,
        d_max=d_max,
        cfg=diff_cfg,
    ).to(args.device)

    agent_cfg = OfflineAgentConfig(
        device=args.device,
        gamma=0.99,
        tau=0.005,
        critic_lr=1e-4,
        actor_lr=1e-4,
        eta_bc=args.eta_bc,
        cql_alpha=args.cql_alpha,
        cql_num_random=args.cql_num_random,
        cql_temp=1.0,
        grad_clip_norm=1.0,
    )

    agent = DiffusionCQLHybridAgent(
        state_dim=state_dim,
        num_discrete=num_discrete,
        d_max=d_max,
        actor=actor,
        critic_hidden=(256, 256, 256),
        cfg=agent_cfg,
    ).to(args.device)

    eval_env = build_eval_env(state_dim, seed=args.seed)

    for epoch in range(1, args.epochs + 1):
        logs_acc = {
            "critic_total": 0.0,
            "critic_bellman": 0.0,
            "cql1": 0.0,
            "cql2": 0.0,
            "actor_total": 0.0,
            "bc_loss": 0.0,
            "q_guidance_loss": 0.0,
            "mean_q_pi": 0.0,
        }
        n_batches = 0

        agent.train()
        for batch in dl:
            logs = agent.update(batch)
            for k in logs_acc:
                logs_acc[k] += float(logs.get(k, 0.0))
            n_batches += 1

        for k in logs_acc:
            logs_acc[k] /= max(n_batches, 1)

        print(
            f"Epoch {epoch:04d} | "
            f"CriticTot {logs_acc['critic_total']:.4f} (Bell {logs_acc['critic_bellman']:.4f} "
            f"+ CQL {logs_acc['cql1']:.4f}/{logs_acc['cql2']:.4f}) | "
            f"ActorTot {logs_acc['actor_total']:.4f} (BC {logs_acc['bc_loss']:.4f} "
            f"+ QG {logs_acc['q_guidance_loss']:.4f}) | "
            f"Qpi {logs_acc['mean_q_pi']:.4f}"
        )

        if (epoch % args.eval_every) == 0 or epoch == 1:
            agent.eval()
            ev = evaluate_policy(
                env=eval_env,
                agent=agent,
                n_episodes=args.eval_episodes,
                deterministic=True,
                state_normalizer=ds.stats.state_norm if ds.stats is not None else None,
                seed=args.seed + 1000,
            )
            print(
                f"  [EVAL] return {ev['return']:.3f} | AoI {ev['aoi']:.3f} | "
                f"Energy {ev['energy']:.3f} | ep_len {ev['ep_len']:.1f}"
            )

    print("Training finished.")




