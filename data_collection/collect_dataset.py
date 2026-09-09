from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.pipeline import train_online_and_collect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data" / "uav_offline_dataset.npz"))
    parser.add_argument(
        "--history", default=str(ROOT / "results" / "online_history.json")
    )
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "online_sac.pt"))
    parser.add_argument(
        "--resume", default=str(ROOT / "results" / "online_training_state.pt")
    )
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--random-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--online-dataset-fraction", type=float, default=0.5)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    history = train_online_and_collect(
        output_dataset=args.output,
        checkpoint_path=args.checkpoint,
        history_path=args.history,
        resume_path=args.resume,
        seed=args.seed,
        episodes=args.episodes,
        capacity=args.capacity,
        random_steps=args.random_steps,
        batch_size=args.batch_size,
        reward_scale=args.reward_scale,
        online_dataset_fraction=args.online_dataset_fraction,
        validation_every=args.validation_every,
        device=args.device,
    )
    print(f"saved_dataset={args.output}")
    print(f"saved_history={args.history}")
    print(f"saved_checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()

