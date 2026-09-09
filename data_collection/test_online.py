from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiment import build_env, default_env_config, evaluate_across_layouts
from data_collection.online_agent import HybridSACAgent, SACConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the online SAC policy")
    parser.add_argument(
        "--checkpoint", default=str(REPO_ROOT / "results" / "online_sac.pt")
    )
    parser.add_argument("--layouts", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    try:
        payload = torch.load(
            args.checkpoint, map_location=args.device, weights_only=False
        )
    except TypeError:
        payload = torch.load(args.checkpoint, map_location=args.device)
    metadata = payload["metadata"]
    num_devices = int(metadata["K"])
    cfg = default_env_config()
    reference_env = build_env(
        num_devices, int(metadata["training_layout_seeds"][0]), cfg
    )
    agent = HybridSACAgent(
        reference_env.observation_dim,
        num_devices + 1,
        cfg.d_max,
        cfg=SACConfig(
            device=args.device,
            sample_discrete_at_evaluation=True,
        ),
    ).to(args.device)
    agent.load_state_dict(payload["agent_state_dict"])
    agent.eval()

    class PhysicalNormalizer:
        def normalize(self, values):
            return reference_env.normalize_observation(values)

    metrics = evaluate_across_layouts(
        agent,
        num_devices,
        tuple(range(args.layouts)),
        state_normalizer=PhysicalNormalizer(),
        cfg=cfg,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "layouts": args.layouts,
        "metrics": metrics,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

