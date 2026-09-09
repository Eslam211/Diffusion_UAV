from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import torch

from .data import make_dataloader
from .evaluation import set_seed
from .experiment import evaluate_across_layouts
from .training import train_offline


Builder = Callable[..., torch.nn.Module]


def _add_method_arguments(parser, defaults: Dict[str, object]) -> None:
    for name, default in defaults.items():
        option = "--" + name.replace("_", "-")
        parser.add_argument(option, dest=name, type=type(default), default=default)


def train_entrypoint(
    method_name: str,
    builder: Builder,
    repo_root: Path,
    method_defaults: Dict[str, object],
) -> None:
    parser = argparse.ArgumentParser(description=f"Train {method_name}")
    parser.add_argument(
        "--dataset",
        default=str(repo_root / "data" / "uav_offline_dataset.npz"),
    )
    parser.add_argument(
        "--output-dir", default=str(repo_root / "results" / method_name)
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=30_000)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--quality", choices=("all", "good", "bad"), default="all")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-layouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    _add_method_arguments(parser, method_defaults)
    args = parser.parse_args()
    set_seed(args.seed)
    method_config = {
        name: getattr(args, name) for name in method_defaults
    }
    loader, dataset = make_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        quality=args.quality,
        reward_scale=args.reward_scale,
    )
    state_dim = int(dataset.arrays["s"].shape[1])
    num_devices = (state_dim - 4) // 4
    d_max = float(dataset.metadata.get("d_max_m", 25.0))
    agent = builder(
        state_dim=state_dim,
        num_discrete=num_devices + 1,
        d_max=d_max,
        device=args.device,
        **method_config,
    ).to(args.device)
    output_dir = Path(args.output_dir)
    checkpoint = output_dir / "checkpoint.pt"
    history = output_dir / "history.json"
    run_metadata = {
        "method": method_name,
        "method_config": method_config,
        "dataset_config": {
            "subset_size": args.subset_size,
            "subset_seed": args.subset_seed,
            "quality": args.quality,
            "reward_scale": args.reward_scale,
        },
        "state_dim": state_dim,
        "num_devices": num_devices,
        "d_max": d_max,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "eval_layouts": args.eval_layouts,
        "seed": args.seed,
    }
    train_offline(
        agent,
        loader,
        num_devices=num_devices,
        epochs=args.epochs,
        eval_every=args.eval_every,
        eval_layout_seeds=tuple(range(args.eval_layouts)),
        state_normalizer=dataset.stats.state_norm,
        checkpoint_path=checkpoint,
        history_path=history,
        run_metadata=run_metadata,
        pretrain_epochs=int(method_config.get("pretrain_epochs", 0)),
        resume=not args.no_resume,
    )
    print(f"checkpoint={checkpoint}")
    print(f"history={history}")


def test_entrypoint(
    method_name: str,
    builder: Builder,
    repo_root: Path,
) -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate {method_name}")
    parser.add_argument(
        "--checkpoint",
        default=str(repo_root / "results" / method_name / "checkpoint.pt"),
    )
    parser.add_argument(
        "--dataset",
        default=str(repo_root / "data" / "uav_offline_dataset.npz"),
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
    metadata = payload["run_metadata"]
    if metadata["method"] != method_name:
        raise RuntimeError(
            f"Checkpoint contains {metadata['method']}, expected {method_name}"
        )
    dataset_config = metadata["dataset_config"]
    _, dataset = make_dataloader(
        args.dataset,
        batch_size=1,
        shuffle=False,
        subset_size=dataset_config["subset_size"],
        subset_seed=dataset_config["subset_seed"],
        quality=dataset_config["quality"],
        reward_scale=dataset_config["reward_scale"],
    )
    agent = builder(
        state_dim=metadata["state_dim"],
        num_discrete=metadata["num_devices"] + 1,
        d_max=metadata["d_max"],
        device=args.device,
        **metadata["method_config"],
    ).to(args.device)
    agent.load_state_dict(payload["agent_state_dict"])
    agent.eval()
    metrics = evaluate_across_layouts(
        agent,
        metadata["num_devices"],
        tuple(range(args.layouts)),
        state_normalizer=dataset.stats.state_norm,
    )
    result = {
        "method": method_name,
        "checkpoint_epoch": int(payload["epoch"]),
        "layouts": args.layouts,
        "metrics": metrics,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")

