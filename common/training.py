from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from .experiment import evaluate_across_layouts


def _optimizer_states(agent) -> Dict[str, dict]:
    return {
        name: value.state_dict()
        for name, value in vars(agent).items()
        if isinstance(value, torch.optim.Optimizer)
    }


def _save_json(history: Dict[str, Iterable[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: [float(value) for value in values]
        for name, values in history.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def train_offline(
    agent,
    loader,
    num_devices: int,
    epochs: int,
    eval_every: int,
    eval_layout_seeds: Sequence[int],
    state_normalizer,
    checkpoint_path: Path,
    history_path: Path,
    run_metadata: dict,
    pretrain_epochs: int = 0,
    resume: bool = True,
) -> Dict[str, list]:
    history: Dict[str, list] = defaultdict(list)
    start_epoch = 1
    if resume and checkpoint_path.exists():
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=agent.device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=agent.device)
        if payload.get("run_metadata") != run_metadata:
            raise RuntimeError("Checkpoint configuration does not match this run")
        agent.load_state_dict(payload["agent_state_dict"])
        for name, state in payload.get("optimizer_state_dicts", {}).items():
            optimizer = getattr(agent, name, None)
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(state)
        history.update(
            {name: list(values) for name, values in payload["history"].items()}
        )
        np.random.set_state(payload["numpy_rng_state"])
        torch.random.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_states"]]
            )
        start_epoch = int(payload["epoch"]) + 1
        print(f"resumed_from_epoch={start_epoch - 1}", flush=True)

    if start_epoch == 1 and pretrain_epochs > 0:
        if not hasattr(agent, "pretrain_update"):
            raise ValueError("This agent does not support pretraining")
        for epoch in range(1, pretrain_epochs + 1):
            for batch in loader:
                agent.pretrain_update(batch)
            print(f"pretrain_epoch={epoch}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        totals: Dict[str, float] = defaultdict(float)
        batches = 0
        agent.train()
        for batch in loader:
            logs = agent.update(batch)
            for name, value in logs.items():
                totals[name] += float(value)
            batches += 1
        evaluate_now = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if not evaluate_now:
            continue
        agent.eval()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            metrics = evaluate_across_layouts(
                agent,
                num_devices,
                eval_layout_seeds,
                state_normalizer=state_normalizer,
            )
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        history["epoch"].append(epoch)
        for name, value in metrics.items():
            history[name].append(value)
        for name, value in totals.items():
            history[f"train_{name}"].append(value / max(batches, 1))
        print(
            f"epoch={epoch:03d} return={metrics['return']:.6f} "
            f"aoi_s={metrics['aoi']:.6f} "
            f"energy_kj={metrics['total_energy_j'] / 1000.0:.6f}",
            flush=True,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dicts": _optimizer_states(agent),
                "history": dict(history),
                "run_metadata": run_metadata,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            checkpoint_path,
        )
        _save_json(history, history_path)
    return dict(history)

