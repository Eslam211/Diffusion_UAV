"""Hybrid continuous-discrete action utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def project_displacement_np(action: np.ndarray, d_max: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    norm = np.linalg.norm(action, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(d_max) / np.maximum(norm, 1e-12))
    return action * scale


def project_displacement_torch(action: torch.Tensor, d_max: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
    scale = torch.clamp(float(d_max) / torch.clamp(norm, min=1e-12), max=1.0)
    return action * scale


def normalize_displacement(action_m: torch.Tensor, d_max: float) -> torch.Tensor:
    normalized = action_m / (float(d_max) + 1e-8)
    return project_displacement_torch(normalized, 1.0)


def straight_through_softmax(
    logits: torch.Tensor, temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard one-hot forward pass with softmax gradients in the backward pass."""
    probs = F.softmax(logits / float(temperature), dim=-1)
    indices = probs.argmax(dim=-1)
    hard = F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)
    onehot_st = hard + probs - probs.detach()
    return onehot_st, indices, probs


def one_hot(indices: torch.Tensor, num_discrete: int) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=int(num_discrete)).float()


def hybrid_bc_distance(
    predicted_cont_m: torch.Tensor,
    predicted_onehot: torch.Tensor,
    data_cont_m: torch.Tensor,
    data_disc: torch.Tensor,
    d_max: float,
) -> torch.Tensor:
    """Per-sample MSE on the common normalized continuous/one-hot vector."""
    pred_cont = normalize_displacement(predicted_cont_m, d_max)
    data_cont = normalize_displacement(data_cont_m, d_max)
    data_onehot = one_hot(data_disc, predicted_onehot.shape[-1])
    return (pred_cont - data_cont).pow(2).sum(-1, keepdim=True) + (
        predicted_onehot - data_onehot
    ).pow(2).sum(-1, keepdim=True)


def pack_diffusion_action(
    action_cont_m: torch.Tensor,
    action_disc: torch.Tensor,
    d_max: float,
    num_discrete: int,
) -> torch.Tensor:
    cont = normalize_displacement(action_cont_m, d_max)
    disc_pm = one_hot(action_disc, num_discrete) * 2.0 - 1.0
    return torch.cat([cont, disc_pm], dim=-1)


