from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import (
    normalize_displacement,
    project_displacement_torch,
    straight_through_softmax,
)


EPS = 1e-6
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def mlp(
    in_dim: int,
    hidden: Tuple[int, ...],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers = []
    previous = in_dim
    for width in hidden:
        layers.append(nn.Linear(previous, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        previous = width
    layers.append(nn.Linear(previous, out_dim))
    return nn.Sequential(*layers)


def feature_mlp(
    in_dim: int,
    hidden: Tuple[int, ...],
    activation: type[nn.Module] = nn.ReLU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers = []
    previous = in_dim
    for width in hidden:
        layers.append(nn.Linear(previous, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        previous = width
    return nn.Sequential(*layers)


def split_augmented_uav_state(
    state: torch.Tensor, num_devices: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return global and per-device features from the UAV observation.

    The observation is grouped by modality. The device tensor interleaves ``[relative_x, relative_y, AoI, SNR]``.  Shared processing of
    this tensor makes scheduling equivariant to arbitrary device numbering.
    """
    expected = 4 * int(num_devices) + 4
    if state.shape[-1] != expected:
        raise ValueError(
            f"Expected state dimension {expected} for {num_devices} devices, "
            f"got {state.shape[-1]}"
        )
    k = int(num_devices)
    leading = state.shape[:-1]
    relative = state[..., 2 : 2 + 2 * k].reshape(*leading, k, 2)
    aoi = state[..., 2 + 2 * k : 2 + 3 * k].unsqueeze(-1)
    snr = state[..., 2 + 3 * k : 2 + 4 * k].unsqueeze(-1)
    device = torch.cat([relative, aoi, snr], dim=-1)
    global_features = torch.cat([state[..., :2], state[..., -2:]], dim=-1)
    return global_features, device


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4:
            raise ValueError("time embedding dimension must be at least 4")
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float()
        half = self.dim // 2
        scale = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device) * -scale
        )
        arguments = timestep[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class HybridDenoiser(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden_dim: int = 256,
        time_dim: int = 32,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        act = nn.Mish if activation.lower() == "mish" else nn.ReLU
        self.action_dim_total = 2 + int(num_discrete)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            act(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.action_dim_total + time_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, self.action_dim_total),
        )
        self.apply(init_weights)

    def forward(
        self, noisy_action: torch.Tensor, timestep: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(timestep)
        return self.net(torch.cat([noisy_action, time_embedding, state], dim=-1))


class QNetworkHybrid(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.Mish
        self.net = mlp(
            state_dim + 2 + num_discrete,
            hidden,
            1,
            activation=act,
            layer_norm=layer_norm,
        )
        self.apply(init_weights)

    def forward(
        self,
        state: torch.Tensor,
        action_cont_norm: torch.Tensor,
        action_disc_onehot: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([state, action_cont_norm, action_disc_onehot], dim=-1))


class DoubleQCriticHybrid(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.q1 = QNetworkHybrid(
            state_dim, num_discrete, hidden, activation, layer_norm
        )
        self.q2 = QNetworkHybrid(
            state_dim, num_discrete, hidden, activation, layer_norm
        )

    def forward(
        self,
        state: torch.Tensor,
        action_cont_norm: torch.Tensor,
        action_disc_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(state, action_cont_norm, action_disc_onehot),
            self.q2(state, action_cont_norm, action_disc_onehot),
        )


class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.net = mlp(state_dim, hidden, 1, nn.ReLU, layer_norm)
        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class HybridMLPActor(nn.Module):
    """Matched one-step actor for every non-diffusion hybrid baseline."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_discrete = int(num_discrete)
        self.d_max = float(d_max)
        self.trunk = feature_mlp(
            state_dim, hidden, nn.ReLU, layer_norm=layer_norm
        )
        self.mu = nn.Linear(hidden[-1], 2)
        self.log_std = nn.Linear(hidden[-1], 2)
        self.logits = nn.Linear(hidden[-1], self.num_discrete)
        self.apply(init_weights)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(state)
        return {
            "mu": self.mu(hidden),
            "log_std": self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX),
            "logits": self.logits(hidden),
        }

    def sample_train(
        self,
        state: torch.Tensor,
        stochastic_continuous: bool = False,
        stochastic_discrete: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        output = self.forward(state)
        normal = torch.distributions.Normal(
            output["mu"], output["log_std"].exp()
        )
        if stochastic_continuous:
            pre_tanh = normal.rsample()
        else:
            pre_tanh = output["mu"]
        cont_before_projection = torch.tanh(pre_tanh)
        cont_norm = project_displacement_torch(cont_before_projection, 1.0)
        logp_cont = normal.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        logp_cont -= torch.log(
            1.0 - cont_before_projection.pow(2) + EPS
        ).sum(dim=-1, keepdim=True)
        if stochastic_discrete:
            disc_st = F.gumbel_softmax(
                output["logits"], tau=temperature, hard=True, dim=-1
            )
            disc = disc_st.argmax(dim=-1)
            probs = F.softmax(output["logits"] / temperature, dim=-1)
        else:
            disc_st, disc, probs = straight_through_softmax(
                output["logits"], temperature
            )
        logp_disc = (
            disc_st * F.log_softmax(output["logits"], dim=-1)
        ).sum(dim=-1, keepdim=True)
        return {
            **output,
            "a_cont": cont_norm * self.d_max,
            "a_cont_norm": cont_norm,
            "a_disc": disc,
            "a_onehot": disc_st,
            "a_probs": probs,
            "logp_cont": logp_cont,
            "logp_disc": logp_disc,
            "logp": logp_cont + logp_disc,
        }

    @torch.no_grad()
    def sample(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.sample_train(state, stochastic_continuous=False)

    @torch.no_grad()
    def sample_eval(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Deploy the mean displacement and sample the categorical schedule.

        A categorical SAC/BC policy represents a distribution.  Replacing that
        distribution by argmax can collapse a near-balanced scheduler to one
        device (or idle), even though the learned stochastic policy is sound.
        Continuous exploration noise is still disabled at evaluation.
        """
        return self.sample_train(
            state,
            stochastic_continuous=False,
            stochastic_discrete=True,
        )

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.sample(state)
        return output["a_cont"], output["a_disc"]

    def bc_loss(
        self,
        state: torch.Tensor,
        action_cont_m: torch.Tensor,
        action_disc: torch.Tensor,
    ) -> torch.Tensor:
        output = self.forward(state)
        target_cont = normalize_displacement(action_cont_m, self.d_max)
        predicted_cont = project_displacement_torch(torch.tanh(output["mu"]), 1.0)
        cont_loss = F.mse_loss(predicted_cont, target_cont)
        disc_loss = F.cross_entropy(output["logits"], action_disc.long())
        return cont_loss + disc_loss

    def log_prob_data(
        self,
        state: torch.Tensor,
        action_cont_m: torch.Tensor,
        action_disc: torch.Tensor,
    ) -> torch.Tensor:
        output = self.forward(state)
        normalized = normalize_displacement(action_cont_m, self.d_max).clamp(
            -1.0 + EPS, 1.0 - EPS
        )
        pre_tanh = 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))
        normal = torch.distributions.Normal(output["mu"], output["log_std"].exp())
        logp_cont = normal.log_prob(pre_tanh).sum(-1, keepdim=True)
        logp_cont -= torch.log(1.0 - normalized.pow(2) + EPS).sum(-1, keepdim=True)
        logp_disc = F.log_softmax(output["logits"], dim=-1).gather(
            1, action_disc.long().view(-1, 1)
        )
        return logp_cont + logp_disc

    def entropy_proxy(self, state: torch.Tensor) -> torch.Tensor:
        output = self.forward(state)
        normal_entropy = torch.distributions.Normal(
            output["mu"], output["log_std"].exp()
        ).entropy().sum(-1, keepdim=True)
        probs = F.softmax(output["logits"], dim=-1)
        categorical_entropy = -(
            probs * F.log_softmax(output["logits"], dim=-1)
        ).sum(-1, keepdim=True)
        return normal_entropy + categorical_entropy


class HybridSetActor(HybridMLPActor):
    """Permutation-equivariant online actor for augmented UAV observations.

    Every device is scored by the same network using its relative coordinates,
    AoI, and observed SNR. The idle logit and continuous movement distribution
    use permutation-invariant pooled context.
    """

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        d_max: float,
        hidden: Tuple[int, ...] = (256, 256, 256),
    ) -> None:
        nn.Module.__init__(self)
        self.num_discrete = int(num_discrete)
        self.num_devices = self.num_discrete - 1
        self.d_max = float(d_max)
        if self.num_devices <= 0 or state_dim != 4 * self.num_devices + 4:
            raise ValueError("HybridSetActor requires state_dim=4*(N-1)+4")
        device_width = max(32, hidden[0] // 2)
        self.device_encoder = feature_mlp(
            4, (device_width, device_width), nn.ReLU, layer_norm=True
        )
        self.context_encoder = feature_mlp(
            4 + 2 * device_width, hidden, nn.ReLU, layer_norm=True
        )
        self.device_score = mlp(
            device_width + hidden[-1], (hidden[-1],), 1, nn.ReLU
        )
        self.idle_score = mlp(hidden[-1], (hidden[-1],), 1, nn.ReLU)
        self.mu = nn.Linear(hidden[-1], 2)
        self.log_std = nn.Linear(hidden[-1], 2)
        self.apply(init_weights)
        # Begin with hover and uniform scheduling. The random replay warm-up
        # supplies exploration before the actor is updated.
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -1.0)
        nn.init.zeros_(self.device_score[-1].weight)
        nn.init.zeros_(self.device_score[-1].bias)
        nn.init.zeros_(self.idle_score[-1].weight)
        nn.init.zeros_(self.idle_score[-1].bias)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_features, device_features = split_augmented_uav_state(
            state, self.num_devices
        )
        device_embedding = self.device_encoder(device_features)
        pooled = torch.cat(
            [device_embedding.mean(dim=-2), device_embedding.amax(dim=-2)],
            dim=-1,
        )
        context = self.context_encoder(
            torch.cat([global_features, pooled], dim=-1)
        )
        repeated_context = context.unsqueeze(-2).expand(
            -1, self.num_devices, -1
        )
        device_logits = self.device_score(
            torch.cat([device_embedding, repeated_context], dim=-1)
        ).squeeze(-1)
        idle_logit = self.idle_score(context)
        return {
            "mu": self.mu(context),
            "log_std": self.log_std(context).clamp(LOG_STD_MIN, LOG_STD_MAX),
            "logits": torch.cat([idle_logit, device_logits], dim=-1),
        }


class QNetworkSetHybrid(nn.Module):
    """Permutation-invariant Q-network with an equivariant selected-device tag."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        self.num_discrete = int(num_discrete)
        self.num_devices = self.num_discrete - 1
        if self.num_devices <= 0 or state_dim != 4 * self.num_devices + 4:
            raise ValueError("QNetworkSetHybrid requires state_dim=4*(N-1)+4")
        device_width = max(32, hidden[0] // 2)
        self.device_encoder = feature_mlp(
            5, (device_width, device_width), nn.ReLU, layer_norm=True
        )
        self.net = mlp(
            4 + 2 + 1 + 2 * device_width,
            hidden,
            1,
            nn.ReLU,
            layer_norm=True,
        )
        self.apply(init_weights)

    def forward(
        self,
        state: torch.Tensor,
        action_cont_norm: torch.Tensor,
        action_disc_onehot: torch.Tensor,
    ) -> torch.Tensor:
        global_features, device_features = split_augmented_uav_state(
            state, self.num_devices
        )
        scheduled = action_disc_onehot[..., 1:].unsqueeze(-1)
        device_embedding = self.device_encoder(
            torch.cat([device_features, scheduled], dim=-1)
        )
        pooled = torch.cat(
            [device_embedding.mean(dim=-2), device_embedding.amax(dim=-2)],
            dim=-1,
        )
        idle = action_disc_onehot[..., :1]
        return self.net(
            torch.cat(
                [global_features, action_cont_norm, idle, pooled], dim=-1
            )
        )


class DoubleQSetCriticHybrid(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        self.q1 = QNetworkSetHybrid(state_dim, num_discrete, hidden)
        self.q2 = QNetworkSetHybrid(state_dim, num_discrete, hidden)

    def forward(
        self,
        state: torch.Tensor,
        action_cont_norm: torch.Tensor,
        action_disc_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(state, action_cont_norm, action_disc_onehot),
            self.q2(state, action_cont_norm, action_disc_onehot),
        )


class HybridVAE(nn.Module):
    """BCQ behavior model over the common continuous-plus-one-hot vector."""

    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        latent_dim: int = 32,
        hidden: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.action_dim = 2 + int(num_discrete)
        self.latent_dim = int(latent_dim)
        self.encoder = mlp(state_dim + self.action_dim, hidden, 2 * latent_dim)
        self.decoder = mlp(state_dim + latent_dim, hidden, self.action_dim)
        self.apply(init_weights)

    def encode(
        self, state: torch.Tensor, action_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(torch.cat([state, action_vector], dim=-1))
        mean, log_std = moments.chunk(2, dim=-1)
        return mean, log_std.clamp(-4.0, 4.0)

    def decode(
        self, state: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if latent is None:
            latent = torch.randn(
                state.shape[0], self.latent_dim, device=state.device
            ).clamp(-0.5, 0.5)
        return self.decoder(torch.cat([state, latent], dim=-1))

    def loss(self, state: torch.Tensor, action_vector: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.encode(state, action_vector)
        latent = mean + log_std.exp() * torch.randn_like(mean)
        reconstruction = self.decode(state, latent)
        recon_loss = F.mse_loss(reconstruction, action_vector)
        kl_loss = -0.5 * (
            1.0 + 2.0 * log_std - mean.pow(2) - (2.0 * log_std).exp()
        ).mean()
        return recon_loss + 0.5 * kl_loss


class HybridPerturbation(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        phi: float = 0.05,
        hidden: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.num_discrete = int(num_discrete)
        self.phi = float(phi)
        self.net = mlp(
            state_dim + 2 + self.num_discrete,
            hidden,
            2 + self.num_discrete,
        )
        self.apply(init_weights)

    def forward(
        self,
        state: torch.Tensor,
        base_cont_norm: torch.Tensor,
        base_disc_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        change = self.net(
            torch.cat([state, base_cont_norm, base_disc_logits], dim=-1)
        )
        cont = project_displacement_torch(
            base_cont_norm + self.phi * torch.tanh(change[:, :2]), 1.0
        )
        logits = base_disc_logits + self.phi * change[:, 2:]
        disc_st, disc, probs = straight_through_softmax(logits)
        return {
            "a_cont_norm": cont,
            "a_disc": disc,
            "a_onehot": disc_st,
            "a_probs": probs,
        }
