"""UAV data-collection environment used by online and offline experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _project_displacement(action: np.ndarray, d_max: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    norm = float(np.linalg.norm(action))
    return action if norm <= d_max else action * (float(d_max) / max(norm, 1e-12))


@dataclass
class UAVEnvConfig:
    area_size: float = 1000.0
    h: float = 100.0
    T_th: float = 400.0
    # Optional guard only; the physical mission clock is the primary terminal.
    H_max: Optional[int] = None

    # Channel
    fc: float = 2e9
    c: float = 3e8
    B: float = 1e6
    N0: float = 10 ** ((-174.0 - 30.0) / 10.0)
    xi_los_db: float = 1.0
    xi_nlos_db: float = 20.0
    C: float = 10.0
    D: float = 0.6
    fading: str = "rayleigh"

    # Device and UAV
    P_dev: float = 0.1259
    Dk_bits: float = 2e6
    v_u: float = 25.0
    m_tot: float = 0.5
    g: float = 9.807
    r_p: float = 0.2
    n_p: int = 4
    rho: float = 1.225
    P_com: float = 0.0126
    P_idle: float = 0.0
    P_max: float = 5.0
    v_max: float = 25.0
    d_max: float = 25.0

    # Objective
    lam: float = 0.5
    A_max: float = 400.0
    delta: Optional[np.ndarray] = None
    T_exec: float = 0.02


class UAVOfflineRLEnv:
    """Hybrid continuous-discrete finite-mission environment.

    Observation order is
    ``[uav_xy(2), device_xy-uav_xy(2K), AoI(K), SNR_dB(K),
    cumulative_energy_J(1), remaining_mission_time_s(1)]``.

    The SNR vector is measured before scheduling.  Communication therefore uses
    the channel realization in the current observation; movement changes the
    position and channel observation for the next decision.
    """

    def __init__(self, dev_coords_xy: np.ndarray, cfg: UAVEnvConfig):
        self.cfg = cfg
        self.dev_xy = np.asarray(dev_coords_xy, dtype=np.float64)
        if self.dev_xy.ndim != 2 or self.dev_xy.shape[1] != 2:
            raise ValueError("dev_coords_xy must have shape (K, 2)")
        self.K = int(self.dev_xy.shape[0])
        self.delta = (
            np.ones(self.K, dtype=np.float64)
            if cfg.delta is None
            else np.asarray(cfg.delta, dtype=np.float64)
        )
        if self.delta.shape != (self.K,):
            raise ValueError("delta must have shape (K,)")
        self.P_hover = self._hover_power()
        self.rng = np.random.default_rng()
        self.reset()

    @property
    def observation_dim(self) -> int:
        return 4 * self.K + 4

    def observation_slices(self) -> Dict[str, slice]:
        k = self.K
        return {
            "uav_xy": slice(0, 2),
            "relative_xy": slice(2, 2 + 2 * k),
            "aoi": slice(2 + 2 * k, 2 + 3 * k),
            "snr_db": slice(2 + 3 * k, 2 + 4 * k),
            "cumulative_energy_j": slice(2 + 4 * k, 3 + 4 * k),
            "remaining_time_s": slice(3 + 4 * k, 4 + 4 * k),
        }

    def _hover_power(self) -> float:
        cfg = self.cfg
        return math.sqrt(
            ((cfg.m_tot * cfg.g) ** 3)
            / (2.0 * math.pi * cfg.r_p**2 * cfg.n_p * cfg.rho)
        )

    def _move_power_increment(self, velocity: float) -> float:
        cfg = self.cfg
        velocity = float(np.clip(velocity, 0.0, cfg.v_max))
        return ((cfg.P_max - cfg.P_idle) / cfg.v_max) * velocity + cfg.P_idle

    def _los_prob(self, theta_rad: float) -> float:
        theta_deg = math.degrees(theta_rad)
        return 1.0 / (
            1.0 + self.cfg.C * math.exp(-self.cfg.D * (theta_deg - self.cfg.C))
        )

    def _path_loss_db(self, distance_3d: float, xi_db: float) -> float:
        fspl = 20.0 * math.log10(
            (4.0 * math.pi * self.cfg.fc * distance_3d) / self.cfg.c
        )
        return fspl + xi_db

    def _avg_path_loss_db(self, distance_3d: float, theta_rad: float) -> float:
        p_los = self._los_prob(theta_rad)
        return p_los * self._path_loss_db(
            distance_3d, self.cfg.xi_los_db
        ) + (1.0 - p_los) * self._path_loss_db(
            distance_3d, self.cfg.xi_nlos_db
        )

    def _sample_fading_power(self, size: int) -> np.ndarray:
        if self.cfg.fading == "deterministic":
            return np.ones(size, dtype=np.float64)
        if self.cfg.fading != "rayleigh":
            raise ValueError(f"Unsupported fading model: {self.cfg.fading}")
        return self.rng.exponential(scale=1.0, size=size)

    def _all_channel_metrics(
        self, uav_xy: np.ndarray, fading_power: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        snr = np.empty(self.K, dtype=np.float64)
        rate = np.empty(self.K, dtype=np.float64)
        sigma2 = self.cfg.N0 * self.cfg.B
        for index in range(self.K):
            horizontal = float(np.linalg.norm(self.dev_xy[index] - uav_xy))
            distance_3d = math.hypot(self.cfg.h, horizontal)
            theta = math.asin(self.cfg.h / distance_3d)
            loss_db = self._avg_path_loss_db(distance_3d, theta)
            gain = 10.0 ** (-loss_db / 10.0)
            snr[index] = self.cfg.P_dev * fading_power[index] * gain / sigma2
            rate[index] = self.cfg.B * math.log2(1.0 + snr[index])
        return snr, rate

    def _refresh_observed_channels(self) -> None:
        self.channel_fading_power = self._sample_fading_power(self.K)
        self.current_snr, self.current_rate = self._all_channel_metrics(
            self.uav_xy, self.channel_fading_power
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.uav_xy = self.rng.uniform(
            0.0, self.cfg.area_size, size=2
        ).astype(np.float64)
        self.A = np.zeros(self.K, dtype=np.float64)
        self.cumulative_energy = 0.0
        self.time_elapsed = 0.0
        self.step_count = 0
        self._refresh_observed_channels()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        relative_xy = (self.dev_xy - self.uav_xy).reshape(-1)
        snr_db = 10.0 * np.log10(np.maximum(self.current_snr, 1e-12))
        remaining_time = max(0.0, self.cfg.T_th - self.time_elapsed)
        return np.concatenate(
            [
                self.uav_xy,
                relative_xy,
                self.A,
                snr_db,
                np.array([self.cumulative_energy, remaining_time]),
            ]
        ).astype(np.float32)

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Fixed physical scaling used by online SAC only.

        Offline algorithms still fit their normalizer from the saved raw dataset.
        Keeping the dataset raw avoids coupling later experiments to this online
        optimization transform.
        """
        normalized = np.asarray(observation, dtype=np.float32).copy()
        slices = self.observation_slices()
        normalized[..., slices["uav_xy"]] = (
            2.0 * normalized[..., slices["uav_xy"]] / self.cfg.area_size - 1.0
        )
        normalized[..., slices["relative_xy"]] /= self.cfg.area_size
        # Log scaling preserves resolution in the operational 1--20 s range
        # while keeping rare values near A_max bounded.
        normalized[..., slices["aoi"]] = np.log1p(
            np.clip(normalized[..., slices["aoi"]], 0.0, self.cfg.A_max)
        ) / math.log1p(self.cfg.A_max)
        normalized[..., slices["snr_db"]] = np.clip(
            normalized[..., slices["snr_db"]] / 40.0, -2.0, 2.0
        )
        normalized[..., slices["cumulative_energy_j"]] /= 5000.0
        normalized[..., slices["remaining_time_s"]] /= self.cfg.T_th
        return normalized

    def step(
        self, action: Tuple[float, float, int]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        cfg = self.cfg
        displacement = _project_displacement(
            np.asarray(action[:2], dtype=np.float32), cfg.d_max
        ).astype(np.float64)
        scheduled = int(action[2])
        if not 0 <= scheduled <= self.K:
            raise ValueError(f"scheduling action must be in [0, {self.K}]")

        # The scheduling decision uses the SNR already present in the state.
        observed_snr = self.current_snr.copy()
        observed_rate = self.current_rate.copy()
        if scheduled == 0:
            served_idx = None
            snr = rate = T_com = E_com = 0.0
        else:
            served_idx = scheduled - 1
            snr = float(observed_snr[served_idx])
            rate = float(observed_rate[served_idx])
            T_com = cfg.Dk_bits / max(rate, 1e-12)
            E_com = T_com * (self.P_hover + cfg.P_com)

        proposed = self.uav_xy + displacement
        in_bounds = bool(
            np.all(proposed >= 0.0) and np.all(proposed <= cfg.area_size)
        )
        if in_bounds:
            effective_displacement = displacement
            self.uav_xy = proposed
        else:
            effective_displacement = np.zeros(2, dtype=np.float64)

        distance_moved = float(np.linalg.norm(effective_displacement))
        T_move = distance_moved / cfg.v_u if distance_moved > 0.0 else 0.0
        P_move = self._move_power_increment(cfg.v_u) if T_move > 0.0 else 0.0
        E_move = T_move * (self.P_hover + P_move)
        T_total = T_move + T_com + cfg.T_exec
        E_step = E_move + E_com

        self.A = np.minimum(self.A + T_total, cfg.A_max)
        if served_idx is not None:
            self.A[served_idx] = 0.0

        aoi_term = float(np.sum(self.delta * self.A)) / self.K
        reward_aoi_term = aoi_term
        energy_term = E_step / 1000.0
        reward = -(
            cfg.lam * reward_aoi_term
            + (1.0 - cfg.lam) * energy_term
        )

        self.time_elapsed += T_total
        self.cumulative_energy += E_step
        self.step_count += 1

        mission_done = self.time_elapsed >= cfg.T_th
        guard_done = cfg.H_max is not None and self.step_count >= cfg.H_max
        done = bool(mission_done or guard_done)
        termination_reason = (
            "mission_time" if mission_done else "step_guard" if guard_done else None
        )

        self._refresh_observed_channels()
        info: Dict[str, object] = {
            "served_device": served_idx,
            "snr": snr,
            "snr_db": 10.0 * math.log10(max(snr, 1e-12)) if served_idx is not None else 0.0,
            "observed_snr": observed_snr,
            "rate_bps": rate,
            "T_move": T_move,
            "T_com": T_com,
            "T_total": T_total,
            "E_move": E_move,
            "E_com": E_com,
            "E_step": E_step,
            "reward_aoi_term": reward_aoi_term,
            "reward_energy_term_kj": energy_term,
            "cumulative_energy": self.cumulative_energy,
            "time_elapsed": self.time_elapsed,
            "remaining_time": max(0.0, cfg.T_th - self.time_elapsed),
            "step_count": self.step_count,
            "out_of_bounds": not in_bounds,
            "effective_displacement": effective_displacement.copy(),
            "termination_reason": termination_reason,
        }
        return self._get_obs(), float(reward), done, info


def make_device_layout(K: int, area_size: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, area_size, size=(K, 2)).astype(np.float32)
