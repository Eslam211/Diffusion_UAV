import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class UAVEnvConfig:
    area_size: float = 1000.0
    h: float = 100.0
    H_max: int = 400
    T_th: float = 400.0

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

    # Objective weights
    lam: float = 0.5
    A_max: float = 400.0
    delta: Optional[np.ndarray] = None
    T_exec: float = 0.02

class UAVOfflineRLEnv:
    def __init__(self, dev_coords_xy: np.ndarray, cfg: UAVEnvConfig):
        self.cfg = cfg
        self.dev_xy = np.array(dev_coords_xy, dtype=float)
        assert self.dev_xy.ndim == 2 and self.dev_xy.shape[1] == 2
        self.K = self.dev_xy.shape[0]

        if self.cfg.delta is None:
            self.delta = np.ones(self.K, dtype=float)
        else:
            self.delta = np.array(self.cfg.delta, dtype=float)
            assert self.delta.shape == (self.K,)

        self.P_hover = self._hover_power()
        self.reset()

    # UAV Model
    def _hover_power(self) -> float:
        # P_hover = sqrt((m g)^3 / (2 pi r_p^2 n_p rho))
        cfg = self.cfg
        return math.sqrt(((cfg.m_tot * cfg.g) ** 3) / (2.0 * math.pi * (cfg.r_p ** 2) * cfg.n_p * cfg.rho))

    def _move_power_increment(self, v: float) -> float:
        # P_move(v) = (Pmax-Pidle)/vmax * v + Pidle
        cfg = self.cfg
        v = max(0.0, min(v, cfg.v_max))
        return ((cfg.P_max - cfg.P_idle) / cfg.v_max) * v + cfg.P_idle

    def _los_prob(self, theta_rad: float) -> float:
        theta = theta_rad * 180.0 / math.pi  # degrees
        cfg = self.cfg
        return 1.0 / (1.0 + cfg.C * math.exp(-cfg.D * (theta - cfg.C)))

    def _path_loss_db(self, R: float, xi_db: float) -> float:
        # Xi = 20log10(4 pi f_c R / c) + xi_db
        cfg = self.cfg
        fspl = 20.0 * math.log10((4.0 * math.pi * cfg.fc * R) / cfg.c)
        return fspl + xi_db

    def _avg_path_loss_db(self, R: float, theta_rad: float) -> float:
        p_los = self._los_prob(theta_rad)
        pl_los = self._path_loss_db(R, self.cfg.xi_los_db)
        pl_nlos = self._path_loss_db(R, self.cfg.xi_nlos_db)
        return p_los * pl_los + (1.0 - p_los) * pl_nlos

    def _channel_gain_linear(self, L_db: float) -> float:
        # l_bar = 10^(-L_db/10)
        return 10.0 ** (-L_db / 10.0)

    def _sample_fading_power(self) -> float:
        if self.cfg.fading == "deterministic":
            return 1.0
        # Rayleigh: |h|^2 ~ Exp(1)
        return np.random.exponential(scale=1.0)

    def _rate_bps(self, uav_xy: np.ndarray, dev_idx: int) -> Tuple[float, float]:
        cfg = self.cfg
        dx = self.dev_xy[dev_idx, 0] - uav_xy[0]
        dy = self.dev_xy[dev_idx, 1] - uav_xy[1]
        d = math.sqrt(dx * dx + dy * dy)
        R = math.sqrt(cfg.h * cfg.h + d * d)
        theta = math.asin(cfg.h / R)

        L_db = self._avg_path_loss_db(R, theta)
        l_lin = self._channel_gain_linear(L_db)
        h2 = self._sample_fading_power()

        sigma2 = cfg.N0 * cfg.B
        snr = (cfg.P_dev * h2 * l_lin) / sigma2
        rate = cfg.B * math.log2(1.0 + snr)
        return snr, rate

    # MDP environment
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        self.uav_xy = np.array([
            np.random.uniform(0.0, self.cfg.area_size),
            np.random.uniform(0.0, self.cfg.area_size),
        ], dtype=float)

        self.A = np.zeros(self.K, dtype=float)

        self.E_total_last = 0.0
        self.time_elapsed = 0.0
        self.step_count = 0

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.uav_xy, self.A, np.array([self.E_total_last], dtype=float)], axis=0)

    def step(self, action: Tuple[float, float, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        cfg = self.cfg
        w_x, w_y, s = action
        s = int(s)
        
        d = math.sqrt(w_x*w_x + w_y*w_y)
        if d > self.cfg.d_max:
            scale = self.cfg.d_max / (d + 1e-12)
            w_x *= scale
            w_y *= scale

        proposed = self.uav_xy + np.array([w_x, w_y], dtype=float)

        if (0.0 <= proposed[0] <= cfg.area_size) and (0.0 <= proposed[1] <= cfg.area_size):
            self.uav_xy = proposed
            d_move = float(math.sqrt(w_x * w_x + w_y * w_y))
        else:
            d_move = 0.0

        T_move = d_move / cfg.v_u if d_move > 0 else 0.0

        if s == 0:
            T_com = 0.0
            E_com = 0.0
            snr = 0.0
            rate = 0.0
            served_idx = None
        else:
            served_idx = s - 1
            snr, rate = self._rate_bps(self.uav_xy, served_idx)
            T_com = cfg.Dk_bits / max(rate, 1e-12) # Avoid division by zero
            E_com = T_com * (self.P_hover + cfg.P_com)

        P_move = self._move_power_increment(cfg.v_u) if T_move > 0 else 0.0
        E_move = T_move * (self.P_hover + P_move)

        T_total = T_move + T_com + cfg.T_exec
        E_total = E_com + E_move

        self.A = np.minimum(self.A + T_total, cfg.A_max)
        if served_idx is not None:
            self.A[served_idx] = 0.0

        aoi_term = float(np.sum(self.delta * self.A)) / self.K
        energy_term = E_total / 1000.0
        reward = -(cfg.lam * aoi_term + (1-cfg.lam) * energy_term)

        self.time_elapsed += T_total
        self.step_count += 1
        self.E_total_last = float(E_total)

        done = (self.time_elapsed > cfg.T_th) or (self.step_count >= cfg.H_max)

        info = {
            "served_device": served_idx,
            "snr": snr,
            "rate_bps": rate,
            "T_move": T_move,
            "T_com": T_com,
            "T_total": T_total,
            "E_move": E_move,
            "E_com": E_com,
            "E_total": E_total,
            "time_elapsed": self.time_elapsed,
            "step_count": self.step_count,
        }

        return self._get_obs(), reward, done, info

