# uav_qlearning/environments/uav_env.py
# Rewritten reward to follow "Trajectory Optimization for Autonomous Flying Base Station via Reinforcement Learning"
# (Harald Bayerlein, Paul de Kerret, David Gesbert). Equations (3)-(5) used for rate/pathloss.
#
# Changes:
#  - _calculate_reward now computes sum-rate reward using pathloss + optional Rayleigh fading
#  - pathloss uses beta_shadow=0.01 when LOS is blocked (shadow)
#  - P, N, alpha, beta_shadow, stochastic_fading are tunable env attributes
#  - Existing obstacle/out-of-grid penalties preserved
#
# Source: Eurecom PDF of the paper (used for equations and reward description). :contentReference[oaicite:2]{index=2}

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Tuple

class UAVEnv(gym.Env):
    """
    UAV Environment for 15x15 grid with communication constraints.
    Reward is the sum information rate between UAV base station and users,
    following Bayerlein et al. (2018) — sum_k log2(1 + (P/N) * L_k), with
    pathloss L_k = d_k^{-alpha} * 10^(X_Rayleigh/10) * beta_shadow.
    """
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        grid_size=15,
        render_mode=None,
        P: float = 1.0,
        N: float = 1e-3,
        alpha: float = 2.0,
        beta_shadow: float = 0.01,
        stochastic_fading: bool = True,
    ):
        super(UAVEnv, self).__init__()

        # World / render params
        self.grid_size = grid_size
        self.cell_size = 40  # pixels per grid cell for rendering
        self.window_size = grid_size * self.cell_size
        self.render_mode = render_mode

        # UAV mission params
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([0, 0])
        self.max_steps = 50  # fixed flight time horizon (T discretized)

        # Channel model parameters (from paper; tunable)
        self.P = P
        self.N = N
        self.alpha = alpha
        self.beta_shadow = beta_shadow
        self.stochastic_fading = stochastic_fading

        # Action and observation space
        # Actions: 0=up,1=down,2=left,3=right
        self.action_space = spaces.Discrete(4)

        # Observation: [uav_x_norm, uav_y_norm, dist_user1_norm, dist_user2_norm, step_ratio]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # Map / obstacles / users / nlos info
        self.obstacles = self._create_obstacles()
        self.users = self._create_users()
        self.nlos_single, self.nlos_both = self._create_nlos_conditions()

        # State variables
        self.current_pos = None
        self.trajectory = []
        self.steps = 0

        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        if render_mode == "human":
            self._init_pygame()

    # -------------------------
    # World generation helpers
    # -------------------------
    def _create_obstacles(self):
        obstacles = [
            [9, 3], [9, 4], [9, 5], [9, 6],
            [10, 3], [10, 4], [10, 5], [10, 6]
        ]
        return obstacles

    def _create_users(self):
        # Two stationary users as in original setup
        return [
            [4, 12],
            [12, 8]
        ]

    def _create_nlos_conditions(self):
        """Build NLOS single/both lists based on LOS checks (keeps previous behavior)."""
        nlos_single = []
        nlos_both = []
        user1_pos = np.array(self.users[0])
        user2_pos = np.array(self.users[1])

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if [x, y] in self.obstacles:
                    continue
                los1 = self._has_los([x, y], user1_pos)
                los2 = self._has_los([x, y], user2_pos)
                if not los1 and not los2:
                    nlos_both.append([x, y])
                elif not los1 or not los2:
                    nlos_single.append([x, y])
        return nlos_single, nlos_both

    def _has_los(self, pos1, pos2) -> bool:
        """Simple Bresenham check: returns False if obstacle lies on line (excluding endpoints)."""
        x1, y1 = int(pos1[0]), int(pos1[1])
        x2, y2 = int(pos2[0]), int(pos2[1])

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            # Skip endpoints
            if not ((x == x1 and y == y1) or (x == x2 and y == y2)):
                if [x, y] in self.obstacles:
                    return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    # -------------------------
    # Rendering helpers
    # -------------------------
    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("UAV Trajectory Optimization - 50 Steps")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    # -------------------------
    # Gym API: reset / step
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        self.trajectory = [self.current_pos.copy()]
        self.steps = 0
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action, compute reward (sum-rate based), and return gymstep tuple."""
        self.steps += 1

        # Candidate new position
        new_pos = self.current_pos.copy()
        if action == 0:
            new_pos[1] = min(new_pos[1] + 1, self.grid_size - 1)
        elif action == 1:
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif action == 2:
            new_pos[0] = max(new_pos[0] - 1, 0)
        elif action == 3:
            new_pos[0] = min(new_pos[0] + 1, self.grid_size - 1)

        # If the chosen cell is an obstacle, heavy penalty (keep position unchanged)
        if list(new_pos) in self.obstacles:
            reward = -10.0
            terminated = False
            truncated = False
        else:
            # Accept move
            self.current_pos = new_pos
            self.trajectory.append(self.current_pos.copy())
            reward, terminated = self._calculate_reward()

            truncated = False

        # Terminal / truncated condition (max steps -> truncated)
        if self.steps >= self.max_steps:
            truncated = True
            final_bonus = self._calculate_final_reward()
            reward += final_bonus

        info = {
            'trajectory': self.trajectory.copy(),
            'steps': self.steps,
            'hit_obstacle': list(new_pos) in self.obstacles,
            'communication_quality': self._get_communication_quality(),  # for backwards compatibility
            'at_goal': np.array_equal(self.current_pos, self.goal_pos)
        }

        return self._get_observation(), float(reward), bool(terminated), bool(truncated), info

    # -------------------------
    # Observations
    # -------------------------
    def _get_observation(self):
        uav_x = self.current_pos[0] / (self.grid_size - 1)
        uav_y = self.current_pos[1] / (self.grid_size - 1)

        user1_dist = np.linalg.norm(self.current_pos - np.array(self.users[0])) / (self.grid_size * 1.5)
        user2_dist = np.linalg.norm(self.current_pos - np.array(self.users[1])) / (self.grid_size * 1.5)

        step_ratio = self.steps / self.max_steps

        return np.array([uav_x, uav_y, user1_dist, user2_dist, step_ratio], dtype=np.float32)

    # -------------------------
    # Reward (sum-rate) & helpers
    # -------------------------
    def _calculate_reward(self) -> Tuple[float, bool]:
        """
        Calculate reward according to the paper:
          - main component: sum_k R_k(t) with R_k(t)=log2(1 + (P/N) * L_k)
          - L_k = d_k^{-alpha} * 10^(X_Rayleigh/10) * beta_shadow
        Additional penalties: obstacle (handled earlier), small step penalty to encourage efficiency.
        Returns (reward, terminated_flag)
        """

        # compute sum-rate
        sum_rate = 0.0
        fading_factors = []  # optional debugging
        for user in self.users:
            d_k = np.linalg.norm(np.array(self.current_pos) - np.array(user))
            # distance in continuous units: add altitude effect? In paper they use constant H.
            # We treat grid positions as planar -> use d_k + 1e-6 to avoid divide by zero
            d_k_eff = math.sqrt((d_k ** 2) + 1e-6)
            # pathloss power term (d^-alpha)
            pathloss_dist = (d_k_eff ** (-self.alpha)) if d_k_eff > 0 else 1.0

            # LOS shadow factor:  beta_shadow if LOS blocked else 1
            los = self._has_los(self.current_pos, np.array(user))
            beta = 1.0 if los else self.beta_shadow

            # small-scale fading: sample Rayleigh (in linear scale via dB mapping used in paper)
            if self.stochastic_fading:
                # Rayleigh random variable (sigma=1) -> obtain a dB-like factor
                # Paper uses 10^(X_Rayleigh/10) term where XRayleigh is Rayleigh distributed in dB scale.
                # We approximate by sampling Rayleigh(sigma=1) and mapping to dB as in the paper.
                x_ray = np.random.rayleigh(scale=1.0)
                fading_linear = 10 ** (x_ray / 10.0)
            else:
                fading_linear = 1.0

            # composite pathloss factor L_k
            L_k = pathloss_dist * fading_linear * beta

            # instantaneous rate R_k(t)
            snr_term = (self.P / self.N) * L_k
            R_k = math.log2(1.0 + snr_term)

            fading_factors.append((float(fading_linear), float(beta), float(d_k_eff)))
            sum_rate += R_k

        # small step penalty to encourage shorter/efficient trajectories (paper mentions energy concerns)
        step_penalty = -0.01

        # Optional extra penalty if outside grid (shouldn't happen due to bounds above)
        outside_penalty = 0.0

        # Compose reward: main term is sum_rate (as in paper).
        # Keep reward scale manageable for learning by scaling factor if needed (expose if you want).
        reward = float(sum_rate) + step_penalty + outside_penalty

        # Termination: in this formulation we do not terminate early based solely on rate;
        # mission termination handled by max_steps/truncation. Keep terminated False here.
        terminated = False

        return reward, terminated

    def _calculate_final_reward(self):
        """Final bonus similar to your previous implementation — averages sum-rate over trajectory."""
        # compute average sum-rate along trajectory
        if len(self.trajectory) == 0:
            return 0.0

        total_sum_rate = 0.0
        for pos in self.trajectory:
            # compute instantaneous sum-rate at pos (deterministic fading for evaluation)
            sum_rate_pos = 0.0
            for user in self.users:
                d_k = np.linalg.norm(np.array(pos) - np.array(user))
                d_k_eff = math.sqrt((d_k ** 2) + 1e-6)
                pathloss_dist = (d_k_eff ** (-self.alpha)) if d_k_eff > 0 else 1.0
                los = self._has_los(pos, np.array(user))
                beta = 1.0 if los else self.beta_shadow
                fading_linear = 1.0  # use nominal fading for final bonus
                L_k = pathloss_dist * fading_linear * beta
                R_k = math.log2(1.0 + (self.P / self.N) * L_k)
                sum_rate_pos += R_k
            total_sum_rate += sum_rate_pos
        avg_sum_rate = total_sum_rate / len(self.trajectory)

        # give final bonus proportional to average sum-rate and small bonus for visiting users region
        visited_users = any(
            (np.linalg.norm(np.array(pos) - np.array(self.users[0])) < 8) or
            (np.linalg.norm(np.array(pos) - np.array(self.users[1])) < 8)
            for pos in self.trajectory
        )
        bonus = avg_sum_rate * 2.0 + (5.0 if visited_users else 0.0)

        return float(bonus)

    # -------------------------
    # Communication-quality wrapper
    # -------------------------
    def _get_communication_quality_for_pos(self, pos):
        """
        Compute a normalized communication quality in [0,1] for a given pos.
        This is a convenience measure (not the reward) based on the same channel model.
        """
        # compute sum-rate and normalize by an estimate of max achievable rate
        sum_rate = 0.0
        for user in self.users:
            d_k = np.linalg.norm(np.array(pos) - np.array(user))
            d_k_eff = math.sqrt((d_k ** 2) + 1e-6)
            pathloss_dist = (d_k_eff ** (-self.alpha)) if d_k_eff > 0 else 1.0
            los = self._has_los(pos, np.array(user))
            beta = 1.0 if los else self.beta_shadow
            # use deterministic fading for quality estimate
            L_k = pathloss_dist * 1.0 * beta
            R_k = math.log2(1.0 + (self.P / self.N) * L_k)
            sum_rate += R_k

        # Normalize: very rough normalization by a heuristic maximum (tunable)
        # With P/N large and small distances, sum_rate can be up to a few bits/s/Hz.
        # Choose normalization constant to squash values into [0,1].
        norm_const = 10.0  # empirical; tune if needed
        quality = float(min(1.0, sum_rate / norm_const))
        return quality

    def _get_communication_quality(self):
        return self._get_communication_quality_for_pos(self.current_pos)

    # -------------------------
    # Visualization (unchanged, bottom-left (0,0))
    # -------------------------
    def render(self):
        if self.render_mode != "human" or self.screen is None:
            return

        self.screen.fill((255, 255, 255))

        # grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (x * self.cell_size, 0),
                             (x * self.cell_size, self.window_size))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (0, y * self.cell_size),
                             (self.window_size, y * self.cell_size))

        # NLOS regions
        for cell in self.nlos_both:
            rect = pygame.Rect(cell[0] * self.cell_size,
                               (self.grid_size - 1 - cell[1]) * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (120, 120, 120), rect)
        for cell in self.nlos_single:
            rect = pygame.Rect(cell[0] * self.cell_size,
                               (self.grid_size - 1 - cell[1]) * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (180, 180, 180), rect)

        # obstacles
        for obs in self.obstacles:
            rect = pygame.Rect(obs[0] * self.cell_size,
                               (self.grid_size - 1 - obs[1]) * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (80, 80, 80), rect)

        # users
        for user in self.users:
            center_x = user[0] * self.cell_size + self.cell_size // 2
            center_y = (self.grid_size - 1 - user[1]) * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, (0, 100, 255), (center_x, center_y), 10)
            user_text = self.font.render(f"UE", True, (255, 255, 255))
            text_rect = user_text.get_rect(center=(center_x, center_y))
            self.screen.blit(user_text, text_rect)

        # goal
        goal_rect = pygame.Rect(0 * self.cell_size,
                                (self.grid_size - 1 - 0) * self.cell_size,
                                self.cell_size, self.cell_size)
        at_goal = np.array_equal(self.current_pos, self.goal_pos)
        goal_color = (0, 255, 0) if not at_goal else ((0, 255, 0) if ((pygame.time.get_ticks() // 500) % 2) else (255, 255, 0))
        pygame.draw.rect(self.screen, goal_color, goal_rect)
        goal_center_x = 0 * self.cell_size + self.cell_size // 2
        goal_center_y = (self.grid_size - 1 - 0) * self.cell_size + self.cell_size // 2
        pygame.draw.line(self.screen, (255, 0, 0), (goal_center_x - 10, goal_center_y - 10),
                         (goal_center_x + 10, goal_center_y + 10), 3)
        pygame.draw.line(self.screen, (255, 0, 0), (goal_center_x + 10, goal_center_y - 10),
                         (goal_center_x - 10, goal_center_y + 10), 3)

        # trajectory
        if len(self.trajectory) > 1:
            points = []
            for pos in self.trajectory:
                x_pixel = pos[0] * self.cell_size + self.cell_size // 2
                y_pixel = (self.grid_size - 1 - pos[1]) * self.cell_size + self.cell_size // 2
                points.append((x_pixel, y_pixel))
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 0), False, points, 3)

        # UAV
        uav_x = self.current_pos[0] * self.cell_size + self.cell_size // 2
        uav_y = (self.grid_size - 1 - self.current_pos[1]) * self.cell_size + self.cell_size // 2
        uav_color = (255, 165, 0)
        pygame.draw.circle(self.screen, uav_color, (uav_x, uav_y), 8)

        # info panel
        info_text = [
            f"Step: {self.steps}/{self.max_steps}",
            f"Pos: ({self.current_pos[0]}, {self.current_pos[1]})",
            f"Comm Quality: {self._get_communication_quality():.3f}",
        ]
        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()

    # -------------------------
    # Utilities
    # -------------------------
    def get_state_index(self, state):
        uav_x = int(state[0] * (self.grid_size - 1))
        uav_y = int(state[1] * (self.grid_size - 1))
        state_idx = uav_x * self.grid_size + uav_y
        return min(state_idx, self.grid_size * self.grid_size - 1)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

    def seed(self, seed=None):
        np.random.seed(seed)
