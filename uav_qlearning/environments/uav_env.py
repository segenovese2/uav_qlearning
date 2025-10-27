import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class UAVEnv(gym.Env):
    """
    UAV Environment for 15x15 grid with communication constraints
    - Bottom-left is (0,0) - both start and end point marked with X
    - 50-step trajectory
    - Obstacles, NLOS conditions, and stationary users
    """
    
    def __init__(self, grid_size=15, render_mode=None):
        super(UAVEnv, self).__init__()
        
        self.grid_size = grid_size
        self.cell_size = 40  # pixels per grid cell
        self.window_size = grid_size * self.cell_size
        self.render_mode = render_mode
        
        # Start and goal are both at bottom-left (0,0)
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([0, 0])
        
        # Define action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: (x, y) position
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # Define obstacles (dark grey cells) - matching the original repo
        self.obstacles = self._create_obstacles()
        
        # Define user locations (blue circles) - matching the original repo
        self.users = self._create_users()
        
        # Define NLOS conditions - matching the original repo
        self.nlos_single, self.nlos_both = self._create_nlos_conditions()
        
        # Trajectory history
        self.trajectory = []
        
        self.current_pos = None
        self.steps = 0
        self.max_steps = 50  # Fixed 50-step trip
        
        # Pygame initialization
        self.screen = None
        self.clock = None
        self.font = None
        if render_mode == "human":
            self._init_pygame()
    
    def _create_obstacles(self):
        """Create obstacle positions matching the original repo"""
        obstacles = []
        # Main obstacle structures from the original repo
        # Vertical bar
        for y in range(5, 12):
            obstacles.append([7, y])
        # Horizontal bar
        for x in range(3, 10):
            obstacles.append([x, 9])
        # L-shaped obstacle
        for x in range(11, 14):
            obstacles.append([x, 3])
        for y in range(4, 7):
            obstacles.append([11, y])
        # Additional scattered obstacles
        obstacles.extend([
            [2, 2], [2, 13], [5, 5], [8, 12], 
            [12, 10], [13, 13], [10, 2], [4, 14]
        ])
        return obstacles
    
    def _create_users(self):
        """Create stationary user positions matching the original repo"""
        return [
            [3, 12],   # User 1 - top-left area
            [12, 3]    # User 2 - bottom-right area
        ]
    
    def _create_nlos_conditions(self):
        """Create NLOS conditions based on obstacle positions and user locations"""
        nlos_single = []  # NLOS from one UE (light grey)
        nlos_both = []    # NLOS from both UEs (darker grey)
        
        user1_pos = np.array(self.users[0])
        user2_pos = np.array(self.users[1])
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if [x, y] in self.obstacles:
                    continue
                
                # Check LOS to user 1 (simple obstacle blocking)
                los_user1 = self._has_los([x, y], user1_pos)
                los_user2 = self._has_los([x, y], user2_pos)
                
                if not los_user1 and not los_user2:
                    nlos_both.append([x, y])
                elif not los_user1 or not los_user2:
                    nlos_single.append([x, y])
        
        return nlos_single, nlos_both
    
    def _has_los(self, pos1, pos2):
        """Check if there's line-of-sight between two positions (no obstacles in between)"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Simple Bresenham line algorithm to check cells between positions
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
            # Skip checking the endpoints
            if (x != x1 or y != y1) and (x != x2 or y != y2):
                if [x, y] in self.obstacles:
                    return False
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        return True
    
    def _init_pygame(self):
        """Initialize Pygame for rendering"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("UAV Trajectory Optimization - 50 Steps from (0,0) to (0,0)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        self.trajectory = [self.current_pos.copy()]
        self.steps = 0
        return self.current_pos, {}
    
    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1
        
        # Move based on action
        new_pos = self.current_pos.copy()
        if action == 0:   # up
            new_pos[1] = min(new_pos[1] + 1, self.grid_size - 1)
        elif action == 1: # down
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif action == 2: # left
            new_pos[0] = max(new_pos[0] - 1, 0)
        elif action == 3: # right
            new_pos[0] = min(new_pos[0] + 1, self.grid_size - 1)
        
        # Check if new position is valid (not in obstacle)
        if list(new_pos) in self.obstacles:
            # Hit obstacle - large penalty
            reward = -20
            terminated = False
            truncated = False
        else:
            self.current_pos = new_pos
            self.trajectory.append(self.current_pos.copy())
            reward, terminated = self._calculate_reward()
            truncated = False
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            truncated = True
            # Check if we reached the goal at exactly step 50
            if np.array_equal(self.current_pos, self.goal_pos):
                reward += 50  # Bonus for perfect completion
            else:
                reward -= 30  # Penalty for not reaching goal
        
        info = {
            'trajectory': self.trajectory.copy(),
            'steps': self.steps,
            'hit_obstacle': list(new_pos) in self.obstacles,
            'communication_quality': self._get_communication_quality(),
            'at_goal': np.array_equal(self.current_pos, self.goal_pos)
        }
        
        return self.current_pos, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """Calculate reward based on current position and communication quality"""
        # Check if we reached the goal (but we want this only at step 50)
        if np.array_equal(self.current_pos, self.goal_pos):
            if self.steps == self.max_steps:
                return 100, True  # Perfect completion!
            else:
                return -5, False  # Reached goal too early - bad!
        
        # Communication quality reward (most important during journey)
        comm_quality = self._get_communication_quality()
        comm_reward = comm_quality * 2.0
        
        # Small step penalty
        step_penalty = -0.1
        
        # Encourage exploration by giving small reward for new cells
        exploration_bonus = 0.1 if len(self.trajectory) == len(set(map(tuple, self.trajectory))) else 0
        
        total_reward = comm_reward + step_penalty + exploration_bonus
        
        return total_reward, False
    
    def _get_communication_quality(self):
        """Calculate communication quality based on LOS conditions"""
        x, y = self.current_pos
        
        # Check if in NLOS regions
        if [x, y] in self.nlos_both:
            return 0.2  # Very poor communication
        elif [x, y] in self.nlos_single:
            return 0.6  # Poor communication
        else:
            return 1.0  # Good communication
    
    def render(self):
        """Render the environment with Pygame - bottom-left is (0,0)"""
        if self.render_mode != "human" or self.screen is None:
            return
            
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid (flip y-axis so bottom is 0,0)
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (x * self.cell_size, 0), 
                           (x * self.cell_size, self.window_size))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (0, y * self.cell_size), 
                           (self.window_size, y * self.cell_size))
        
        # Draw NLOS regions (correct coordinate system)
        for cell in self.nlos_both:
            rect = pygame.Rect(cell[0] * self.cell_size, 
                             (self.grid_size - 1 - cell[1]) * self.cell_size,  # Flip y-axis
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (120, 120, 120), rect)  # Darker grey for both NLOS
        
        for cell in self.nlos_single:
            rect = pygame.Rect(cell[0] * self.cell_size, 
                             (self.grid_size - 1 - cell[1]) * self.cell_size,  # Flip y-axis
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (180, 180, 180), rect)  # Light grey for single NLOS
        
        # Draw obstacles (dark grey)
        for obs in self.obstacles:
            rect = pygame.Rect(obs[0] * self.cell_size, 
                             (self.grid_size - 1 - obs[1]) * self.cell_size,  # Flip y-axis
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (80, 80, 80), rect)  # Dark grey obstacles
        
        # Draw users (blue circles)
        for user in self.users:
            center_x = user[0] * self.cell_size + self.cell_size // 2
            center_y = (self.grid_size - 1 - user[1]) * self.cell_size + self.cell_size // 2  # Flip y-axis
            pygame.draw.circle(self.screen, (0, 100, 255), (center_x, center_y), 10)  # Blue users
            # Add user label
            user_text = self.font.render(f"UE", True, (255, 255, 255))
            text_rect = user_text.get_rect(center=(center_x, center_y))
            self.screen.blit(user_text, text_rect)
        
        # Draw start/goal position (bottom-left (0,0) with green background and red X)
        goal_rect = pygame.Rect(0 * self.cell_size,
                               (self.grid_size - 1 - 0) * self.cell_size,  # Flip y-axis for (0,0)
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)  # Green background
        
        # Draw red X mark
        goal_center_x = 0 * self.cell_size + self.cell_size // 2
        goal_center_y = (self.grid_size - 1 - 0) * self.cell_size + self.cell_size // 2  # Flip y-axis
        pygame.draw.line(self.screen, (255, 0, 0), 
                        (goal_center_x - 10, goal_center_y - 10),
                        (goal_center_x + 10, goal_center_y + 10), 3)
        pygame.draw.line(self.screen, (255, 0, 0), 
                        (goal_center_x + 10, goal_center_y - 10),
                        (goal_center_x - 10, goal_center_y + 10), 3)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            points = []
            for pos in self.trajectory:
                x_pixel = pos[0] * self.cell_size + self.cell_size // 2
                y_pixel = (self.grid_size - 1 - pos[1]) * self.cell_size + self.cell_size // 2  # Flip y-axis
                points.append((x_pixel, y_pixel))
            
            # Draw trajectory line
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 0), False, points, 3)
            
            # Draw trajectory points
            for i, point in enumerate(points):
                if i % 5 == 0:  # Draw point every 5 steps for clarity
                    pygame.draw.circle(self.screen, (0, 0, 0), point, 3)
        
        # Draw current UAV position
        uav_x = self.current_pos[0] * self.cell_size + self.cell_size // 2
        uav_y = (self.grid_size - 1 - self.current_pos[1]) * self.cell_size + self.cell_size // 2  # Flip y-axis
        pygame.draw.circle(self.screen, (255, 165, 0), (uav_x, uav_y), 8)  # Orange UAV
        
        # Add UAV direction indicator if trajectory exists
        if len(self.trajectory) > 1:
            prev_pos = self.trajectory[-2]
            prev_x = prev_pos[0] * self.cell_size + self.cell_size // 2
            prev_y = (self.grid_size - 1 - prev_pos[1]) * self.cell_size + self.cell_size // 2
            pygame.draw.line(self.screen, (0, 0, 0), (prev_x, prev_y), (uav_x, uav_y), 2)
        
        # Display information
        info_text = [
            f"Step: {self.steps}/50",
            f"Position: ({self.current_pos[0]}, {self.current_pos[1]})",
            f"Comm Quality: {self._get_communication_quality():.1f}",
            f"At Goal: {np.array_equal(self.current_pos, self.goal_pos)}"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        # Draw coordinate labels
        coord_font = pygame.font.Font(None, 20)
        for x in range(self.grid_size):
            text = coord_font.render(str(x), True, (0, 0, 0))
            self.screen.blit(text, (x * self.cell_size + 5, self.window_size - 20))
        for y in range(self.grid_size):
            text = coord_font.render(str(y), True, (0, 0, 0))
            self.screen.blit(text, (5, (self.grid_size - 1 - y) * self.cell_size + 5))
        
        # Draw legend
        legend_text = [
            "Green with X: Start/Goal (0,0)",
            "Blue: Users | Orange: UAV",
            "Dark Grey: Obstacles",
            "Grey: NLOS areas"
        ]
        
        for i, text in enumerate(legend_text):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, self.window_size - 100 + i * 25))
        
        pygame.display.flip()
        self.clock.tick(10)  # Control rendering speed
    
    def get_state_index(self, state):
        """Convert continuous state to discrete index for Q-table"""
        return state[0] * self.grid_size + state[1]
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
