from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=6, step_threshold=100):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.has_package = False
        self.package_pickup = False
        self.step_threshold = step_threshold
        self.step_count = 0
        self.delivery_succeeded = False
        self.trap_location = None
        self.hit_trap = False
        self.step_threshold_reached = False
        self.render_enabled = True

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "robot_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "package_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "delivery_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "has_package": spaces.Discrete(2),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"robot_location": self._robot_location.copy(), "package_location": self._package_location.copy(),
                "delivery_location": self._delivery_location.copy(), "has_package": self._has_package}
    

    def _get_info(self):
        return {
            "distance_to_package": np.linalg.norm(
                self._robot_location - self._package_location, ord=1
            ),
            "distance_to_delivery": np.linalg.norm(
                self._robot_location - self._delivery_location, ord=1
            ),
            
            "delivery_succeeded": self._delivery_succeeded,
            "hit_trap": self._hit_trap,
            "step_threshold_reached": self._step_threshold_reached,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._has_package = False
        self._package_pickup = False
        self._delivery_succeeded = False
        self._hit_trap = False
        self._step_threshold_reached = False
        self._step_count = 0

        # Generates the trap_location randomly once, but avoids repositioning the trap with every reset.
        if self.trap_location is None:
            self.trap_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # Choose the agent's location uniformly at random
        self._robot_location = self.trap_location
        while(
            np.array_equal(self._robot_location, self.trap_location)):
            self._robot_location = self.np_random.integers(
                0, self.size, size=2, dtype=int)
        

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._package_location = self._robot_location
        while( 
            np.array_equal(self._package_location, self.trap_location)
            or np.array_equal(self._package_location, self._robot_location)):
            self._package_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        # Do the same for package_delivery
        self._delivery_location = self._package_location
        while(
            np.array_equal(self._delivery_location, self._package_location) 
            or np.array_equal(self._delivery_location, self._robot_location)
            or np.array_equal(self._delivery_location, self.trap_location)):
            self._delivery_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.render_enabled:
            self._render_frame()

        return observation, info

    def step(self, action):
                
        terminated = False
        
        # Rewards
        reward = 0
        step_penalty = -0.07
        pickup_reward = 10
        delivery_reward = 33
        step_threshold_reached = -7
        trap = -10

        # Step count and penalizing
        self._step_count += 1
        reward = step_penalty

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._robot_location = np.clip(
            self._robot_location + direction, 0, self.size - 1
        )
        
        
        if np.array_equal(self._robot_location, self.trap_location):
            reward += trap
            terminated = True
            self._hit_trap = True

        # The logic for the robot agent picking up the package
        if not self._package_pickup and not self._has_package and np.array_equal(
            self._robot_location, self._package_location):
            self._package_pickup = True 
            self._has_package = True
            reward += pickup_reward 
        
        delivery_succeeded = np.array_equal(self._robot_location, self._delivery_location)
        
        if self._has_package and delivery_succeeded:
            reward += delivery_reward
            terminated = True
            self._delivery_succeeded = True

        if self.step_threshold < self._step_count and not delivery_succeeded:
            reward += step_threshold_reached
            terminated = True
            self._step_threshold_reached = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.render_enabled:
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Package drawing
        if not self._has_package:
            pygame.draw.rect(
                canvas,
                (180, 120, 50),
                pygame.Rect(
                    pix_square_size * self._package_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Delivery drawing
        pygame.draw.rect(
            canvas,
            (0, 200, 0),
            pygame.Rect(
                pix_square_size * self._delivery_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Trap drawing
        x, y = self.trap_location * pix_square_size

        pygame.draw.line(
            canvas,
            (200, 0, 0),
            (x + 10, y + 10),
            (x + pix_square_size - 10, y + pix_square_size - 10),
            width=8,
        )

        pygame.draw.line(
            canvas,
            (200, 0, 0),
            (x + pix_square_size - 10, y + 10),
            (x + 10, y + pix_square_size - 10),
            width=8,
        )

        # Drawing the robot-agent
        
        # Robot position in pixels
        x, y = self._robot_location * pix_square_size

        # Robot body
        pygame.draw.rect(
            canvas,
            (0, 100, 200),
            pygame.Rect(
                x + pix_square_size * 0.2,
                y + pix_square_size * 0.35,
                pix_square_size * 0.6,
                pix_square_size * 0.45,
            ),
        )

        # Robot head
        pygame.draw.rect(
            canvas,
            (100, 180, 220),
            pygame.Rect(
                x + pix_square_size * 0.3,
                y + pix_square_size * 0.1,
                pix_square_size * 0.4,
                pix_square_size * 0.3,
            ),
        )

        # Eyes
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (x + pix_square_size * 0.4, y + pix_square_size * 0.22),
            pix_square_size * 0.04,
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (x + pix_square_size * 0.6, y + pix_square_size * 0.22),
            pix_square_size * 0.04,
        )

        if self._has_package:
            pygame.draw.rect(
                canvas,
                (180, 120, 50),
                pygame.Rect(
                    x + pix_square_size * 0.65,
                    y + pix_square_size * 0.15,
                    pix_square_size * 0.2,
                    pix_square_size * 0.2,
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
