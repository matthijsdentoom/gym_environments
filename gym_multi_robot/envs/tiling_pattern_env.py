import gym
import numpy as np
from gym.utils import seeding

from gym_multi_robot.envs.tiling_pattern_view_2d import TilingPatternView2D


class TilingPatternEnv(gym.Env):
    """ This class defines the environment for the tiling pattern problem."""
    metadata = {'render.modes': ['human']}

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=2):
        self.viewer = None

        self.game_view = TilingPatternView2D(
            maze_name="OpenAI Gym - Tiling Pattern ({0} x {1} x {2})".format(x_dim, y_dim, lattice_size),
            lattice_size=2, world_size=(x_dim, y_dim), screen_size=(640, 640))

        # Simulation related variables.
        self._seed(seed)
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def __del__(self):
        self.game_view.quit_game()

    def _seed(self, seed=None):
        # TODO: Use this random seed.
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_game_over(self):
        return self.game_view.game_over

    def _configure(self, display=None):
        self.display = display

    def step(self, actions):

        observation = self.game_view.game.update_robots(actions)
        reward = 0
        done = self.game_view.game_over
        info = dict()

        return observation, reward, done, info

    def reset(self):
        return self.game_view.reset_game()

    def render(self, mode='human', close=False):
        if close:
            self.game_view.quit_game()

        return self.game_view.update(mode)
