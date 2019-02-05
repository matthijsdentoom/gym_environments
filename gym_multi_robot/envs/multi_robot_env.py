import os
import pickle

import gym
from gym.utils import seeding

from gym_multi_robot.envs.tiling_pattern_view_2d import TilingPatternView2D


def check_path(path):
    """ This function checks whether the given path exist, possibly in the samples folder,
    and if not raises an exception."""
    if not os.path.exists(path):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        rel_path = os.path.join(dir_path, "samples", path)
        if os.path.exists(rel_path):
            path = rel_path
        else:
            raise FileExistsError("Cannot find %s." % path)

    return path


class MultiRobotEnv(gym.Env):
    """ This abstract environment contains gives a template for a multi robot environment"""

    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        self.game = None
        self.game_view = None

        # Simulation related variables.
        self._seed(seed)

        # Just need to initialize the relevant attributes
        self._configure()

    def __del__(self):
        if self.game_view is not None:
            self.game_view.quit_game()

    def _seed(self, seed=None):
        # TODO: Use this random seed.
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _configure(self, display=None):
        self.display = display

    def get_fitness(self):
        """ This function returns the fitness of the current game."""
        return self.game.get_fitness()

    def step(self, actions):

        observation = self.game.update_robots(actions)
        reward = 0  # 0 during running, for speed, can be requested by env.get_fitness()
        done = self.game.game_over
        info = dict()

        return observation, reward, done, info

    def reset(self):
        return self.game.reset()

    def render(self, mode='human', close=False):
        if self.game_view is None:      # Set the pygame environment if required.
            self.game_view = TilingPatternView2D(self.game)     # TODO: update to generic.

        if close:
            self.game_view.quit_game()
            self.game_view = None

        return self.game_view.update(mode)

    def create_storage(self):
        """ This function should store the environment as it is now."""
        pass

    @classmethod
    def get_static_storage(cls, game_storage_path):

        # Check the tile paths.
        game_storage_path = check_path(game_storage_path)
        game_storage = pickle.load(open(game_storage_path, 'rb'))

        return game_storage
