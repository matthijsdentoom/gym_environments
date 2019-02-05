import pickle

import numpy as np

from gym_multi_robot.envs.foraging_game import ForagingGame, ForagingGameStorage, StaticForagingGame
from gym_multi_robot.envs.multi_robot_env import MultiRobotEnv, check_path


class ForagingEnv(MultiRobotEnv):
    """ This class defines the environment for the foraging task.
    A grid world is used an one part of the world compromises a grid, and robots are required to bring all tiles to
    a designated area.
    """

    def __init__(self, x_dim=7, y_dim=5, num_tiles=2, target_area=(0, 0, 1, 1), seed=None, num_robots=5, env_storage_path=None):
        super().__init__(seed)

        if env_storage_path is not None:
            env_storage = self.get_static_storage(env_storage_path)
            assert isinstance(env_storage, ForagingGameStorage)
            self.game = StaticForagingGame(env_storage)
        else:
            self.game = ForagingGame((x_dim, y_dim), num_tiles, target_area, num_robots)