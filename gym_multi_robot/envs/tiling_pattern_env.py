from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame, StaticTilingPatternGame, \
    TilingPatternGameStorage
from gym_multi_robot.envs.multi_robot_env import MultiRobotEnv


class TilingPatternEnv(MultiRobotEnv):
    """ This class defines the environment for the tiling pattern problem.
    A few important notes:
    Every robot requires an action list so the actions of step should look like:
        action = [[actions robot 1], ... [actions robot n]]
    The fitness returned by step is always 0, this is because the fitness is only important in the end.
    Then the fitness can be requested using env.get_fitness()
    """

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=5, env_storage_path=None):
        super().__init__(seed)

        if env_storage_path is not None:
            env_storage = self.get_static_storage(env_storage_path)
            assert isinstance(env_storage, TilingPatternGameStorage)
            self.game = StaticTilingPatternGame(env_storage)
        else:
            self.game = TilingPatternGame((x_dim, y_dim), lattice_size, num_robots)
