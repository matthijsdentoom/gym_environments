from gym_multi_robot.envs.gripping_robot import GripperRobot
from gym_multi_robot.envs.robot_reset import RandomRobotReset, StaticRobotReset
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame, TilingPatternGameStorage
from gym_multi_robot.envs.multi_robot_env import MultiRobotEnv
from gym_multi_robot.envs.world_reset import RandomWorldReset, StaticWorldReset


class TilingPatternEnv(MultiRobotEnv):
    """ This class defines the environment for the tiling pattern problem.
    A few important notes:
    Every robot requires an action list so the actions of step should look like:
        action = [[actions robot 1], ... [actions robot n]]
    The fitness returned by step is always 0, this is because the fitness is only important in the end.
    Then the fitness can be requested using env.get_fitness()
    """

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=5, env_storage_path=None,
                 game_cls=TilingPatternGame):
        super().__init__(seed)

        if env_storage_path is not None:
            env_storage = self.get_static_storage(env_storage_path)
            assert isinstance(env_storage, TilingPatternGameStorage)

            robot_reset = StaticRobotReset(GripperRobot, env_storage.robot_pos)
            world_reset = StaticWorldReset(env_storage.grid)
            world_size = env_storage.grid.shape
            lattice_size = env_storage.lattice_size
        else:
            robot_reset = RandomRobotReset(GripperRobot, num_robots)
            world_reset = RandomWorldReset()
            world_size = (x_dim, y_dim)

        self.game = game_cls(world_size, lattice_size, robot_reset, world_reset)
