from gym_multi_robot.envs.foraging_game import ForagingGame, ForagingGameStorage
from gym_multi_robot.envs.foraging_robot import ForagingRobot
from gym_multi_robot.envs.multi_robot_env import MultiRobotEnv
from gym_multi_robot.envs.robot_reset import RandomRobotReset, StaticRobotReset
from gym_multi_robot.envs.world_reset import RandomWorldReset, StaticWorldReset


class ForagingEnv(MultiRobotEnv):
    """ This class defines the environment for the foraging task.
    A grid world is used an one part of the world compromises a grid, and robots are required to bring all tiles to
    a designated area.
    """

    def __init__(self, x_dim=7, y_dim=5, num_tiles=2, target_area=(0, 0, 1, 1), seed=None, num_robots=5,
                 env_storage_path=None):
        super().__init__(seed)

        if env_storage_path is not None:
            env_storage = self.get_static_storage(env_storage_path)
            assert isinstance(env_storage, ForagingGameStorage)

            world_size = env_storage.grid.shape
            num_tiles = env_storage.num_tiles
            target_area = env_storage.target_area
            robot_reset = StaticRobotReset(ForagingRobot, env_storage.robot_pos)
            world_reset = StaticWorldReset(env_storage.grid)
        else:
            world_size = (x_dim, y_dim)
            robot_reset = RandomRobotReset(ForagingRobot, num_robots)
            world_reset = RandomWorldReset()

        self.game = ForagingGame(world_size, num_tiles, target_area, robot_reset, world_reset)
