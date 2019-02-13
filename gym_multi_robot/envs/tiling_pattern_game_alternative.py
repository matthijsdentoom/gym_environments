from gym_multi_robot.envs import TilingPatternEnv
from gym_multi_robot.envs.gripping_robot import GripperRobot
from gym_multi_robot.envs.multi_robot_env import MultiRobotEnv
from gym_multi_robot.envs.robot_reset import StaticRobotReset, RandomRobotReset
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame, TilingPatternGameStorage
from gym_multi_robot.envs.world_reset import StaticWorldReset, RandomWorldReset


class CountTilingPatternGame(TilingPatternGame):

    def get_fitness(self):
        """ Alternative fitness function that counts the number of locations that are correctly occupied."""
        fitness = 0
        for i in range(0, len(self.grid), self.lattice_size):  # -1 to not take into account the last block.
            for j in range(0, len(self.grid[0]), self.lattice_size):
                fitness += self.grid[i, j]

        return fitness


class CountTilingPatternEnv(MultiRobotEnv):

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=5, env_storage_path=None):
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

        self.game = CountTilingPatternGame(world_size, lattice_size, robot_reset, world_reset)
