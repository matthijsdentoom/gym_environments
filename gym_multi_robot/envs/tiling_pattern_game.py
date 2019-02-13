import pickle
import math
import numpy as np

from gym_multi_robot.envs.gripping_robot import GripperRobot
from gym_multi_robot.envs.multi_robot_game import MultiRobotGame
from gym_multi_robot.envs.robot_reset import StaticRobotReset


class TilingPatternGame(MultiRobotGame):
    """ This class represents a grid used for the tiling pattern problem."""

    def __init__(self, grid_size, lattice_size, robot_reset):
        super().__init__(grid_size, robot_reset)

        self.lattice_size = lattice_size
        self.num_tiles = int(math.ceil(self.GRID_W / self.lattice_size) * math.ceil(self.GRID_H / self.lattice_size))
        self.robot_cls = GripperRobot

    def reset_grid(self):

        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)

        for _ in range(self.num_tiles):
            self.randomly_drop_tile()

    def get_fitness(self):
        """ This function gets the fitness the current tile construction.
            The grid is divided in grid blocks, which all need to have the same number of tiles on it in order to
            be a perfect tile construction.
        """
        s = -100
        p_js = []

        for i in range(0, len(self.grid) - 1, self.lattice_size): # -1 to not take into account the last block.
            for j in range(0, len(self.grid[0]) - 1, self.lattice_size):

                sub_grid = self.grid[i:i + self.lattice_size + 1, j:j + self.lattice_size + 1]
                count = np.sum(np.sum(sub_grid))
                p_js.append(count)

        summed_p_j = sum(p_js)

        f = 0
        for p in p_js:
                p /= summed_p_j         # Divide by sum of all elements.
                if p != 0:
                    f += p * math.log(p)    # Calculate entropy.

        f *= s / math.log(len(p_js))
        # TODO: possibly count tiles that robot holds.
        return f

    def write(self, storage_file='tiling_pattern_game.pickle'):
        """ Writes the current configuration of robots and tiles to 2 different files."""
        storage = TilingPatternGameStorage(self)
        pickle.dump(storage, open(storage_file, 'wb'))


class TilingPatternGameStorage:
    """ This class stores all relevant information of the tiling pattern game."""

    def __init__(self, game):
        assert isinstance(game, TilingPatternGame)
        self.robot_pos = [(robot.location, robot.heading) for robot in game.robots]
        self.lattice_size = game.lattice_size
        self.grid = np.copy(game.grid)


class StaticTilingPatternGame(TilingPatternGame):
    """ This is a variant of the tiling pattern game, but now the grid is static, meaning the tiles and robots are
    at the same locations every time.
    """

    def __init__(self, game_storage):
        assert isinstance(game_storage, TilingPatternGameStorage)
        super().__init__(game_storage.grid.shape, game_storage.lattice_size,
                         StaticRobotReset(GripperRobot, game_storage.robot_pos))
        self.default_grid = np.copy(game_storage.grid)
        self.default_robot_pos = game_storage.robot_pos  # contains a position at index 0 and a heading at index 1

    def reset_grid(self):
        self.grid = np.copy(self.default_grid)
