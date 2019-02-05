import pickle
import random

import numpy as np

from gym_multi_robot.envs.foraging_robot import ForagingRobot
from gym_multi_robot.envs.multi_robot_game import MultiRobotGame


class ForagingGame(MultiRobotGame):
    """
    This class represent the foraging game, with as goal bringing as many tiles as possible to the foraging area.
    """

    def __init__(self, grid_size, num_tiles, target_area, num_robots=5):
        """ Target Area should be a tuple (x, y, x_length, y_length). """
        super().__init__(grid_size, num_robots)

        self.target_area = target_area
        self.num_tiles = num_tiles
        self.collected = 0
        self.robot_cls = ForagingRobot

    def reset(self):
        self.collected = 0
        self.reset_grid()
        return self.reset_robots()

    def reset_grid(self):
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)

        for _ in range(self.num_tiles):
            self.randomly_drop_tile()

    def randomly_drop_tile(self):
        while True:
            rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if not self.has_tile(rand_loc) and not self.on_target_area(rand_loc):
                self.grid[rand_loc[0]][rand_loc[1]] = 1
                break

    def on_target_area(self, loc):
        """ Returns true if the given position is within the target area of the robot."""
        return self.target_area[0] <= loc[0] < self.target_area[0] + self.target_area[2] \
               and self.target_area[1] <= loc[1] < self.target_area[1] + self.target_area[3]

    def get_fitness(self):
        return self.collected

    def direction_to_target_area(self, location):
        """ This function gives the direction to the target area with respect to the current location."""
        is_north = location[1] >= self.target_area[1] + self.target_area[3]
        is_east = location[0] < self.target_area[0]
        is_south = location[1] < self.target_area[1]
        is_west = location[0] >= self.target_area[0] + self.target_area[2]
        return is_north, is_east, is_south, is_west

    def write(self, storage_file='foraging_game.pickle'):
        """ Writes the current configuration of robots and tiles to 2 different files."""
        storage = ForagingGameStorage(self)
        pickle.dump(storage, open(storage_file, 'wb'))


class ForagingGameStorage:
    """ This class stores all objects relevant to the foraging game."""

    def __init__(self, game):
        assert isinstance(game, ForagingGame)
        self.robot_pos = [(robot.location, robot.heading) for robot in game.robots]
        self.num_tiles = game.num_tiles
        self.target_area = game.target_area
        self.grid = np.copy(game.grid)


class StaticForagingGame(ForagingGame):
    """ This class represents a static foraging game."""

    def __init__(self, game_storage):
        assert isinstance(game_storage, ForagingGameStorage)
        super().__init__(game_storage.grid.shape, game_storage.num_tiles, game_storage.target_area,
                         len(game_storage.robot_pos))
        self.default_grid = np.copy(game_storage.grid)
        self.default_robot_pos = game_storage.robot_pos  # contains a position at index 0 and a heading at index 1

    def reset_grid(self):
        self.grid = np.copy(self.default_grid)

    def reset_robots(self):
        self.robots = [ForagingRobot(i, self.default_robot_pos[i][1], self.default_robot_pos[i][0])
                       for i in range(len(self.default_robot_pos))]

        return [robot.get_observation(self) for robot in self.robots]
