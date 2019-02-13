import pickle

import numpy as np

from gym_multi_robot.envs.multi_robot_game import MultiRobotGame


class ForagingGame(MultiRobotGame):
    """
    This class represent the foraging game, with as goal bringing as many tiles as possible to the foraging area.
    """

    def __init__(self, grid_size, num_tiles, target_area, robot_reset, world_reset):
        """ Target Area should be a tuple (x, y, x_length, y_length). """
        super().__init__(grid_size, num_tiles, robot_reset, world_reset)

        self.target_area = target_area
        self.collected = 0

    def reset(self):
        observations = super().reset()
        self.collected = 0
        return observations

    def valid_initial_drop(self, location):
        return not self.has_tile(location) and not self.on_target_area(location)

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
