import numpy as np


class MultiRobotGame:
    """ This class defines a multi-robot game in a grid world."""

    def __init__(self, grid_size, num_tiles, robot_reset, world_reset):
        self.grid_size = grid_size
        self.num_tiles = num_tiles
        self.world_reset = world_reset
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.robot_reset = robot_reset
        self.robots = []
        self.done = False

    def update_robots(self, actions):
        if len(actions) is not len(self.robots):
            raise TypeError("Should give an action for each robot.")

        return [self.robots[i].step(actions[i], self) for i in range(len(actions))]

    def reset(self):
        self.world_reset.reset(self)
        self.robot_reset.reset(self)
        return [robot.get_observation(self) for robot in self.robots]

    def add_robot(self, robot):
        self.robots.append(robot)

    def valid_initial_drop(self, location):
        return not self.has_tile(location)

    def has_tile(self, location):
        """ Returns true if the location has a grid."""
        return self.inside_grid(location) and bool(self.grid[location[0]][location[1]])

    def has_robot(self, location):
        """" Returns true if the given location contains a robot."""
        for robot in self.robots:
            if location == robot.location:
                return True

        return False

    def inside_grid(self, location):
        """ Returns whether the the given locations is within the grid."""
        return 0 <= location[0] < self.GRID_W and 0 <= location[1] < self.GRID_H

    def get_fitness(self):
        """ This function should return the fitness of the current game."""
        pass

    def write(self, storage_file):
        """ This function should write a representation of the game to the given file."""
        pass

    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])

    @property
    def game_over(self):
        return self.done
