import random

import numpy as np

from gym_multi_robot.envs.gripping_robot import Heading


class MultiRobotGame:
    """ This class defines a multi-robot game in a grid world."""

    def __init__(self, grid_size, robot_reset):
        self.grid_size = grid_size
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.robot_reset = robot_reset

        self.robots = []
        self.done = False
        self.robot_cls = None   # This variable should set the robot that is used.

    def update_robots(self, actions):
        if len(actions) is not len(self.robots):
            raise TypeError("Should give an action for each robot.")

        return [self.robots[i].step(actions[i], self) for i in range(len(actions))]

    def reset_grid(self):
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)

    def reset(self):
        self.reset_grid()
        return self.reset_robots()

    def reset_robots(self):
        """ The method creates new robots at random positions."""

        self.robot_reset.reset(self)
        return [robot.get_observation(self) for robot in self.robots]

    def add_robot(self, robot):
        self.robots.append(robot)

    def randomly_drop_tile(self):
        while True:
            rand_loc = (random.randrange(0, self.GRID_W), random.randrange(0, self.GRID_H))
            if not self.has_tile(rand_loc):
                self.grid[rand_loc[0]][rand_loc[1]] = 1
                break

    def has_tile(self, location):
        """ Returns true if the location has a grid."""
        if self.inside_grid(location):
            return bool(self.grid[location[0]][location[1]])
        else:
            return False

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
