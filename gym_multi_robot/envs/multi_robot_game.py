import random

import numpy as np

from gym_multi_robot.envs.gripping_robot import Heading


class MultiRobotGame:
    """ This class defines a multi-robot game in a grid world."""

    def __init__(self, grid_size, num_robots):
        self.grid_size = grid_size
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)

        self.num_robots = num_robots
        self.robots = []
        self.done = False
        self.robot_cls = None   # This variable should set the robot that is used.

    def update_robots(self, actions):
        if len(actions) is not len(self.robots):
            raise TypeError("Should give an action for each robot.")

        return [self.robots[i].step(actions[i], self) for i in range(len(actions))]

    def reset_robots(self):
        """ The method creates new robots at random positions."""

        self.robots.clear()
        for i in range(self.num_robots):
            self.robots.append(self.randomly_drop_robot(i))

        observations = [robot.get_observation(self) for robot in self.robots]

        return observations

    def randomly_drop_tile(self):
        while True:
            rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if not self.has_tile(rand_loc):
                self.grid[rand_loc[0]][rand_loc[1]] = 1
                break

    def randomly_drop_robot(self, identifier):
        """ This function randomly drops a robot at a non occupied place."""
        while True:     # Return statement breaks the loop.
            rand_loc = (random.randrange(0, self.grid_size[0]), random.randrange(0, self.grid_size[1]))

            if not self.has_robot(rand_loc):
                rand_heading = Heading.random_heading()
                return self.robot_cls(identifier, location=rand_loc, heading=rand_heading)

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
        if location[0] < 0 or location[1] < 0:
            return False
        if location[0] >= len(self.grid) or location[1] >= len(self.grid[0]):
            return False

        return True

    def get_fitness(self):
        """ This function should return the fitness of the current game."""
        pass

    def reset(self):
        """ This function should reset the current game. This can be randomly or static. """
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
