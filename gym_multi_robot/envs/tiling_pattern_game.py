import random
import math

from gym_multi_robot.envs.gripping_robot import GripperRobot, Heading


class TilingPatternGame:
    """ This class represents a grid used for the tiling pattern problem."""

    def __init__(self, grid_size, lattice_size, num_robots=5):
        self.grid_size = grid_size
        self.lattice_size = lattice_size
        self.grid = [[0 for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        self.num_robots = num_robots
        self.robots = []

        self.reset_grid()
        self.reset_robots()

    def reset_grid(self):

        num_tiles = int(math.floor(self.GRID_W / self.lattice_size) * math.floor(self.GRID_H / self.lattice_size))

        for _ in range(num_tiles):

            put = False
            while not put:
                rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
                if not self.has_tile(rand_loc):
                    self.grid[rand_loc[0]][rand_loc[1]] = True
                    put = True

    def reset_robots(self):
        """ The method creates new robots at random positions."""

        observations = []

        while len(self.robots) is not self.num_robots:

            rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if not self.has_robot(rand_loc):
                rand_heading = Heading(random.randint(0, 3))
                robot = GripperRobot(len(self.robots), location=rand_loc, heading=rand_heading)
                self.robots.append(robot)
                observations.append(robot.get_observation(self))

        return observations

    def update_robots(self, actions):
        if len(actions) is not len(self.robots):
            raise TypeError("Should give an action for each robot.")

        observations = []

        for i in range(len(actions)):
            observations.append(self.robots[i].step(actions[i], self))

        return observations

    def inside_grid(self, location):
        """ Returns whether the the given locations is within the grid."""
        if location[0] < 0 or location[1] < 0:
            return False
        if location[0] >= len(self.grid) or location[1] >= len(self.grid[0]):
            return False

        return True

    def has_tile(self, location):
        """ Returns true if the location has a grid."""
        if self.inside_grid(location):
            return self.grid[location[0]][location[1]]
        else:
            return False

    def has_robot(self, location):
        """" Returns true if the given location contains a robot."""
        for robot in self.robots:
            if location[0] is robot.location[0] and location[1] is robot.location[1]:
                return True

        return False

    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])
