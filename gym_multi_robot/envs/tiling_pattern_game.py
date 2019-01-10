import random
import math
import gym

from gym_multi_robot.envs.gripping_robot import GripperRobot, Heading


class TilingPatternGame:
    """ This class represents a grid used for the tiling pattern problem."""

    def __init__(self, grid_size, lattice_size, num_robots=5):
        self.grid_size = grid_size
        self.lattice_size = lattice_size
        self.grid = [[False for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        self.num_robots = num_robots
        self.robots = []

        self.reset_grid()
        self.reset_robots()

    def reset_grid(self):

        self.grid = [[False for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]

        num_tiles = int(math.floor(self.GRID_W / self.lattice_size) * math.floor(self.GRID_H / self.lattice_size))

        for _ in range(num_tiles):

            while True:
                rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
                if not self.has_tile(rand_loc):
                    self.grid[rand_loc[0]][rand_loc[1]] = True
                    break

        # for i in range(0, len(self.grid), self.lattice_size):
        #     for j in range(0, len(self.grid[0]), self.lattice_size):
        #         self.grid[i][j] = True

    def reset_robots(self):
        """ The method creates new robots at random positions."""

        while len(self.robots) is not self.num_robots:

            rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if not self.has_robot(rand_loc):
                rand_heading = Heading(random.randint(0, 3))
                robot = GripperRobot(len(self.robots), location=rand_loc, heading=rand_heading)
                self.robots.append(robot)

        observations = []
        for robot in self.robots:
            observations.append(robot.get_observation(self))

        return observations

    def get_fitness(self):
        """ This function gets the fitness the current tile construction.
            The grid is divided in grid blocks, which all need to have the same number of tiles on it in order to
            be a perfect tile construction.
        """
        s = -100
        p_js = []

        for i in range(0, len(self.grid) - 1, self.lattice_size): # -1 to not take into account the last block.
            for j in range(0, len(self.grid[0]) - 1, self.lattice_size):

                count = 0
                # Count the number of squares in this lattice. p_j in paper.
                for x in range(self.lattice_size + 1):
                    for y in range(self.lattice_size + 1):
                        count += int(self.grid[i + x][j + y])

                p_js.append(count)

        summed_p_j = sum(p_js)
        f = 0

        for i in range(len(p_js)):
            if p_js[i] is not 0:
                p_js[i] /= summed_p_j               # Divide by sum of all elements.
                f += p_js[i] * math.log(p_js[i])    # Calculate entropy.

        f *= s / math.log(len(p_js))

        # TODO: possibly count tiles that robot holds.

        return f

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
