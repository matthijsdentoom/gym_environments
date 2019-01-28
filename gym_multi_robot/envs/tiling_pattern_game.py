import pickle
import random
import math
import numpy as np

from gym_multi_robot.envs.gripping_robot import GripperRobot, Heading


class TilingPatternGame:
    """ This class represents a grid used for the tiling pattern problem."""

    def __init__(self, grid_size, lattice_size, num_robots=5):
        self.grid_size = grid_size
        self.lattice_size = lattice_size

        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.num_tiles = int(math.ceil(self.GRID_W / self.lattice_size) * math.ceil(self.GRID_H / self.lattice_size))

        self.num_robots = num_robots
        self.robots = []

    def reset_grid(self):

        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)

        for _ in range(self.num_tiles):
            self.randomly_drop_tile()

    def randomly_drop_tile(self):
        while True:
            rand_loc = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if not self.has_tile(rand_loc):
                self.grid[rand_loc[0]][rand_loc[1]] = 1
                break

    def reset_robots(self):
        """ The method creates new robots at random positions."""

        self.robots.clear()
        for i in range(self.num_robots):
            self.robots.append(self.randomly_drop_robot(i))

        observations = [robot.get_observation(self) for robot in self.robots]

        return observations

    def randomly_drop_robot(self, identifier):
        """ This function randomly drops a robot at a non occupied place."""
        while True:     # Return statement breaks the loop.
            rand_loc = (random.randrange(0, self.grid_size[0]), random.randrange(0, self.grid_size[1]))

            if not self.has_robot(rand_loc):
                rand_heading = Heading.random_heading()
                return GripperRobot(identifier, location=rand_loc, heading=rand_heading)

    def reset(self):
        """ This function resets the game and returns an initial observation from all robots."""
        self.reset_grid()
        return self.reset_robots()

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

    def update_robots(self, actions):
        if len(actions) is not len(self.robots):
            raise TypeError("Should give an action for each robot.")

        return [self.robots[i].step(actions[i], self) for i in range(len(actions))]

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
            return bool(self.grid[location[0]][location[1]])
        else:
            return False

    def has_robot(self, location):
        """" Returns true if the given location contains a robot."""
        for robot in self.robots:
            if location == robot.location:
                return True

        return False

    def write_config(self, tile_file_name='tiles.npy', robot_file_name='robots.pickle'):
        """ Writes the current configuration of robots and tiles to 2 different files."""
        robot_pos = [(robot.location, robot.heading) for robot in self.robots]
        pickle.dump(robot_pos, open(robot_file_name, 'wb'))
        np.save(tile_file_name, self.grid)

    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])

    @property
    def game_over(self):
        return False


class StaticTilingPatternGame(TilingPatternGame):
    """ This is a variant of the tiling pattern game, but now the grid is static, meaning the tiles and robots are
    at the same locations every time.
    """

    def __init__(self, grid, lattice_size, robot_pos):
        super().__init__(grid.shape, lattice_size, len(robot_pos))
        self.default_grid = np.copy(grid)
        self.default_robot_pos = robot_pos  # contains a position at index 0 and a heading at index 1

    def reset_grid(self):
        self.grid = np.copy(self.default_grid)

    def reset_robots(self):
        self.robots = [GripperRobot(i, self.default_robot_pos[i][1], self.default_robot_pos[i][0])
                       for i in range(len(self.default_robot_pos))]

        return [robot.get_observation(self) for robot in self.robots]
