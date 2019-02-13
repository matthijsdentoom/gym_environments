import random

import numpy as np


class WorldReset:
    """ This class resets the world"""

    def reset(self, game):
        pass

    @staticmethod
    def clear_world(game):
        """ This function clears all robots from the given game."""

        game.grid = np.zeros((game.GRID_W, game.GRID_H), dtype=int)


class RandomWorldReset(WorldReset):
    """ This class randomly resets the world by randomly dropping the required number of tiles."""
    def reset(self, game):

        self.clear_world(game)
        for _ in range(game.num_tiles):
            while True:
                rand_loc = (random.randrange(0, game.GRID_W), random.randrange(0, game.GRID_H))
                if game.valid_initial_drop(rand_loc):
                    game.grid[rand_loc[0]][rand_loc[1]] = 1
                    break


class StaticWorldReset(WorldReset):
    """ This function resets the world such that it is similar to the given game. """

    def __init__(self, default_grid):
        self.default_grid = default_grid

    def reset(self, game):
        game.grid = np.copy(self.default_grid)
