# This file generates and stores a grid generated.
import numpy as np

from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame

if __name__ == '__main__':

    x_dim = 11
    y_dim = 11
    output_file = 'tiles11x11.npy'

    game = TilingPatternGame((11, 11), 2)
    game.reset()

    print('Initial fitness:' + str(game.get_fitness()))
    print(game.num_tiles)
    print(game.grid)
    np.save(output_file, game.grid)



