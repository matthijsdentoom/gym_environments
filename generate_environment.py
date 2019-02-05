# This file generates and stores a grid generated.
from gym_multi_robot.envs.foraging_game import ForagingGame

from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame

if __name__ == '__main__':

    x_dim = 11
    y_dim = 11
    output_file = 'foraging50x50.pickle'

    game = ForagingGame((50, 50), 1000, (0, 0, 2, 2))
    game.reset()

    print('Initial fitness:' + str(game.get_fitness()))
    print(game.num_tiles)
    print(game.grid)
    print(game.target_area)

    game.write(output_file)



