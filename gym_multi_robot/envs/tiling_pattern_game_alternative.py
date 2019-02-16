import numpy as np

from gym_multi_robot.envs import TilingPatternEnv
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame


class CountTilingPatternGame(TilingPatternGame):
    """" This class offers an alternative fitness function for the tiling pattern game.
    It sums all the blocks that are on the right location.
    """

    def get_fitness(self):
        """ Alternative fitness function that counts the number of locations that are correctly occupied."""
        fitness = 0
        for i in range(0, len(self.grid), self.lattice_size):  # -1 to not take into account the last block.
            for j in range(0, len(self.grid[0]), self.lattice_size):
                fitness += self.grid[i, j]

        return fitness


class DifferenceTilingPatternGame(TilingPatternGame):
    """ This class offers an alternative fitness function calculation for the tiling pattern game.
    It takes the difference of each lattice block from having 4 tiles (the required number).
    """

    def get_fitness(self):
        """ This function returns the difference by taking the difference from the ideal field."""
        p_js = []

        for i in range(0, len(self.grid) - 1, self.lattice_size):  # -1 to not take into account the last block.
            for j in range(0, len(self.grid[0]) - 1, self.lattice_size):

                sub_grid = self.grid[i:i + self.lattice_size + 1, j:j + self.lattice_size + 1]
                count = np.sum(np.sum(sub_grid))
                p_js.append(count)

        # Alternative fitness calculation, difference from 4 per square.
        p_js = np.array(p_js)
        difference = sum(abs(p_js - 4))
        return -difference


class SteppedTilingPatternEnv(TilingPatternEnv):
    """ Child classes of this class should set summed_reward in update function which can be returned by get_fitness."""

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=5, env_storage_path=None,
                 game_cls=TilingPatternGame):
        super().__init__(lattice_size, x_dim, y_dim, seed, num_robots, env_storage_path, game_cls)

        self.summed_reward = 0

    def reset(self):
        self.summed_reward = 0
        return super().reset()

    def get_fitness(self):
        return self.summed_reward


class SummedTilingPatternEnv(SteppedTilingPatternEnv):
    """ This function returns a fitness at every time step. """

    def step(self, actions):

        observation = self.game.update_robots(actions)
        reward = self.game.get_fitness()
        done = self.game.game_over
        info = dict()

        self.summed_reward += reward

        return observation, self.summed_reward, done, info


class WeightedSumTilingPatternEnv(SteppedTilingPatternEnv):
    """ This class calculates the fitness by getting a weighted sum over all timesteps. (weight increases with step) """

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=5, env_storage_path=None,
                 game_cls=TilingPatternGame):
        super().__init__(lattice_size, x_dim, y_dim, seed, num_robots, env_storage_path, game_cls)

        self.step_nr = 0

    def reset(self):
        self.step_nr = 0
        return super().reset()

    def step(self, actions):
        observation = self.game.update_robots(actions)
        reward = self.game.get_fitness()
        done = self.game.game_over
        info = dict()

        self.summed_reward += (reward * self.step_nr)
        self.step_nr += 1

        return observation, self.summed_reward, done, info
