import math

from gym_multi_robot.envs import ForagingEnv
from gym_multi_robot.envs.foraging_game import ForagingGame


class ClosestGame(ForagingGame):
    """ This class offers an alternative fitness function calculation for the tiling pattern game.
    It takes the difference of each lattice block from having 4 tiles (the required number).
    """

    def get_fitness(self):
        """ This function returns the difference by taking the difference from the ideal field."""
        fitness = self.collected

        max_distance = math.sqrt(math.pow(self.GRID_W, 2) + math.pow(self.GRID_H, 2))
        for i in range(self.GRID_W):  # -1 to not take into account the last block.
            for j in range(self.GRID_H):
                if self.grid[i][j]:
                    distance = math.sqrt(math.pow(i - self.target_area[0], 2) + math.pow(j - self.target_area[1], 2))
                    fitness += 1 - distance / max_distance

        return fitness


class SteppedForagingEnv(ForagingEnv):
    """ Child classes of this class should set summed_reward in update function which can be returned by get_fitness."""

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, target_area=(0, 0, 1, 1), seed=None, num_robots=5, env_storage_path=None,
                 game_cls=ForagingEnv):
        super().__init__(lattice_size, x_dim, y_dim, target_area, seed, num_robots, env_storage_path, game_cls)

        self.summed_reward = 0

    def reset(self):
        self.summed_reward = 0
        return super().reset()

    def get_fitness(self):
        return self.summed_reward


class WeightedSumForagingEnv(SteppedForagingEnv):
    """ This class calculates the fitness by getting a weighted sum over all timesteps. (weight increases with step) """

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, target_area=(0, 0, 1, 1), seed=None, num_robots=5, env_storage_path=None,
                 game_cls=ForagingGame):
        super().__init__(lattice_size, x_dim, y_dim, target_area, seed, num_robots, env_storage_path, game_cls)

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
