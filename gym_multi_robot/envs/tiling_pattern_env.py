import gym
from gym.utils import seeding

from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame
from gym_multi_robot.envs.tiling_pattern_view_2d import TilingPatternView2D


class TilingPatternEnv(gym.Env):
    """ This class defines the environment for the tiling pattern problem.
    A few important notes:
    Every robot requires an action list so the actions of step should look like:
        action = [[actions robot 1], ... [actions robot n]]
    The fitness returned by step is always 0, this is because the fitness is only important in the end.
    Then the fitness can be requested using env.get_fitness()
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, lattice_size=2, x_dim=7, y_dim=5, seed=None, num_robots=5):

        self.game = TilingPatternGame((x_dim, y_dim), lattice_size, num_robots)
        self.game_view = None

        # Simulation related variables.
        self._seed(seed)

        # Just need to initialize the relevant attributes
        self._configure()

    def __del__(self):
        if self.game_view is not None:
            self.game_view.quit_game()

    def _seed(self, seed=None):
        # TODO: Use this random seed.
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_fitness(self):
        """ This function returns the fitness of the current game."""
        return self.game.get_fitness()

    def is_game_over(self):
        return self.game.game_over

    def _configure(self, display=None):
        self.display = display

    def step(self, actions):

        observation = self.game.update_robots(actions)
        reward = 0  # 0 during running, for speed, can be requested by env.get_fitness()
        done = self.game.game_over
        info = dict()

        return observation, reward, done, info

    def reset(self):
        return self.game.reset()

    def render(self, mode='human', close=False):
        if self.game_view is None:      # Set the pygame environment if required.
            self.game_view = TilingPatternView2D(self.game)

        if close:
            self.game_view.quit_game()
            self.game_view = None

        return self.game_view.update(mode)
