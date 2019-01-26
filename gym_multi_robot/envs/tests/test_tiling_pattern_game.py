import unittest

import numpy as np

from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame


class TestGripperRobot(unittest.TestCase):

    def test_one_by_one_field(self):
        game = TilingPatternGame((1, 1), 1, 0)
        self.assertEqual(1, game.num_tiles)
        self.assertTrue(game.grid[0][0])

    def test_2_by_2_field(self):
        game = TilingPatternGame((2, 2), 2, 0)
        self.assertEqual(1, game.num_tiles)

    def test_5x5_field(self):
        game = TilingPatternGame((7, 5), 2, 0)
        self.assertEqual(12, game.num_tiles)

    def test_11x11_field(self):
        game = TilingPatternGame((11, 11), 2, 0)
        self.assertEqual(36, game.num_tiles)
        self.assertEqual(36, np.sum(np.sum(game.grid)))

    def test_inside_grid_true(self):
        game = TilingPatternGame((11, 11), 2, 0)
        self.assertTrue(game.inside_grid((0, 0)))
        self.assertTrue(game.inside_grid((10, 0)))
        self.assertTrue(game.inside_grid((10, 10)))
        self.assertTrue(game.inside_grid((0, 10)))

    def test_inside_grid_false(self):
        game = TilingPatternGame((11, 11), 2, 0)
        self.assertFalse(game.inside_grid((-1, 0)))
        self.assertFalse(game.inside_grid((0, -1)))
        self.assertFalse(game.inside_grid((-1, -1)))

        self.assertFalse(game.inside_grid((11, 0)))
        self.assertFalse(game.inside_grid((0, 11)))
        self.assertFalse(game.inside_grid((11, 11)))
        self.assertFalse(game.inside_grid((-1, 11)))
        self.assertFalse(game.inside_grid((11, -1)))


if __name__ == '__main__':
    unittest.main()