import unittest

import numpy as np

from gym_multi_robot.envs.gripping_robot import Heading
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame, StaticTilingPatternGame


def static_correct_grid():

    grid = np.zeros((7, 5), dtype=int)
    for i in range(0, 7, 2):
        for j in range(0, 5, 2):
            grid[i][j] = 1

    return grid


class TestGripperRobot(unittest.TestCase):

    def test_one_by_one_field(self):
        game = TilingPatternGame((1, 1), 1, 0)
        game.reset()
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
        game.reset()
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

    def test_tiling_pattern_fitness(self):

        grid = static_correct_grid()
        game = StaticTilingPatternGame(grid, 2, [])
        game.reset()

        self.assertEqual(12, game.num_tiles)
        self.assertAlmostEqual(100, game.get_fitness())

    def test_static_pattern_reset(self):

        grid = static_correct_grid()
        game = StaticTilingPatternGame(grid, 2, [])
        game.reset()

        grid[0][0] = 0
        grid[0][3] = 1

        self.assertLessEqual(99.5, game.get_fitness())
        game.reset()

        self.assertEqual(12, game.num_tiles)
        self.assertAlmostEqual(100, game.get_fitness())

    def test_static_robot_reset(self):
        grid = static_correct_grid()
        game = StaticTilingPatternGame(grid, 2, [((0, 0), Heading.NORTH)])
        game.reset()
        self.assertEqual((0, 0), game.robots[0].location)

        # Move the robot.
        game.robots[0].move(1, 1, game)
        self.assertFalse(game.has_robot((0, 0)))

        # Reset the game and check that robot has original parameters.
        game.reset()
        self.assertEqual((0, 0), game.robots[0].location)
        self.assertEqual(Heading.NORTH, game.robots[0].heading)
        self.assertTrue(game.has_robot((0, 0)))

    def test_static_robots_reset(self):
        grid = static_correct_grid()
        game = StaticTilingPatternGame(grid, 2, [((0, 0), Heading.NORTH), ((3, 3), Heading.SOUTH)])
        game.reset()
        self.assertEqual((0, 0), game.robots[0].location)
        self.assertEqual((3, 3), game.robots[1].location)

        # Move the robot.
        game.robots[0].move(1, 1, game)
        game.robots[1].move(1, 1, game)
        self.assertFalse(game.has_robot((0, 0)))
        self.assertFalse(game.has_robot((3, 3)))

        # Reset the game and check that robot has original parameters.
        game.reset()
        self.assertEqual((0, 0), game.robots[0].location)
        self.assertEqual((3, 3), game.robots[1].location)
        self.assertEqual(Heading.NORTH, game.robots[0].heading)
        self.assertEqual(Heading.SOUTH, game.robots[1].heading)
        self.assertTrue(game.has_robot((0, 0)))
        self.assertTrue(game.has_robot((3, 3)))


if __name__ == '__main__':
    unittest.main()