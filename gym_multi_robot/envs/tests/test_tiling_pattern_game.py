import unittest

import numpy as np

from gym_multi_robot.envs.gripping_robot import Heading, GripperRobot
from gym_multi_robot.envs.robot_reset import RandomRobotReset, StaticRobotReset
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame, TilingPatternGameStorage
from gym_multi_robot.envs.world_reset import RandomWorldReset, StaticWorldReset


def static_correct_storage():

    game = TilingPatternGame((7, 5), 2, 0, RandomWorldReset())

    game.grid = np.zeros((7, 5), dtype=int)
    for i in range(0, 7, 2):
        for j in range(0, 5, 2):
            game.grid[i][j] = 1

    return TilingPatternGameStorage(game)


def create_static_game(env_storage):
    robot_reset = StaticRobotReset(GripperRobot, env_storage.robot_pos)
    world_reset = StaticWorldReset(env_storage.grid)
    world_size = env_storage.grid.shape
    lattice_size = env_storage.lattice_size

    return TilingPatternGame(world_size, lattice_size, robot_reset, world_reset)


class TestTilingPatternGame(unittest.TestCase):

    def test_one_by_one_field(self):
        game = TilingPatternGame((1, 1), 1, RandomRobotReset(GripperRobot, 0), RandomWorldReset())
        game.reset()
        self.assertEqual(1, game.num_tiles)
        self.assertTrue(game.grid[0][0])

    def test_2_by_2_field(self):
        game = TilingPatternGame((2, 2), 2, RandomRobotReset(GripperRobot, 0), RandomWorldReset())
        self.assertEqual(1, game.num_tiles)

    def test_5x5_field(self):
        game = TilingPatternGame((7, 5), 2, RandomRobotReset(GripperRobot, 0), RandomWorldReset())
        self.assertEqual(12, game.num_tiles)

    def test_11x11_field(self):
        game = TilingPatternGame((11, 11), 2, RandomRobotReset(GripperRobot, 0), RandomWorldReset())
        game.reset()
        self.assertEqual(36, game.num_tiles)
        self.assertEqual(36, np.sum(np.sum(game.grid)))

    def test_inside_grid_true(self):
        game = TilingPatternGame((11, 11), 2, RandomRobotReset(GripperRobot, 0), RandomWorldReset())
        self.assertTrue(game.inside_grid((0, 0)))
        self.assertTrue(game.inside_grid((10, 0)))
        self.assertTrue(game.inside_grid((10, 10)))
        self.assertTrue(game.inside_grid((0, 10)))

    def test_inside_grid_false(self):
        game = TilingPatternGame((11, 11), 2, RandomRobotReset(GripperRobot, 0), RandomWorldReset())
        self.assertFalse(game.inside_grid((-1, 0)))
        self.assertFalse(game.inside_grid((0, -1)))
        self.assertFalse(game.inside_grid((-1, -1)))

        self.assertFalse(game.inside_grid((11, 0)))
        self.assertFalse(game.inside_grid((0, 11)))
        self.assertFalse(game.inside_grid((11, 11)))
        self.assertFalse(game.inside_grid((-1, 11)))
        self.assertFalse(game.inside_grid((11, -1)))

    def test_tiling_pattern_fitness(self):

        storage = static_correct_storage()
        game = create_static_game(storage)
        game.reset()

        self.assertEqual(12, game.num_tiles)
        self.assertAlmostEqual(100, game.get_fitness())

    def test_static_pattern_reset(self):

        storage = static_correct_storage()
        game = create_static_game(storage)
        game.reset()

        game.grid[0][0] = 0
        game.grid[0][3] = 1

        self.assertLessEqual(game.get_fitness(), 99.5)
        game.reset()

        self.assertEqual(12, game.num_tiles)
        self.assertAlmostEqual(100, game.get_fitness())

    def test_static_robot_reset(self):
        storage = static_correct_storage()
        storage.robot_pos = [((0, 0), Heading.NORTH)]
        game = create_static_game(storage)
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
        storage = static_correct_storage()
        storage.robot_pos = [((0, 0), Heading.NORTH), ((3, 3), Heading.SOUTH)]
        game = create_static_game(storage)
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
