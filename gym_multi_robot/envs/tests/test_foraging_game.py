import numpy as np
import unittest

from gym_multi_robot.envs.foraging_game import ForagingGame, StaticForagingGame, ForagingGameStorage
from gym_multi_robot.envs.foraging_robot import ForagingRobot
from gym_multi_robot.envs.gripping_robot import Heading
from gym_multi_robot.envs.robot_reset import RandomRobotReset


def static_correct_storage():

    game = ForagingGame((7, 5), 2, (2, 2, 2, 2), RandomRobotReset(ForagingRobot, 5))
    game.grid = np.zeros((7, 5), dtype=int)
    game.grid[0][0] = 1
    game.grid[0][1] = 1

    return ForagingGameStorage(game)


class TestForagingGame(unittest.TestCase):

    def test_one_by_one_field(self):
        game = ForagingGame((1, 1), 1, (2, 2, 1, 1), RandomRobotReset(ForagingRobot, 0))
        game.reset()
        self.assertEqual(1, game.num_tiles)
        self.assertTrue(game.grid[0][0])

    def test_on_target_location(self):
        game = ForagingGame((7, 5), 1, (2, 2, 2, 2), RandomRobotReset(ForagingRobot, 5))
        self.assertTrue(game.on_target_area((2, 2)))
        self.assertTrue(game.on_target_area((2, 3)))
        self.assertTrue(game.on_target_area((3, 2)))
        self.assertTrue(game.on_target_area((3, 3)))
        self.assertFalse(game.on_target_area((1, 2)))
        self.assertFalse(game.on_target_area((2, 4)))
        self.assertFalse(game.on_target_area((3, 1)))
        self.assertFalse(game.on_target_area((4, 2)))

    def test_no_drop_in_target_area(self):
        game = ForagingGame((3, 3), 8, (2, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))
        self.assertTrue(game.on_target_area((2, 2)))
        self.assertFalse(game.has_tile((2, 2)))

    def test_get_fitness(self):
        game = ForagingGame((3, 3), 8, (2, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))
        game.collected = 2
        self.assertEqual(2, game.get_fitness())

    def test_reset(self):
        game = ForagingGame((3, 3), 8, (2, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))
        game.collected = 2

        game.reset()
        self.assertEqual(0, game.get_fitness())

    def test_static_pattern_reset(self):
        storage = static_correct_storage()
        game = StaticForagingGame(storage)
        game.reset()

        self.assertTrue(game.has_tile((0, 0)))
        self.assertTrue(game.has_tile((0, 1)))

    def test_direction_to_target_area_1(self):
        game = ForagingGame((4, 4), 8, (1, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))

        self.assertEqual((0, 0, 0, 0), game.direction_to_target_area((1, 2)))

    def test_direction_to_target_area_2(self):
        game = ForagingGame((4, 4), 8, (1, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))

        self.assertEqual((0, 1, 0, 0), game.direction_to_target_area((0, 2)))

    def test_direction_to_target_area_3(self):
        game = ForagingGame((4, 4), 8, (1, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))

        self.assertEqual((0, 0, 1, 0), game.direction_to_target_area((1, 1)))

    def test_direction_to_target_area_4(self):
        game = ForagingGame((4, 4), 8, (1, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))

        self.assertEqual((0, 0, 0, 1), game.direction_to_target_area((2, 2)))

    def test_direction_to_target_area_5(self):
        game = ForagingGame((4, 4), 8, (1, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))

        self.assertEqual((1, 0, 0, 0), game.direction_to_target_area((1, 3)))

    def test_direction_to_target_area_angles(self):
        game = ForagingGame((4, 4), 8, (1, 2, 1, 1), RandomRobotReset(ForagingRobot, 5))

        self.assertEqual((0, 1, 1, 0), game.direction_to_target_area((0, 0)))
        self.assertEqual((1, 0, 0, 1), game.direction_to_target_area((3, 3)))
        self.assertEqual((1, 1, 0, 0), game.direction_to_target_area((0, 3)))
        self.assertEqual((0, 0, 1, 1), game.direction_to_target_area((3, 0)))

    def test_static_robot_reset(self):
        storage = static_correct_storage()
        storage.robot_pos = [((0, 0), Heading.NORTH)]
        game = StaticForagingGame(storage)
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
        game = StaticForagingGame(storage)
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
