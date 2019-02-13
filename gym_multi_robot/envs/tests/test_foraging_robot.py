import numpy as np
import unittest

from gym_multi_robot.envs.foraging_game import ForagingGame
from gym_multi_robot.envs.foraging_robot import ForagingRobot
from gym_multi_robot.envs.gripping_robot import Heading
from gym_multi_robot.envs.robot_reset import RandomRobotReset
from gym_multi_robot.envs.world_reset import RandomWorldReset


class TestForagingRobot(unittest.TestCase):

    def test_heading_observation(self):
        robot = ForagingRobot(1, Heading.NORTH)
        self.assertEqual([1, 0, 0, 0], robot.heading_to_observation())

    def test_heading_observation_2(self):
        robot = ForagingRobot(1, Heading.EAST)
        self.assertEqual([0, 1, 0, 0], robot.heading_to_observation())

    def test_heading_observation_3(self):
        robot = ForagingRobot(1, Heading.SOUTH)
        self.assertEqual([0, 0, 1, 0], robot.heading_to_observation())

    def test_heading_observation_4(self):
        robot = ForagingRobot(1, Heading.WEST)
        self.assertEqual([0, 0, 0, 1], robot.heading_to_observation())

    def test_drop(self):
        robot = ForagingRobot(10, Heading.WEST, (1, 1))
        robot.hold_object = True
        game = ForagingGame((2, 2), 2, (0, 0, 1, 1), RandomRobotReset(ForagingRobot, 5), RandomWorldReset())
        game.grid[1][1] = False
        robot.drop(game)

        self.assertFalse(robot.hold_object)
        self.assertTrue(game.grid[1][1])

    def test_drop_occupied(self):
        robot = ForagingRobot(10, Heading.WEST)
        robot.hold_object = True
        game = ForagingGame((2, 2), 2, (1, 1, 1, 1), RandomRobotReset(ForagingRobot, 5), RandomWorldReset())
        game.grid[0][0] = True
        robot.drop(game)

        self.assertTrue(robot.hold_object)
        self.assertTrue(game.grid[0][0])

    def test_drop_foraging_area(self):
        robot = ForagingRobot(10, Heading.WEST)
        robot.hold_object = True
        game = ForagingGame((2, 2), 2, (0, 0, 1, 1), RandomRobotReset(ForagingRobot, 5), RandomWorldReset())
        game.grid[0][0] = False
        self.assertEqual(0, game.get_fitness())

        robot.drop(game)
        self.assertFalse(robot.hold_object)
        self.assertEqual(1, game.get_fitness())

    def test_empty_observation(self):
        robot = ForagingRobot(10, Heading.NORTH, (5, 5))
        game = ForagingGame((10, 10), 2, (0, 0, 1, 1), RandomRobotReset(ForagingRobot, 5), RandomWorldReset())
        game.grid = np.zeros((10, 10))
        observation = robot.get_observation(game)

        self.assertEqual(26, len(observation))
        for i in range(len(observation)):
            if i in [18, 21, 22]:
                self.assertTrue(observation[i])
            else:
                self.assertEqual(0, observation[i])
