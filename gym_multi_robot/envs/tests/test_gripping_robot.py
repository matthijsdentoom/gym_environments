import unittest

import numpy as np

from gym_multi_robot.envs.gripping_robot import Heading, GripperRobot
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame


class TestGripperRobot(unittest.TestCase):

    def init_robot_test(self):

        robot = GripperRobot(10, Heading.SOUTH, (5, 6))
        self.assertEqual(10, robot.identifier)
        self.assertEqual(Heading.SOUTH, robot.heading)
        self.assertEqual((5, 6), robot.location)

    def test_rotate_robot(self):
        robot = GripperRobot(10, Heading.NORTH)
        robot.rotate(1)
        self.assertEqual(Heading.EAST, robot.heading)

    def test_rotate_robot_1(self):
        robot = GripperRobot(10, Heading.NORTH)
        robot.rotate(-1)
        self.assertEqual(Heading.WEST, robot.heading)

    def test_rotate_robot_2(self):
        robot = GripperRobot(10, Heading.NORTH)
        robot.rotate(0)
        self.assertEqual(Heading.NORTH, robot.heading)

    def test_rotate_robot_3(self):
        robot = GripperRobot(10, Heading.EAST)
        robot.rotate(1)
        self.assertEqual(Heading.SOUTH, robot.heading)

    def test_rotate_robot_4(self):
        robot = GripperRobot(10, Heading.EAST)
        robot.rotate(-1)
        self.assertEqual(Heading.NORTH, robot.heading)

    def test_rotate_robot_5(self):
        robot = GripperRobot(10, Heading.SOUTH)
        robot.rotate(1)
        self.assertEqual(Heading.WEST, robot.heading)

    def test_rotate_robot_6(self):
        robot = GripperRobot(10, Heading.SOUTH)
        robot.rotate(-1)
        self.assertEqual(Heading.EAST, robot.heading)

    def test_rotate_robot_7(self):
        robot = GripperRobot(10, Heading.WEST)
        robot.rotate(1)
        self.assertEqual(Heading.NORTH, robot.heading)

    def test_rotate_robot_8(self):
        robot = GripperRobot(10, Heading.WEST)
        robot.rotate(-1)
        self.assertEqual(Heading.SOUTH, robot.heading)

    def test_pickup(self):
        robot = GripperRobot(10, Heading.WEST)
        game = TilingPatternGame((1, 1), 1, 0)
        game.reset()
        self.assertTrue(game.grid[0][0])
        robot.pickup(game)

        self.assertTrue(robot.hold_object)
        self.assertFalse(game.grid[0][0])

    def test_pickup_none(self):
        robot = GripperRobot(10, Heading.WEST, (1, 1))
        game = TilingPatternGame((2, 2), 2, 0)
        game.grid[1][1] = False
        robot.pickup(game)

        self.assertFalse(robot.hold_object)
        self.assertFalse(game.grid[1][1])

    def test_drop(self):
        robot = GripperRobot(10, Heading.WEST, (1, 1))
        robot.hold_object = True
        game = TilingPatternGame((2, 2), 2, 0)
        game.grid[1][1] = False
        robot.drop(game)

        self.assertFalse(robot.hold_object)
        self.assertTrue(game.grid[1][1])

    def test_drop_occupied(self):
        robot = GripperRobot(10, Heading.WEST)
        robot.hold_object = True
        game = TilingPatternGame((2, 2), 2, 0)
        game.grid[0][0] = True
        robot.drop(game)

        self.assertTrue(robot.hold_object)
        self.assertTrue(game.grid[0][0])

    def test_move_forward_test(self):
        robot = GripperRobot(10, Heading.EAST)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((1, 0), robot.location)

    def test_move_forward_test_1(self):
        robot = GripperRobot(10, Heading.SOUTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((0, 1), robot.location)

    def test_move_forward_test_2(self):
        robot = GripperRobot(10, Heading.WEST, (1, 0))
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((0, 0), robot.location)

    def test_move_forward_test_3(self):
        robot = GripperRobot(10, Heading.NORTH, (0, 1))
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((0, 0), robot.location)

    def test_move_off_screen_test(self):
        robot = GripperRobot(10, Heading.NORTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((0, 0), robot.location)

    def test_move_off_screen_test_1(self):
        robot = GripperRobot(10, Heading.WEST)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((0, 0), robot.location)

    def test_move_off_screen_test_2(self):
        robot = GripperRobot(10, Heading.EAST, (1, 0))
        game = TilingPatternGame((1, 1), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((1, 0), robot.location)

    def test_move_off_screen_test_3(self):
        robot = GripperRobot(10, Heading.SOUTH, (1, 1))
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([True, 0, False, False], game)

        self.assertEqual((1, 1), robot.location)

    def test_pickup_action(self):
        robot = GripperRobot(10, Heading.WEST)
        game = TilingPatternGame((1, 1), 1, 0)
        game.reset()
        self.assertTrue(game.grid[0][0])
        robot.step([False, 0, True, False], game)

        self.assertTrue(robot.hold_object)
        self.assertFalse(game.grid[0][0])

    def test_pickup_none_action(self):
        robot = GripperRobot(10, Heading.WEST, (1, 1))
        game = TilingPatternGame((2, 2), 2, 0)
        game.grid[1][1] = False
        robot.step([False, 0, True, False], game)

        self.assertFalse(robot.hold_object)
        self.assertFalse(game.grid[1][1])

    def test_drop_action(self):
        robot = GripperRobot(10, Heading.WEST, (1, 1))
        robot.hold_object = True
        game = TilingPatternGame((2, 2), 2, 0)
        game.grid[1][1] = False
        robot.step([False, 0, False, True], game)

        self.assertFalse(robot.hold_object)
        self.assertTrue(game.grid[1][1])

    def test_drop_occupied_action(self):
        robot = GripperRobot(10, Heading.WEST)
        robot.hold_object = True
        game = TilingPatternGame((2, 2), 2, 0)
        game.grid[0][0] = True
        robot.step([False, 0, False, True], game)

        self.assertTrue(robot.hold_object)
        self.assertTrue(game.grid[0][0])

    def test_rotate_robot_action(self):
        robot = GripperRobot(10, Heading.NORTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, 1, False, False], game)
        self.assertEqual(Heading.EAST, robot.heading)

    def test_rotate_robot_1_action(self):
        robot = GripperRobot(10, Heading.NORTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, -1, False, False], game)
        self.assertEqual(Heading.WEST, robot.heading)

    def test_rotate_robot_2_action(self):
        robot = GripperRobot(10, Heading.NORTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, 0, False, False], game)
        self.assertEqual(Heading.NORTH, robot.heading)

    def test_rotate_robot_3_action(self):
        robot = GripperRobot(10, Heading.EAST)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, 1, False, False], game)
        self.assertEqual(Heading.SOUTH, robot.heading)

    def test_rotate_robot_4_action(self):
        robot = GripperRobot(10, Heading.EAST)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, -1, False, False], game)
        self.assertEqual(Heading.NORTH, robot.heading)

    def test_rotate_robot_5_action(self):
        robot = GripperRobot(10, Heading.SOUTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, 1, False, False], game)
        self.assertEqual(Heading.WEST, robot.heading)

    def test_rotate_robot_6_action(self):
        robot = GripperRobot(10, Heading.SOUTH)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, -1, False, False], game)
        self.assertEqual(Heading.EAST, robot.heading)

    def test_rotate_robot_7_action(self):
        robot = GripperRobot(10, Heading.WEST)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, 1, False, False], game)
        self.assertEqual(Heading.NORTH, robot.heading)

    def test_rotate_robot_8_action(self):
        robot = GripperRobot(10, Heading.WEST)
        game = TilingPatternGame((2, 2), 2, 0)
        robot.step([False, -1, False, False], game)
        self.assertEqual(Heading.SOUTH, robot.heading)

    def test_empty_observation(self):
        robot = GripperRobot(10, Heading.NORTH, (5,5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        observation = robot.get_observation(game)

        self.assertEqual(17, len(observation))
        for i in range(len(observation)):
            self.assertEqual(0, observation[i])

    def test_border_observation(self):
        robot = GripperRobot(10, Heading.NORTH, (0, 0))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        observation = robot.get_observation(game)

        for i in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 14, 15, 16]:
            self.assertEqual(0, observation[i])

        for i in [2, 5, 8, 11]:
            self.assertEqual(1, observation[i])

    def test_border_observation_1(self):
        robot = GripperRobot(10, Heading.SOUTH, (0, 9))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        observation = robot.get_observation(game)

        for i in [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]:
            self.assertEqual(0, observation[i])

        for i in [5, 8, 11, 14]:
            self.assertEqual(1, observation[i])

    def test_border_observation_2(self):
        robot = GripperRobot(10, Heading.WEST, (9, 9))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        observation = robot.get_observation(game)

        for i in [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            self.assertEqual(0, observation[i])

        for i in [2, 5]:
            self.assertEqual(1, observation[i])

    def test_border_observation_3(self):
        robot = GripperRobot(10, Heading.EAST, (9, 0))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        observation = robot.get_observation(game)

        for i in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 14, 15, 16]:
            self.assertEqual(0, observation[i])

        for i in [2, 5, 8, 11]:
            self.assertEqual(1, observation[i])

    def test_robot_observation_1(self):
        robot = GripperRobot(10, Heading.NORTH, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.robots.append(GripperRobot(11, location=(6, 5)))
        observation = robot.get_observation(game)

        for i in range(16):
            self.assertEqual(0, observation[i])
        self.assertEqual(1, observation[16])

    def test_robot_observation_2(self):
        robot = GripperRobot(10, Heading.EAST, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.robots.append(GripperRobot(11, location=(5, 4)))
        observation = robot.get_observation(game)

        self.assertEqual(1, observation[4])

        for i in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            self.assertEqual(0, observation[i])

    def test_robot_observation_3(self):
        robot = GripperRobot(10, Heading.SOUTH, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.robots.append(GripperRobot(11, location=(5, 6)))
        observation = robot.get_observation(game)

        self.assertEqual(1, observation[10])

        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]:
            self.assertEqual(0, observation[i])

    def test_robot_observation_4(self):
        robot = GripperRobot(10, Heading.WEST, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.robots.append(GripperRobot(11, location=(4, 4)))
        observation = robot.get_observation(game)

        self.assertEqual(1, observation[13])

        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 11, 12, 14, 15, 16]:
            self.assertEqual(0, observation[i])

    def test_robot_observation_5(self):
        robot = GripperRobot(10, Heading.NORTH, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.robots.append(GripperRobot(11, location=(4, 4)))
        observation = robot.get_observation(game)

        self.assertEqual(1, observation[7])

        for i in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            self.assertEqual(0, observation[i])

    def test_robot_everywhere(self):
        robot = GripperRobot(10, Heading.SOUTH, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.robots.append(GripperRobot(11, location=(4, 5)))
        game.robots.append(GripperRobot(12, location=(6, 5)))
        game.robots.append(GripperRobot(13, location=(4, 6)))
        game.robots.append(GripperRobot(14, location=(5, 6)))
        game.robots.append(GripperRobot(15, location=(6, 6)))
        observation = robot.get_observation(game)

        for i in [4, 7, 10, 13, 16]:
            self.assertEqual(1, observation[i])

        for i in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15]:
            self.assertEqual(0, observation[i])

    def test_tile_everywhere(self):
        robot = GripperRobot(10, Heading.NORTH, (5, 5))
        game = TilingPatternGame((10, 10), 2, 0)
        game.grid = np.zeros((10, 10))
        game.grid[4][5] = 1
        game.grid[6][5] = 1
        game.grid[4][4] = 1
        game.grid[5][4] = 1
        game.grid[6][4] = 1
        observation = robot.get_observation(game)

        for i in [3, 6, 9, 12, 15]:
            self.assertEqual(1, observation[i])

        for i in [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16]:
            self.assertEqual(0, observation[i])


if __name__ == '__main__':
    unittest.main()