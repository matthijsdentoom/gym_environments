import random

from gym_multi_robot.envs.gripping_robot import Heading


class RobotReset:
    """ This class is an abstract class used for resetting a robot.."""

    def __init__(self, robot_cls):
        self.robot_cls = robot_cls

    def reset(self, robot):
        pass

    @staticmethod
    def clear_robots(game):
        """ This function clears all robots from the given game."""

        game.robots.clear()

    @staticmethod
    def add_robot(game, robot):
        """ This function adds the robot to the given game."""
        game.robots.append(robot)


class StaticRobotReset(RobotReset):
    """ This class resets the robots by putting them on the grid at predefined locations."""

    def __init__(self, robot_cls, robot_pos):
        super().__init__(robot_cls)
        self.robot_pos = robot_pos

    def reset(self, game):
        self.clear_robots(game)

        for i in range(len(self.robot_pos)):
            robot = self.robot_cls(i, self.robot_pos[i][1], self.robot_pos[i][0])
            self.add_robot(game, robot)


class RandomRobotReset(RobotReset):
    """ This class resets the robots by randomly putting them in the grid."""

    def __init__(self, robot_cls, num_robots):
        super().__init__(robot_cls)
        self.num_robots = num_robots

    def reset(self, game):
        self.clear_robots(game)

        for i in range(self.num_robots):
            while True:  # Return statement breaks the loop.
                rand_loc = (random.randrange(0, game.GRID_W), random.randrange(0, game.GRID_H))

                if not game.has_robot(rand_loc):
                    rand_heading = Heading.random_heading()
                    robot = self.robot_cls(i, rand_heading, rand_loc)
                    self.add_robot(game, robot)
                    break
