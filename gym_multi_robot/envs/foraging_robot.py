from gym_multi_robot.envs.gripping_robot import GripperRobot, Heading


class ForagingRobot(GripperRobot):
    """ This class is an extension of a gripper robot, that can sense the ground it is on and gets a direction
    of the robot. """

    def __init__(self, identifier, heading=Heading.NORTH, location=(0, 0)):
        super().__init__(identifier, heading, location)

    def drop(self, game):
        """ Redefinition of drop as it now should take into account the target zone."""

        if self.hold_object and not game.has_tile(self.location):

            if game.on_target_area(self.location):  # If on the target area the object disappears and is collected.
                game.collected += 1
            else:
                game.grid[self.location[0]][self.location[1]] = True

            self.hold_object = False

    def get_observation(self, game):
        """ Redefine observation as target area information is now required."""
        observation = super().get_observation(game)             # Same observation as other robots.
        observation.append(game.on_target_area(self.location))  # Add whether robot is on target area.
        observation.extend(game.direction_to_target_area(self.location))
        observation.extend(self.heading_to_observation())

        return observation

    def heading_to_observation(self):
        """ Returns an observation for every heading."""
        return [self.heading == heading for heading in Heading]
