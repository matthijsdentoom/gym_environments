import operator
from random import choice
from enum import Enum


class Heading(Enum):
    """""This enum indicates the heading of the robot."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @staticmethod
    def heading_to_change(heading):

        if heading is Heading.NORTH:
            return 0, -1
        if heading is Heading.EAST:
            return 1, 0
        if heading is Heading.SOUTH:
            return 0, 1
        if heading is Heading.WEST:
            return -1, 0

    @staticmethod
    def random_heading():
        return choice(list(Heading))


class Rotation(Enum):
    """ This enum indicates the rotation direction of the robot."""
    CLOCKWISE = 1
    NOT = 0
    COUNTERCLOCKWISE = -1


class Observation:
    """ This class represents an observation which is made by a robot."""

    def __init__(self):
        self.has_tile = False
        self.has_obstacle = False
        self.has_robot = False

    def to_int_list(self):
        return (int(self.has_tile), int(self.has_obstacle), int(self.has_robot))


class GripperRobot:
    """This class comprises a simple gripper robot."""

    # This variable stores the locations of the robot relative to (0, 0)
    relative_locations = ((-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1))

    def __init__(self, identifier, heading=Heading.NORTH, location=(0, 0)):
        self.hold_object = False
        self.heading = heading

        self.location = location
        self.identifier = identifier

    def pickup(self, game):
        """This method picks up a object if it is not already holding one."""

        if self.hold_object:
            return False

        if not self.hold_object and game.has_tile(self.location):
            self.hold_object = True
            game.grid[self.location[0]][self.location[1]] = False

    def drop(self, game):
        """This method drops an object if it is holding any."""

        if self.hold_object and not game.has_tile(self.location):
            game.grid[self.location[0]][self.location[1]] = True
            self.hold_object = False

    def rotate(self, rotation):
        """This function changes the heading of the robot as indicated by the rotation."""

        self.heading = Heading((rotation + self.heading.value) % 4)

    def move(self, move_forward, rotation, game):
        """" This function moves the robot based on the commands given to the robot."""
        self.rotate(rotation)

        if move_forward:
            change = Heading.heading_to_change(self.heading)
            new_location = tuple(map(operator.add, self.location, change))

            # Check whether the location is within the grid.
            if not game.inside_grid(new_location):
                return

            # Check whether the new location is free.
            if game.has_robot(new_location):    # TODO: maybe it is better to leave this out.
                return

            self.location = new_location

    def step(self, actions, game):
        """This function executes the action of this robot and returns a new observation.
            action[0] = True, move forward
            action[1] = Rotation, rotate in the indicated direction
            action[2] = True, pickup object if not holding any.
            action[3] = True, drop object if holding any.
            observation[0] = True if holds a tile
            observation[1] = True on top of a tile
            observation[2 .. n] = observations about neighbouring locations.
        """

        # execute actions
        if bool(round(actions[2])) and not actions[2] < 0:  # only pickup if positive integer.
            self.pickup(game)

        if bool(round(actions[3])) and not actions[3] < 0:  # only drop if positive integer.
            self.drop(game)

        move_bool = bool(round(actions[0])) and not actions[0] < 0

        actions[1] = int(round(actions[1]))
        rotate = int(actions[1] > 0) - int(actions[1] < 0)

        self.move(move_bool, rotate, game)

        # get observation
        return self.get_observation(game)

    def get_observation(self, game):
        """ This function generates an observation based on the current location of the robot."""

        locations = self.generate_observed_locations()
        observations = [int(self.hold_object), int(game.has_tile(self.location))]  # the observations.

        for location in locations:
            observation = (int(not game.inside_grid(location)),     # Obstacle, for now only inside grid.
                           int(game.has_tile(location)),            # Has tile
                           int(game.has_robot(location)))           # Has robot.
            observations.extend(observation)

        return observations

    def generate_observed_locations(self):
        """ This method generates the locations that this robot currently observes."""
        start_location = self.heading.value * 2

        # Generate the relative locations
        rel_locations = [GripperRobot.relative_locations[(start_location + i) % len(GripperRobot.relative_locations)]
                     for i in range(5)]

        # Add current location to the set of sensor locations.
        locations = [(location[0] + self.location[0], location[1] + self.location[1]) for location in rel_locations]


        return locations
