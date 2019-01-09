import operator
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
            return 0, 1
        if heading is Heading.EAST:
            return 1, 0
        if heading is Heading.SOUTH:
            return 0, -1
        if heading is Heading.WEST:
            return -1, 0


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


class GripperRobot:
    """This class comprises a simple gripper robot."""

    # This variable stores the locations of the robot relative to (0, 0)
    relative_locations = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def __init__(self, heading=Heading.NORTH, location=(0, 0)):
        self.hold_object = False
        self.heading = heading
        self.location = location

    def pickup(self, grid):
        """This method picks up a object if it is not already holding one."""

        if self.hold_object:
            return False

        if self.hold_object is False and grid[self.location.first][self.location.second] is True:
            self.hold_object = True
            grid[self.location.first][self.location.second] = False

    def drop(self, grid):
        """This method drops an object if it is holding any."""

        if self.hold_object is True and grid[self.location.first][self.location.second] is False:
            grid[self.location.first][self.location.second] = True
            self.hold_object = False

    def rotate(self, rotation):
        """This function changes the heading of the robot as indicated by the rotation."""
        self.heading = Heading(rotation.value + self.heading.value % 4)

    def move(self, move_forward, rotation, grid):
        """" This function moves the robot based on the commands given to the robot."""
        self.rotate(rotation)

        if move_forward:
            change = Heading.heading_to_change(self.heading)
            new_location = tuple(map(operator.add, self.location, change))

            # Check whether the location is within the grid.
            if new_location[0] < 0 or new_location[0] >= len(grid):
                return

            if new_location[1] < 0 or new_location[1] >= len(grid[0]):
                return

            # Check whether the new location is free.
            if grid[new_location[0]][new_location[1]] is not None:
                return

            self.location = new_location

    def step(self, actions, grid):
        """This function executes the action of this robot and returns a new observation.
            action[0] = True, move forward
            action[1] = Rotation, rotate in the indicated direction
            action[2] = True, pickup object if not holding any.
            action[3] = True, drop object if holding any.
        """

        # execute actions
        if actions[2]:
            self.pickup(grid)

        if actions[3]:
            self.drop(grid)

        self.move(actions[0], actions[1], grid)

        # get observation
        return self.get_observation(grid)

    def get_observation(self, grid):
        """ This function generates an observation based on the current location of the robot."""

        locations = self.generate_observed_locations()
        observations = []  # the observations.

        for location in locations:
            observation = Observation()

            if grid.inside_grid(location):
                observation.has_tile = grid.has_tile(location)
                observation.has_robot = grid.has_robot(location)
            else:
                observation.has_obstacle = True

            observations.append(observation)

        return observations

    def generate_observed_locations(self):
        """ This method generates the locations that this robot currently observes."""

        start_location = self.heading.value * 2

        locations = [GripperRobot.relative_locations[(start_location + i) % len(GripperRobot.relative_locations)]
                     for i in range(5)]

        return locations


class TilingPatternGame:
    """ This class represents a grid used for the tiling pattern problem."""

    def __init__(self, grid_size, lattice_size):
        self.grid_size = grid_size
        self.lattice_size = lattice_size
        self.grid = [[0 for _ in range(self.GRID_H)] for _ in range(self.GRID_W)]
        self.robots = []
        self.reset_grid()

    def reset_grid(self):

        # Create the grid.
        for x in range(0, self.GRID_W, self.lattice_size):
            for y in range(0, self.GRID_H, self.lattice_size):
                self.grid[x][y] = 1

    def inside_grid(self, location):
        """ Returns whether the the given locations is within the grid."""
        if location.first < 0 or location.second < 0:
            return False
        if location.first >= len(self.grid) or location.second >= len(self.grid):
            return False

        return True

    def has_tile(self, location):
        """ Returns true if the location has a grid."""
        if self.inside_grid(location):
            return self.grid[location.first][location.second]
        else:
            return False

    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])