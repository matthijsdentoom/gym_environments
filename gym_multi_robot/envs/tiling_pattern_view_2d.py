import pygame
import numpy as np
import operator

from gym_multi_robot.envs.gripping_robot import Heading
from gym_multi_robot.envs.tiling_pattern_game import TilingPatternGame


class TilingPatternView2D:

    def __init__(self, maze_name="TilingPattern2D", lattice_size=2,
                 world_size=(30, 30), screen_size=(600, 600)):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False

        self.__game = TilingPatternGame(grid_size=world_size, lattice_size=lattice_size)

        # to show the right and bottom border
        self.screen = pygame.display.set_mode(screen_size)
        self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))

        # show the maze
        self.__draw_grid()

        self.__draw_tiles()

        # show the robot
        self.__draw_robots()

    def update(self, mode="human"):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def reset_game(self):
        self.__game.reset_grid()
        return self.__game.reset_robots()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            self.maze_layer.fill((0, 0, 0, 0,))
            self.__draw_grid()
            self.__draw_tiles()
            self.__draw_robots()

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer, (0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __draw_grid(self):

        line_colour = (0, 0, 0, 255)

        # drawing the horizontal lines
        for y in range(self.__game.GRID_H + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.__game.GRID_W + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

    def __draw_tiles(self, colour=(0, 0, 150), transparency=235):

        for i in range(len(self.__game.grid)):
            for j in range(len(self.__game.grid[0])):

                if self.__game.grid[i][j]:
                    self.__colour_cell((i, j), colour=colour, transparency=transparency)

    def __draw_robots(self, colour=(150, 0, 0), object_colour=(0, 0, 150), transparency=255):

        for robot in self.__game.robots:
            x = int(robot.location[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
            y = int(robot.location[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
            r = int(min(self.CELL_W, self.CELL_H) / 2)

            pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

            heading = Heading.heading_to_change(robot.heading)
            scaled_heading = [r * x for x in heading]
            pygame.draw.line(self.maze_layer, (0, 0, 0), (x, y), tuple(map(operator.add, scaled_heading, (x, y))))

            if robot.hold_object:
                r_object = int(min(self.CELL_W, self.CELL_H) / 4 + 0.5)
                pygame.draw.circle(self.maze_layer, object_colour + (transparency,), (x, y), r_object)

    def __colour_cell(self, cell, colour, transparency):

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.25 * self.CELL_W + 1)
        y = int(cell[1] * self.CELL_H + 0.25 * self.CELL_H + 1)
        w = int(self.CELL_W / 2 + 0.5 - 1)
        h = int(self.CELL_H / 2 + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))

    @property
    def game(self):
        return self.__game

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.__game.GRID_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.__game.GRID_H)


if __name__ == "__main__":

    maze = TilingPatternView2D(screen_size= (500, 500), lattice_size=2, world_size=(10,10))
    maze.update()
    input("Enter any key to quit.")

