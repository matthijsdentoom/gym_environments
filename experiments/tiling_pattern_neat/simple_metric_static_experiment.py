"""
Simple example using the tile structure creation task.
"""

from __future__ import print_function
import os
import time

import neat
import gym
from gym_multi_robot import visualize
from gym_multi_robot.object_serializer import ObjectSerializer

num_steps = 3000
num_robots = 5
num_generations = 100

env = gym.make('tiling-pattern7x5-static-v0')


def eval_genomes(genomes, config):
    count = 0
    for genome_id, genome in genomes:
        count += 1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        start_time = time.time()
        genome.fitness = run_environment(net)
        # sub rewards.
        end_time = time.time()
        avg_time = end_time - start_time

        print("%d : avg_runtime: %s seconds ---" %(count, avg_time))


def run_environment(net):
    observation = env.reset()

    for i in range(num_steps):
        output = [net.activate(observation[i]) for i in range(len(observation))]
        observation, _, _, _ = env.step(output)

    reward = env.get_fitness()

    return reward


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, num_generations)

    visualize.visualize_stats(stats, fitness_out_file='avg_fitness_static.svg', species_out_file='species_static.svg')
    output_winner(winner, config, net_filename='nn_winner_static', genome_filename='winner_static')


def output_winner(winner, config, net_filename='nn_winner', genome_filename='winner'):
    """This function outputs the current winner in graph and in pickle file."""

    node_names = {-1: 'hold', -2: 'on object', -3: '1_obstacle', -4: '1_tile', -5: '1_robot', -6: '2_obstacle',
                  -7: '2_tile', -8: '2_robot', -9: '3_obstacle', -10: '3_tile', -11: '3_robot', -12: '4_obstacle',
                  -13: '4_tile', -14: '4_robot', -15: '5_obstacle', -16: '5_tile', -17: '5_robot',
                  0: 'drive', 1: 'rotation', 2: 'pickup', 3: 'put down'}
    visualize.draw_net(config, winner, node_names=node_names, filename=net_filename)

    ObjectSerializer.serialize(winner, genome_filename)

    print(winner)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)