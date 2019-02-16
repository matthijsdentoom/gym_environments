from gym.envs.registration import register

from gym_multi_robot.envs.tiling_pattern_game_alternative import CountTilingPatternGame

register(
    id='tiling-pattern-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
)

register(
    id='tiling-pattern11x11-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'x_dim': 11, 'y_dim': 11}
)

register(
    id='tiling-pattern7x5-static-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'env_storage_path': 'tiles7x5.pickle'}
)

register(
    id='tiling-pattern11x11-static-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'env_storage_path': 'tiles11x11_2_5.pickle'}
)

# This environment has all the tiles distributed in a 6x6 block giving a fitness of 67.24
register(
    id='tiling-pattern11x11-block-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'env_storage_path': 'tiles11x11_block.pickle'}
)

register(
    id='tiling-pattern11x11-block-alt-v0',
    entry_point='gym_multi_robot.envs:TilingPatternEnv',
    kwargs={'env_storage_path': 'tiles11x11_block.pickle', 'game_cls': CountTilingPatternGame}
)

register(
    id='foraging11x11-static-v0',
    entry_point='gym_multi_robot.envs:ForagingEnv',
    kwargs={'env_storage_path': 'foraging11x11.pickle'}
)

register(
    id='foraging50x50-v0',
    entry_point='gym_multi_robot.envs:ForagingEnv',
    kwargs={'x_dim': 50, 'y_dim': 50, 'num_tiles': 1000, 'target_area': (0, 0, 2, 2)}
)