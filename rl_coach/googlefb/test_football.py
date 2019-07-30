


# vec_env = SubprocVecEnv([
#                             (lambda _i=i: create_single_football_env(_i))
#                             for i in range(FLAGS.num_envs)
#                             ], context=None)



env = create_single_football_env(123)
env

GymEnvironment(
             level = FLAGS.level,
             frame_skip: int,
             visualization_parameters: VisualizationParameters,
             target_success_rate: float = 1.0,
             additional_simulator_parameters: Dict[str, Any] = {},
             seed: Union[None, int] = None,
             human_control: bool = False,
             custom_reward_threshold: Union[int, float] = None,
             random_initialization_steps: int = 1,
             max_over_num_frames: int = 1,
             observation_space_type: ObservationSpaceType = None)

env = GymEnvironment(level='Breakout-v0',
                     seed=1,
                     frame_skip=4,
                     visualization_parameters=VisualizationParameters())
