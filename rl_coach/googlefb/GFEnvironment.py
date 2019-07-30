from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gfootball.env as football_env
import tensorflow as tf
import multiprocessing
import os

from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gfootball.env as football_env
import tensorflow as tf
from gym import Wrapper

from rl_coach.environments.gym_environment import GymEnvironment

# flags = tf.app.flags
# FLAGS = tf.app.flags.FLAGS

# flags.DEFINE_string('level', 'academy_empty_goal_close',
#                     'Defines type of problem being solved')
# flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
#                                                  'extracted_stacked'],
#                   'Observation to be used for training.')
# flags.DEFINE_enum('reward_experiment', 'scoring',
#                   ['scoring', 'scoring_with_checkpoints'],
#                   'Reward to be used for training.')
# flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp'],
#                   'Policy architecture')
# flags.DEFINE_integer('num_timesteps', int(2e6),
#                      'Number of timesteps to run for.')
# flags.DEFINE_integer('num_envs', 8,
#                      'Number of environments to run in parallel.')
# flags.DEFINE_integer('nsteps', 128, 'Number of environment steps per epoch; '
#                      'batch size is nsteps * nenv')
# flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
# flags.DEFINE_integer('nminibatches', 8,
#                      'Number of minibatches to split one epoch to.')
# flags.DEFINE_integer('save_interval', 100,
#                      'How frequently checkpoints are saved.')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_float('lr', 0.00008, 'Learning rate')
# flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
# flags.DEFINE_float('gamma', 0.993, 'Discount factor')
# flags.DEFINE_float('cliprange', 0.27, 'Clip range')
# flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
# flags.DEFINE_bool('dump_full_episodes', False,
#                   'If True, trace is dumped after every episode.')
# flags.DEFINE_bool('dump_scores', False,
#                   'If True, sampled traces after scoring are dumped.')


state = ["extracted_stacked"]
reward_experiment = "scoring"
dump_scores = False
dump_full_episodes = False
render = False


class GFEnvironment(Wrapper):
    def create_single_football_env(self, seed):
        """Creates gfootball environment."""
        env = football_env.create_environment(
            env_name="academy_empty_goal_close", stacked=('stacked' in state),
            with_checkpoints=('with_checkpoints' in reward_experiment),
            logdir=logger.get_dir(),
            enable_goal_videos=dump_scores and (seed == 0),
            enable_full_episode_videos=dump_full_episodes and (seed == 0),
            render=render and (seed == 0),
            dump_frequency=50 if render and seed == 0 else 0)
        env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                                     str(seed)))
        return env

    def __init__(self):
        self.env = self.create_single_football_env(seed = 0)
        super().__init__(self.env)
