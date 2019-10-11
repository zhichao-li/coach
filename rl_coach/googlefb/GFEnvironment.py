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



# /home/lizhichao/anaconda3/envs/coachpy36/lib/python3.6/site-packages/gfootball/scenarios/academy_run_to_score.py
class GFEnvironment(Wrapper):

    def __init__(self,
                 state=["extracted_stacked"],
                 rewards="scoring",
                 dump_full_episodes=True,
                 render=True,
                 env_name="academy_empty_goal_close",
                 seed=0,
                 **kwargs
                 ):
        """Creates gfootball environment."""
        log_dir = logger.get_dir()
        print("log dir: %s" % log_dir)
        print("rewards: %s" % rewards)
        env = football_env.create_environment(
            # env_name="academy_empty_goal_close", stacked=('stacked' in state),
            # env_name="academy_corner", stacked=('stacked' in state),
            env_name=env_name, stacked=('stacked' in state),
            rewards=rewards,
            logdir=log_dir,
            enable_goal_videos=True,
            write_video=True,
            enable_full_episode_videos=dump_full_episodes,
            render=render,
            dump_frequency=50)
        env = monitor.Monitor(env, logger.get_dir() and os.path.join(log_dir,
                                                                     str(seed)))
        super().__init__(env)
