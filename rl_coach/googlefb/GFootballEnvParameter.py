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
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter

from rl_coach.base_parameters import Parameters

from rl_coach.environments.environment import Environment




class GFootballEnvParameter(Parameters):
    def __init__(self, level=None):
        super().__init__()
        self.level = level
        self.frame_skip = 1
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()
        self.level = level
        self.frame_skip = 4
        self.seed = None
        self.human_control = False
        self.custom_reward_threshold = None

        self.experiment_path = None

        # Set target reward and target_success if present
        self.target_success_rate = 1.0

    @property
    def path(self):
        return 'rl_coach.environments.gym_environment:GymEnvironment'
        # return 'rl_coach.googlefb.GFEnvironment:GFEnvironment'
