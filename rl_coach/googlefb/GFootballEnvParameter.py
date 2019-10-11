from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rl_coach.environments.gym_environment import GymEnvironmentParameters

from rl_coach.base_parameters import Parameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter


class GFootballEnvParameter(GymEnvironmentParameters):
    def __init__(self, level=None):
        super().__init__()
        self.level = level
        self.frame_skip = 1
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()
        self.level = level
        self.frame_skip = 4

        self.human_control = False
        # self.custom_reward_threshold = 1
        self.experiment_path = None

        # google football environment parameters
        # self.rewards = "scoring",
        # self.dump_full_episodes = True,
        # self.render = True,
        # self.env_name = "academy_empty_goal_close"
        # self.seed = 0
        # Set target reward and target_success if present
        self.target_success_rate = 1.0

    @property
    def path(self):
        return 'rl_coach.environments.gym_environment:GymEnvironment'
        # return 'rl_coach.googlefb.GFEnvironment:GFEnvironment'
