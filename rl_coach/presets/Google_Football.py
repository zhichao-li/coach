from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters

from rl_coach.agents.ppo_agent import PPOAgentParameters

from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.coach import CoachLauncher
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.googlefb.GFootballEnvParameter import GFootballEnvParameter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

graph_manager = BasicRLGraphManager(
    # agent_params= RainbowDQNAgentParameters(),
    agent_params=ClippedPPOAgentParameters(),
    # agent_params=  PPOAgentParameters(), #ActorCriticAgentParameters(), #DQNAgentParameters(),
    # DQNAgentParameters(),
    env_params=GFootballEnvParameter(
        level='rl_coach.googlefb.GFEnvironment:GFEnvironment'),
    # env_params=GymVectorEnvironment(level='CartPole-v0'),
    vis_params=VisualizationParameters(render=False, native_rendering=False),
    # schedule_params=HumanPlayScheduleParameters()
    schedule_params=SimpleSchedule()
)