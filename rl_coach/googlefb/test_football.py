from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.coach import CoachLauncher
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.googlefb.GFootballEnvParameter import GFootballEnvParameter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

env_params = GFootballEnvParameter(
        level='rl_coach.googlefb.GFEnvironment:GFEnvironment')

env_params.additional_simulator_parameters["env_name"] = "academy_run_to_score_with_keeper"


graph_manager = BasicRLGraphManager(
    # agent_params= RainbowDQNAgentParameters(),
    # agent_params=ClippedPPOAgentParameters(),
    agent_params= ActorCriticAgentParameters(), # PPOAgentParameters(), #DQNAgentParameters(),
    # DQNAgentParameters(),
    env_params=env_params,
    # env_params=GymVectorEnvironment(level='CartPole-v0'),
    vis_params=VisualizationParameters(render=False, native_rendering=False),
    # schedule_params=HumanPlayScheduleParameters()
    schedule_params=SimpleSchedule()
)

# coachLauncher = CoachLauncher()
#
# parser = coachLauncher.get_argument_parser()
# args = parser.parse_args()
# # args = coachLauncher.get_config_args(parser) # This is require to enrich the checkpoint dir
# coachLauncher.run_graph_manager(graph_manager, args)

# coachLauncher.launch()
graph_manager.improve()

# graph_manager.heatup(EnvironmentSteps(100))
# graph_manager.train_and_act(EnvironmentSteps(2000))

# env PYTHONPATH=/home/lizhichao/bin/god/coach:$PYTHONPATH DISPLAY=:0.0
# MESA_GL_VERSION_OVERRIDE=3.2 MESA_GLSL_VERSION_OVERRIDE=150 python /home/lizhichao/bin/god/
# coach/rl_coach/googlefb/test_football.py

# result: agent: Finished evaluation phase. Success rate = 0.0, Avg Total Reward = 0.0

# Training - Name: main_level/agent Worker: 0 Episode: 50 Total reward: 0.0 Exploration: 1.0 Steps: 4742 Training iteration: 1185
# agent: Starting evaluation phase
# 2019-07-31 16:06:10,895: Dump "episode_done": count limit reached / disabled
# 2019-07-31 16:06:10,895: Episode reward: 0.00 score: [0, 0], steps: 130, FPS: 17.6, gameFPS: 23.5
# Testing - Name: main_level/agent Worker: 0 Episode: 50 Total reward: 0.0 Exploration: 0.05 Steps: 4742 Training iteration: 1185
# 2019-07-31 16:06:15,285: Dump "episode_done": count limit reached / disabled
# 2019-07-31 16:06:15,285: Episode reward: 0.00 score: [0, 0], steps: 88, FPS: 20.2, gameFPS: 30.8
# Testing - Name: main_level/agent Worker: 0 Episode: 50 Total reward: 0.0 Exploration: 0.05 Steps: 4742 Training iteration: 1185
# 2019-07-31 16:06:22,710: Dump "episode_done": count limit reached / disabled
# 2019-07-31 16:06:22,710: Episode reward: 0.00 score: [0, 0], steps: 131, FPS: 17.7, gameFPS: 23.4
# Testing - Name: main_level/agent Worker: 0 Episode: 50 Total reward: 0.0 Exploration: 0.05 Steps: 4742 Training iteration: 1185
# 2019-07-31 16:06:39,279: Dump "episode_done": count limit reached / disabled
# 2019-07-31 16:06:39,279: Episode reward: 0.00 score: [0, 0], steps: 400, FPS: 24.2, gameFPS: 30.1
# Testing - Name: main_level/agent Worker: 0 Episode: 50 Total reward: 0.0 Exploration: 0.05 Steps: 4742 Training iteration: 1185
# 2019-07-31 16:06:43,784: Dump "episode_done": count limit reached / disabled
# 2019-07-31 16:06:43,785: Episode reward: 0.00 score: [0, 0], steps: 89, FPS: 19.9, gameFPS: 30.3
# Testing - Name: main_level/agent Worker: 0 Episode: 50 Total reward: 0.0 Exploration: 0.05 Steps: 4742 Training iteration: 1185
# agent: Finished evaluation phase. Success rate = 0.0, Avg Total Reward = 0.0

# Testing - Name: main_level/agent Worker: 0 Episode: 2000 Total reward: 0.0 Exploration: 0.05 Steps: 194440 Training iteration: 48610
# agent: Finished evaluation phase. Success rate = 0.0, Avg Total Reward = 0.0
# 2019-08-03 21:00:19,692: Start dump episode_done
# 2019-08-03 21:00:19,699: Dump written to /tmp/openai-2019-07-31-16-28-08-398826/episode_done_20190803-210019692251.dump