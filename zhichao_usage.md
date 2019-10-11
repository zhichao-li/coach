
- The integration code is under folder: /home/lizhichao/bin/god/coach/rl_coach/googlefb

- Google football modified scenario:

/home/lizhichao/anaconda3/envs/coachpy36/lib/python3.6/site-packages/gfootball/scenarios/academy_run_to_score_with_keeper.py

```
def build_scenario(builder):
  builder.SetFlag('game_duration', 400)
  builder.SetFlag('deterministic', False)
  builder.SetFlag('offsides', False)
  builder.SetFlag('end_episode_on_score', True)
  builder.SetFlag('end_episode_on_out_of_play', True)
  builder.SetFlag('end_episode_on_possession_change', True)
  builder.SetBallPosition(0.02, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.0, 0.0, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.12, 0.2, e_PlayerRole_LB)
  builder.AddPlayer(0.12, 0.1, e_PlayerRole_CB)
  builder.AddPlayer(0.12, 0.0, e_PlayerRole_CM)
  builder.AddPlayer(0.12, -0.1, e_PlayerRole_CB)
  builder.AddPlayer(0.12, -0.2, e_PlayerRole_RB)
```

- Code need to be changed for the google football environment:

/home/lizhichao/anaconda3/envs/coachpy36/lib/python3.6/site-packages/gfootball/env/wrappers.py

```
self.env._config.update({'render': True})
```

change to:

```
self.env._config.update({'render': self._original_render})
```

- Running command:

COACH_HOME=/home/lizhichao/bin/god/coach

export PYTHONPATH=$COACH_HOME:$PYTHONPATH
export DISPLAY=:0.0
export MESA_GL_VERSION_OVERRIDE=3.2
export MESA_GLSL_VERSION_OVERRIDE=150
coach -p $COACH_HOME/rl_coach/presets/Google_Football_keeper.py:graph_manager \
 -c --verbosity high \
--print_networks_summary --tensorboard \
 -s 300  \
 --nocolor