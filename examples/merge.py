# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution import UniformVehicleDistribution
from modules.runtime.ml.runtime_rl import RuntimeRL
from modules.runtime.ml.nn_state_observer import StateConcatenation
from modules.runtime.ml.action_wrapper import MotionPrimitives
from modules.runtime.ml.state_evaluator import GoalReached
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer

from modules.runtime.ml.dqn import DQN
import numpy as np
import csv, pathlib

params = ParameterServer(filename="modules/runtime/tests/data/highway_merging.json")
scenario_generation = UniformVehicleDistribution(num_scenarios=3, random_seed=1, params=params)
state_observer = StateConcatenation(params=params)
action_wrapper = MotionPrimitives(params=params)
evaluator = GoalReached(params=params)
viewer = MPViewer(params=params, x_range=[-30,30], y_range=[-20,40], follow_agent_id=True) #use_world_bounds=True) # 
viewer.draw_eval_goals = True

runtimerl = RuntimeRL(action_wrapper=action_wrapper, nn_observer=state_observer,
                evaluator=evaluator, step_time=0.1, viewer=viewer,
                scenario_generator=scenario_generation)

savePath = pathlib.Path(pathlib.Path.home() / 'barkout')
savePath.mkdir(exist_ok=True)


episodes = 500

dqn = DQN(20, 4, 32, 0.9, 0.9, 50, 2000)
runtimes = []
rewards = []

for episode in range(episodes): 
    state = runtimerl.reset()
    totalReward = 0
    runtime = 0
    for _ in range(0, 1000): 
        action = dqn.choose_action(state)
        # print(action, flush=True)
        next_state, reward, done, info = runtimerl.step(action)
        dqn.store_transition(state, action, reward, next_state)
        # runtimerl.render()
        totalReward += reward
        runtime += 1

        if dqn.memory_counter > 5:
			# env.render()
            dqn.learn()
    
        if info["success"] or done:
            runtimes.append(runtime)
            rewards.append(totalReward)
            print("State: {} \n Reward: {} \n Done {}, Info: {} \n \
                    ================================================="\
                    .format( next_state, totalReward, done, info), flush=True)
            break

        state = next_state

    with open(f'{savePath}/stats.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(runtimes)
        wr.writerow(rewards)

    dqn.save(f'{savePath}/model.pt')
