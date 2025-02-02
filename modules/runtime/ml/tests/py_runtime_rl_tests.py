# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution import UniformVehicleDistribution
from modules.runtime.ml.runtime_rl import RuntimeRL
from modules.runtime.ml.nn_state_observer import StateConcatenation
from modules.runtime.ml.action_wrapper import MotionPrimitives
from modules.runtime.ml.state_evaluator import GoalReached
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
import numpy as np


class RuntimeRLTests(unittest.TestCase):
    def test_motion_primitives_concat_state(self):
        params = ParameterServer(filename="modules/runtime/tests/data/highway_merging.json")
        scenario_generation = UniformVehicleDistribution(num_scenarios=3, random_seed=0, params=params)
        state_observer = StateConcatenation(params=params)
        action_wrapper = MotionPrimitives(params=params)
        evaluator = GoalReached(params=params)
        viewer = MPViewer(params=params, x_range=[-30,30], y_range=[-20,40], follow_agent_id=True) #use_world_bounds=True) # 

        runtimerl = RuntimeRL(action_wrapper=action_wrapper, nn_observer=state_observer,
                        evaluator=evaluator, step_time=0.05, viewer=viewer,
                        scenario_generator=scenario_generation)


        for _ in range(0,5): # run 5 scenarios in a row, repeating after 3
            nn_state = runtimerl.reset()
            i = 0;
            actionsteps = []
            for _ in range(0, 1000): # run each scenario for 1000 steps
                actionstep = action_wrapper.action_space.sample()
                next_nn_state, reward, done, info = runtimerl.step(actionstep)
                actionsteps.append(actionstep)
                i += 1
                runtimerl.render()
                if info["success"] or done:
                    print("Finish in ", i)
                    print("Actions taken:", actionsteps)
                    print("State: {} \n Reward: {} \n Done {}, Info: {} \n \
                         =================================================".format( next_nn_state, reward, done, info))
                    break
                
        params.save(filename="highway_merging_written.json")
if __name__ == '__main__':
    unittest.main()