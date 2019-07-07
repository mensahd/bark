from bark.world.evaluation import *
from modules.runtime.commons.parameters import ParameterServer

class StateEvaluator:
    def get_evaluation(self, world):
        return # reward, done, info

    def reset(self, world, agents_to_evaluate):
        return # world


class GoalReached(StateEvaluator):
    def __init__(self, params=ParameterServer()):
        self.params = params
        self.goal_reward = params["Runtime"]["RL"]["StateEvaluator"]["GoalReward", "The reward given for goals", 0.01]
        self.collision_reward = params["Runtime"]["RL"]["StateEvaluator"]["CollisionReward", "The (negative) \
                                                     reward given for collisions", -1]

    def get_evaluation(self, world):
        eval_results = world.evaluate()
        agentCollision = eval_results["collision_agents"].split(',')
        collision = (str(world.controlled_agent) in agentCollision) or eval_results["collision_driving_corridor"]
        success = eval_results["success"]
        reward = collision * self.collision_reward + success * self.goal_reward
        done = success or collision
        info = {"success": success, "collision_agents": collision, \
                 "collision_driving_corridor": eval_results["collision_driving_corridor"]}
        return reward, done, info
        
    def reset(self, world, agents_to_evaluate):
        if len(agents_to_evaluate) != 1:
            raise ValueError("Invalid number of agents provided for GoalReached \
                        evaluation, number= {}".format(len(agents_to_evaluate)))
        evaluator1 = EvaluatorGoalReached(agents_to_evaluate[0])
        evaluator2 = EvaluatorCollisionAgents()
        evaluator3 = EvaluatorCollisionDrivingCorridor()

        world.add_evaluator("success", evaluator1)
        world.add_evaluator("collision_agents", evaluator2)
        world.add_evaluator("collision_driving_corridor", evaluator3)

        return world
        

