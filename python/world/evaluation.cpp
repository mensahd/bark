// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "evaluation.hpp"
#include "modules/world/world.hpp"
#include "modules/world/evaluation/evaluator_goal_reached.hpp"
#include "modules/world/evaluation/evaluator_collision_agents.hpp"
#include "modules/world/evaluation/evaluator_collision_driving_corridor.hpp"

namespace py = pybind11;
using namespace modules::world::evaluation;


void python_evaluation(py::module m)
{
  py::class_<BaseEvaluator,
             PyBaseEvaluator,
             EvaluatorPtr>(m, "BaseEvaluator")
      .def(py::init<>());

  py::class_<EvaluatorGoalReached, BaseEvaluator, 
      std::shared_ptr<EvaluatorGoalReached> >(m, "EvaluatorGoalReached")
      .def(py::init<const AgentId&>())
      .def("__repr__", [](const EvaluatorGoalReached &g) {
        return "bark.world.evaluation.EvaluatorGoalReached";
      });

    py::class_<EvaluatorCollisionAgents, BaseEvaluator, 
      std::shared_ptr<EvaluatorCollisionAgents> >(m, "EvaluatorCollisionAgents")
      .def(py::init<>())
      .def("__repr__", [](const EvaluatorGoalReached &g) {
        return "bark.world.evaluation.EvaluatorGoalReached";
      });

        py::class_<EvaluatorCollisionDrivingCorridor, BaseEvaluator, 
      std::shared_ptr<EvaluatorCollisionDrivingCorridor> >(m, "EvaluatorCollisionDrivingCorridor")
      .def(py::init<>())
      .def("__repr__", [](const EvaluatorGoalReached &g) {
        return "bark.world.evaluation.EvaluatorCollisionDrivingCorridor";
      });
}