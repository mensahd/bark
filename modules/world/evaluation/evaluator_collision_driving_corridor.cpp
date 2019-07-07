// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "modules/world/evaluation/evaluator_collision_driving_corridor.hpp"
#include "modules/world/world.hpp"

namespace modules
{
namespace world
{
namespace evaluation
{

EvaluationReturn EvaluatorCollisionDrivingCorridor::Evaluate(const world::World &world) const
{
  // checks collision with inner and outer line of driving corridor
  // assumption: agent is initially inside the local map and the world steps are fine
  //   to prevent the agent from "jumping" outside the driving corridor

  modules::geometry::Polygon poly_agent;
  bool colliding = false;
  modules::geometry::Line lane_inner;
  modules::geometry::Line lane_outer;

  for (auto agent : world.get_agents())
  {
    poly_agent = agent.second->GetPolygonFromState(agent.second->get_current_state());

    lane_inner = agent.second->get_local_map()->get_horizon_driving_corridor().get_inner();
    lane_outer = agent.second->get_local_map()->get_horizon_driving_corridor().get_outer();

    if (Collide(poly_agent, lane_outer))// || Collide(poly_agent, lane_inner)) # Allow the agent to cross the centre line
    {
      colliding = true;
      break;
    }
  }
  return colliding;
};

} // namespace collision
} // namespace world
} // namespace modules