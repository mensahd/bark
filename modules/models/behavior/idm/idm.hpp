// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef MODULES_MODELS_BEHAVIOR_IDM_HPP_
#define MODULES_MODELS_BEHAVIOR_IDM_HPP_

#include "modules/models/behavior/behavior_model.hpp"
#include "modules/world/world.hpp"

namespace modules {
namespace models {
namespace behavior {

using dynamic::Trajectory;
using world::objects::AgentId;
using world::ObservedWorld;

class IDM : public BehaviorModel {
 public:
  explicit IDM(commons::Params *params) :
    BehaviorModel(params) {}

  virtual ~IDM() {}

  double IDMModel(double v_ego, double v_other, double s);

  Trajectory Plan(float delta_time,
                 const ObservedWorld& observed_world);

  virtual BehaviorModel *Clone() const;
};

inline BehaviorModel *IDM::Clone() const {
  return new IDM(*this);
}

}  // namespace behavior
}  // namespace models
}  // namespace modules

#endif  // MODULES_MODELS_BEHAVIOR_IDM_HPP_
