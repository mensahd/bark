// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
#include <cmath>
#include "modules/models/behavior/idm/idm.hpp"
#include "modules/world/observed_world.hpp"
#include "modules/commons/params/default_params.hpp"

namespace modules {
namespace models {

double behavior::IDM::IDMModel(double v_ego, double v_other, double s) {
  // avoid division by zero
  if ( s == 0.0)
    return 0.0;
  double delta_v = v_ego - v_other;
  double v0, T, a, b, delta, s0, l;
  v0 = 10.0;  // TOD(@hart): Desired velocity
  T = this->get_params()->get_real("T",
                                   "Safe time headway",
                                   1.5);
  a = this->get_params()->get_real("a",
                                   "Maximum acceleration",
                                   0.73);
  b = this->get_params()->get_real("b",
                                   "Comfortable deceleration",
                                   1.67);
  delta = this->get_params()->get_real("delta",
                                       "Acceleration component",
                                       4.0);
  s0 = this->get_params()->get_real("s0",
                                    "Minimum distance",
                                    2.0);
  l = this->get_params()->get_real("l",
                                   "Vehicle length",
                                   5.0);
  double s_star = s0 + v_ego*T + v_ego*delta_v/(2*sqrt(a*b));
  double u = pow(v_ego/v0, delta);
  double w = pow(s_star/s, 2.0);
  double acceleration = a * (1.0 - u - w);
  return acceleration;
}

dynamic::Trajectory behavior::IDM::Plan(
    float delta_time,
    const world::ObservedWorld& observed_world) {

  using dynamic::StateDefinition::MIN_STATE_SIZE;
  using dynamic::StateDefinition::X_POSITION;
  using dynamic::StateDefinition::Y_POSITION;
  using dynamic::StateDefinition::VEL_POSITION;
  using dynamic::StateDefinition::TIME_POSITION;
  using dynamic::StateDefinition::THETA_POSITION;
  
  const int num_traj_time_points =
    this->get_params()->get_int("num_traj_time_points",
                                 "Vehicle length",
                                 100);
  dynamic::Trajectory traj(num_traj_time_points,
                           static_cast<int>(MIN_STATE_SIZE));

  auto const sample_time = delta_time / num_traj_time_points;

  dynamic::State ego_vehicle_state = observed_world.get_ego_state();

  // select state and get p0
  geometry::Point2d pose(ego_vehicle_state(X_POSITION),
                         ego_vehicle_state(Y_POSITION));


  geometry::Line line =
    observed_world.get_local_map()->get_driving_corridor().get_center();
  
  // check whether linestring is empty
  if (line.obj_.size() > 0) {
    float s_start = get_nearest_s(line, pose);
    double start_time = observed_world.get_world_time();
    float ego_velocity = ego_vehicle_state(VEL_POSITION);

    // v = s/t
    double run_time = start_time;
    for (int i = 0; i < traj.rows(); i++) {

      // TODO(@hart): fill these using the local map
      double velocity_other = 10.0;
      double s = 10.0;
      double acceleration_ego = IDMModel(ego_velocity, velocity_other, s);

      // TODO(@hart): we need tripple integrator model to determin del_s
      float del_s = ego_velocity * (run_time - start_time);

      // this pretty much should stay the same
      geometry::Point2d traj_point = get_point_at_s(line, s_start + del_s);
      float traj_angle = get_tangent_angle_at_s(line, s_start + del_s);
      traj(i, TIME_POSITION) = run_time;
      traj(i, X_POSITION) =
        boost::geometry::get<0>(traj_point);
      traj(i, Y_POSITION) =
        boost::geometry::get<1>(traj_point);
      traj(i, THETA_POSITION) = traj_angle;
      traj(i, VEL_POSITION) = ego_velocity;

      // increasing time
      run_time += sample_time;
    }
  }
  this->set_last_trajectory(traj);
  return traj;
}

}  // namespace models
}  // namespace modules
