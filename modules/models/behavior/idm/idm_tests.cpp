// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#include <Eigen/Core>
#include "gtest/gtest.h"

#include "modules/geometry/polygon.hpp"
#include "modules/geometry/line.hpp"
#include "modules/geometry/commons.hpp"
#include "modules/models/dynamic/single_track.hpp"
#include "modules/models/dynamic/integration.hpp"
#include "modules/models/behavior/idm/idm.hpp"
#include "modules/commons/params/default_params.hpp"

TEST(idm_test, equations) {
  using modules::models::behavior::IDM;
  using modules::commons::DefaultParams;

  DefaultParams params;
  IDM idm(&params);
  // the other vehicle is slower by 1 m/s
  std::cout << idm.IDMModel(10.0, 9.0, 25.0) << std::endl;
  // the other vehicle is faster by 1 m/s
  std::cout << idm.IDMModel(10.0, 11.0, 25.0) << std::endl;
  // the other vehicle is faster by 2 m/s
  std::cout << idm.IDMModel(10.0, 12.0, 25.0) << std::endl;

  // same but smaller s
  std::cout << idm.IDMModel(10.0, 9.0, 10.0) << std::endl;
  std::cout << idm.IDMModel(10.0, 11.0, 10.0) << std::endl;

  // edge case
  std::cout << idm.IDMModel(0.0, 0.0, 0.0) << std::endl;


}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
