// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef MODULES_WORLD_OPENDRIVE_COMMONS_HPP_
#define MODULES_WORLD_OPENDRIVE_COMMONS_HPP_

#include <inttypes.h>
#include <string>
#include <vector>

#include <boost/geometry.hpp>
#include "modules/geometry/line.hpp"

namespace modules {
namespace world {
namespace opendrive {

using LaneId = uint32_t;
using LanePosition = int32_t;
using RoadId = uint32_t;

struct LinkInfo {
  LinkInfo() : position_(1000000), type_("") {}
  LinkInfo(const LanePosition& pos, const std::string& type) : position_(pos), type_(type) {}
  LanePosition position_;
  std::string type_;
};

struct Link {
  Link() : predecessor_(), successor_() {}
  Link(const LinkInfo& predecessor, const LinkInfo& successor) : predecessor_(predecessor), successor_(successor) {}
  LinkInfo predecessor_;
  LinkInfo successor_;
  //! getter
  LinkInfo get_predecessor() const { return predecessor_; }
  LinkInfo get_successor() const { return successor_; }
  void set_predecessor(const LinkInfo& info) { predecessor_ = info; }
  void set_successor(const LinkInfo& info) { successor_ = info; }
};

inline std::string print(const Link &l) {
  std::stringstream ss;
  ss << "Link.predecessor: " << l.predecessor_.position_ << "of type" << l.predecessor_.type_ << "; ";
  ss << "Link.successor: " << l.successor_.position_ << "of type" << l.successor_.type_ << std::endl;
  return ss.str();
}

struct LaneOffset {
  float a, b, c, d;
};



inline float polynom(float x, float a, float b, float c, float d) {
  return a + b * x + c * x * x + d * x * x * x;
}

struct LaneLink {
  int from_id;
  int to_id;
};

using LaneLinks = std::vector<LaneLink>;

struct Connection {
  void add_lane_link(LaneLink link) { lane_links_.push_back(link); }
  LaneLinks get_lane_links() const { return lane_links_; }
  uint32_t id_;
  uint32_t incoming_road_;
  uint32_t connecting_road_;
  LaneLinks lane_links_;
};

enum LaneType {
  NONE = 0,
  DRIVING = 1,
  //STOP = 2,
  //SHOULDER = 3,
  BIKING = 4,
  SIDEWALK = 5,
  BORDER = 6,
  /*RESTRICTED = 7,
  PARKING = 8,
  BIDIRECTIONAL = 9,
  MEDIAN = 10,
  SPECIAL1 = 11,
  SPECIAL2 = 12,
  SPECIAL3 = 13,
  ROAD_WORKS = 14,
  TRAM = 15,
  RAIL = 16,
  ENTRY = 17,
  EXIT = 18,
  OFF_RAMP = 19,
  ON_RAMP = 20,
  CONNECTING_RAMP = 21,
  BUS = 22,
  TAXI = 23,
  HOV = 24
  */
};

struct RoadMark {
  std::string type_;
  std::string color_;
  float width_;
};

inline std::string print(const RoadMark &r) {
  std::stringstream ss;
  ss << "RoadMark: type: " << r.type_ << ", id: " << r.color_ << ", width: " << r.width_ << std::endl;
  return ss.str();
}

struct LaneWidth {
  float s_start;
  float s_end;
  LaneOffset off;
};


inline geometry::Line create_line_with_offset_from_line(geometry::Line previous_line, int id, LaneWidth lane_width_current_lane, float s_inc = 0.5f) {

  namespace bg = boost::geometry;

  LaneOffset off = lane_width_current_lane.off;

  float s = lane_width_current_lane.s_start;
  float scale = 0.0f;
  geometry::Line tmp_line;
  geometry::Point2d normal(0.0f, 0.0f);
  int sign = id > 0 ? -1 : 1;

  // TODO(fortiss): check if sampling does work with relative s, probably not
  if (off.b != 0.0f || off.c != 0.0f || off.d != 0.0f || (lane_width_current_lane.s_end - lane_width_current_lane.s_start) != 1.0) {
    for (; s < lane_width_current_lane.s_end; s += s_inc) {
      geometry::Point2d point = get_point_at_s(previous_line, s);
      normal = get_normal_at_s(previous_line, s);
      scale = -sign * polynom(s, off.a, off.b, off.c, off.d);
      tmp_line.add_point(geometry::Point2d(bg::get<0>(point) + scale * bg::get<0>(normal),
                                  bg::get<1>(point) + scale * bg::get<1>(normal)));
    }

    // fill last point if increment does not match
    double delta_s = fabs(lane_width_current_lane.s_end-s);
    if(delta_s>0.0){
      geometry::Point2d point = get_point_at_s(previous_line, lane_width_current_lane.s_end);
      normal = get_normal_at_s(previous_line, lane_width_current_lane.s_end);
      scale = -sign * polynom(lane_width_current_lane.s_end, off.a, off.b, off.c, off.d);
      tmp_line.add_point(geometry::Point2d(bg::get<0>(point) + scale * bg::get<0>(normal),
                                  bg::get<1>(point) + scale * bg::get<1>(normal)));
    }
  } else {
      for (uint32_t i = 0; i < previous_line.obj_.size() - 1; i++) {
        normal = get_normal_at_s(previous_line, previous_line.s_[i]);
        scale = -sign * polynom(s, off.a, off.b, off.c, off.d);
        tmp_line.add_point(geometry::Point2d(bg::get<0>(previous_line.obj_[i]) + scale * bg::get<0>(normal),
                                  bg::get<1>(previous_line.obj_[i]) + scale * bg::get<1>(normal)));
        s += geometry::distance(previous_line.obj_[i + 1], previous_line.obj_[i]);
      }
      // add last point
      normal = get_normal_at_s(previous_line, previous_line.s_[previous_line.obj_.size() - 1]);
      int size = previous_line.obj_.size() - 1;
      float length = bg::length(previous_line.obj_);
      scale = -sign * polynom(length, off.a, off.b, off.c, off.d);
      tmp_line.add_point(geometry::Point2d(bg::get<0>(previous_line.obj_[size]) + scale * bg::get<0>(normal),
                                bg::get<1>(previous_line.obj_[size]) + scale * bg::get<1>(normal)));
  }

  return tmp_line;
}

//using LaneWidths = std::vector<LaneWidth>;

}  // namespace opendrive
}  // namespace world
}  // namespace modules

#endif  // MODULES_WORLD_OPENDRIVE_COMMONS_HPP_
