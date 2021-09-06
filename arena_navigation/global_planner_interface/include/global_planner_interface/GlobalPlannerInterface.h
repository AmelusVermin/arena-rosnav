//
// Created by johannes on 27.07.21.
//

#ifndef GLOBAL_PLANNER_INTERFACE_GLOBALPLANNERINTERFACE_H
#define GLOBAL_PLANNER_INTERFACE_GLOBALPLANNERINTERFACE_H

#include "ros/ros.h"
#include "tf2_ros/buffer.h"
#include "nav_core/base_global_planner.h"
#include <tf2_ros/transform_listener.h>
#include "global_planner_interface/MakeGlobalPlan.h"
#include "std_srvs/Empty.h"
#include <ros/console.h>

class GlobalPlannerInterface {

public:
    GlobalPlannerInterface();
    bool makeNewPlanCallback(global_planner_interface::MakeGlobalPlan::Request &req,
                             global_planner_interface::MakeGlobalPlan::Response &rep);

    bool resetCostmapCallback(std_srvs::Empty::Request& request,
                              std_srvs::Empty::Response& response);

private:
    void reset_costmap();
    void automatic_reset();

    pluginlib::ClassLoader<nav_core::BaseGlobalPlanner> bgp_loader_;
    boost::shared_ptr<nav_core::BaseGlobalPlanner> _global_planner;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;
    costmap_2d::Costmap2DROS* _costmap_ros;
    std::string _global_planner_type = std::string("navfn/NavfnROS");

    bool _reset_costmap_automatically = true;
    float _reset_costmap_interval = 3.0f;
    ros::Time _time_last_resetted;

    ros::ServiceServer _getGlobalPlan;
    ros::ServiceServer _resetCostmap;
};


#endif //GLOBAL_PLANNER_INTERFACE_GLOBALPLANNERINTERFACE_H