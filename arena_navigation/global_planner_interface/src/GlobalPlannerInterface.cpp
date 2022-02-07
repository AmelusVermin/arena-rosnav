//
// Created by johannes on 27.07.21.
//

#include "../include/global_planner_interface/GlobalPlannerInterface.h"
#include <string.h>
#include <iostream>
#include <fstream>

using namespace std;

GlobalPlannerInterface::GlobalPlannerInterface() : bgp_loader_("nav_core", "nav_core::BaseGlobalPlanner"),
                                                   tfBuffer(ros::Duration(10)),
                                                   tfListener(tfBuffer) {
    
    // create   ROS handle
    ros::NodeHandle nh("~");
    
    // prepare some parameter values according to namespace
    string static_layer_topic = "";
    string scan_topic = "";
    nh.getParam("global_costmap/static_layer/map_topic", static_layer_topic);
    nh.getParam("global_costmap/obstacle_layer/scan/topic", scan_topic);
    string ns = ros::this_node::getNamespace();
    string ns_prefix = ns.compare("") ? "/" + ns + "/" : "/"; 
    nh.param("global_costmap/static_layer/map_topic", ns_prefix + static_layer_topic);
    nh.param("global_costmap/obstacle_layer/scan/topic", ns_prefix + scan_topic);
    
    // create Costmap
    _costmap_ros = new costmap_2d::Costmap2DROS("global_costmap", tfBuffer);
    _costmap_ros->start();
    _time_last_resetted = ros::Time::now();

    nh.param("global_planner_type", _global_planner_type, _global_planner_type);
    nh.param("reset_costmap_automatically", _reset_costmap_automatically, _reset_costmap_automatically);
    nh.param("reset_costmap_interval", _reset_costmap_interval, _reset_costmap_interval);
    try {
        _global_planner = bgp_loader_.createInstance(_global_planner_type);
        _global_planner->initialize(bgp_loader_.getName(_global_planner_type), _costmap_ros);
    } catch (const pluginlib::PluginlibException &ex) {
        ROS_FATAL_STREAM("Failed to create global planner " << _global_planner_type << "!");
        exit(1);
    }
    // create services
    _getGlobalPlan = nh.advertiseService("makeGlobalPlan", &GlobalPlannerInterface::makeNewPlanCallback, this);
    _getGlobalPlanFull = nh.advertiseService("makeGlobalPlanFull", &GlobalPlannerInterface::makeNewPlanFullCallback, this);

    _resetCostmap = nh.advertiseService("resetCostmap", &GlobalPlannerInterface::resetCostmapCallback, this);
}

bool GlobalPlannerInterface::makeNewPlanCallback(global_planner_interface::MakeGlobalPlan::Request &req,
                                                 global_planner_interface::MakeGlobalPlan::Response &rep) {
    geometry_msgs::PoseStamped robot_pose;
    _costmap_ros->getRobotPose(robot_pose);

    vector<geometry_msgs::PoseStamped> global_plan;
    rep.success = true;
    if (!_global_planner->makePlan(robot_pose, req.goal, global_plan) || global_plan.empty()) {
        reset_costmap();
        if (!_global_planner->makePlan(robot_pose, req.goal, global_plan) || global_plan.empty()) {
            ROS_WARN("Couldn't find global plan!");
            rep.global_plan.poses = {robot_pose, req.goal};
            rep.success = false;
            // call automatic reset and return early to avoid overwriting the response with empty plan
            automatic_reset();
            return true;
        }
    }

    automatic_reset();

    
    rep.global_plan.poses.resize(global_plan.size());
    for (unsigned int i = 0; i < global_plan.size(); ++i) {
        rep.global_plan.poses[i] = global_plan[i];
    }
    return true;
}

bool GlobalPlannerInterface::makeNewPlanFullCallback(global_planner_interface::MakeGlobalPlanFull::Request &req,
                                                 global_planner_interface::MakeGlobalPlanFull::Response &rep) {

    vector<geometry_msgs::PoseStamped> global_plan; 
    rep.success = true;
    if (!_global_planner->makePlan(req.start, req.goal, global_plan) || global_plan.empty()) {
        reset_costmap();
        if (!_global_planner->makePlan(req.start, req.goal, global_plan) || global_plan.empty()) {
            ROS_WARN("Couldn't find global plan!");
            rep.global_plan.poses = {};
            rep.success = false;
            // call automatic reset and return early to avoid overwriting the response with empty plan
            automatic_reset();
            return true;
        }
    }

    automatic_reset();

    
    rep.global_plan.poses.resize(global_plan.size());
    for (unsigned int i = 0; i < global_plan.size(); ++i) {
        rep.global_plan.poses[i] = global_plan[i];
    }
    return true;
}


void GlobalPlannerInterface::reset_costmap() {
    boost::unique_lock<costmap_2d::Costmap2D::mutex_t> lock(*(_costmap_ros->getCostmap()->getMutex()));
    _costmap_ros->resetLayers();
    _costmap_ros->updateMap();
    _time_last_resetted = ros::Time::now();
}

bool GlobalPlannerInterface::resetCostmapCallback(global_planner_interface::ResetCostmap::Request &request,
                                                  global_planner_interface::ResetCostmap::Response &response) {
    reset_costmap();
    //response.success = true;
    return true;
}

void GlobalPlannerInterface::automatic_reset() {
    ros::Time current_time = ros::Time::now();
    ros::Duration time_diff = (current_time - _time_last_resetted);
    bool automatic_reset = _reset_costmap_automatically && time_diff.toSec() >= _reset_costmap_interval;

    if (automatic_reset) {
        reset_costmap();
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "global_planner");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    auto *local_planner_node = new GlobalPlannerInterface();
    ROS_INFO("Started global planner interface.");
    ros::spin();

    return 0;
}
