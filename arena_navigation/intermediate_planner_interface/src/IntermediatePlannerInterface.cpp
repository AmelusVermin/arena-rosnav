#include "ros/ros.h"
#include "../include/intermediate_planner_interface/IntermediatePlannerInterface.h"
#include "intermediate_planner_interface/MakeIntermediateGoal.h"
#include "arena_plan_manager/plan_collector.h"
#include "arena_plan_manager/robot_state.h"

IntermediatePlannerInterface::IntermediatePlannerInterface(){
    // create   ROS handle
    ros::NodeHandle nh("~");
    // create PlanCollector which is responsible for creating the subgoal
    _plan_collector.reset(new PlanCollector());
    _plan_collector->initSubgoalModule(nh);
    // register service
    _getIntermediateGoal = nh.advertiseService("makeSubgoal", &IntermediatePlannerInterface::makeNewIntermediateGoal, this);
}

bool IntermediatePlannerInterface::makeNewIntermediateGoal(intermediate_planner_interface::MakeIntermediateGoal::Request &req,
                                                           intermediate_planner_interface::MakeIntermediateGoal::Response &rep) 
{
    std::cout<< "plan length: " << req.global_plan.poses.size() << std::endl;
    std::cout << req.robot_pos.pose << std::endl;
    // prepare start and end position of path
    RobotStatePtr curr_state;
    RobotStatePtr end_state;
    curr_state.reset(new RobotState(req.robot_pos));
    // last element of global plan should be the goal
    end_state.reset(new RobotState(req.global_plan.poses.back().pose));
    // create subgoal
    ROS_INFO("get subgoal");
    _plan_collector->generate_subgoal(curr_state, end_state, req.global_plan, 1.0, 1.0);
    // store subgoal in response
    rep.subgoal = _plan_collector->subgoal_;
    rep.subgoal.header.frame_id = req.global_plan.header.frame_id;
    ROS_INFO("got subgoal");
    std::cout << rep.subgoal << std::endl;
    return true;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "global_planner");
    auto *intermediate_planner_node = new IntermediatePlannerInterface();
    ROS_INFO("Started intermediate planner interface.");
    ros::spin();

    return 0;
}