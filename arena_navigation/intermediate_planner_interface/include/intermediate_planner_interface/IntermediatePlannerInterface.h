#ifndef INTERMEDIATE_PLANNER_INTERFACE_INTERMEDIATEPLANNERINTERFACE_H
#define INTERMEDIATE_PLANNER_INTERFACE_INTERMEDIATEPLANNERINTERFACE_H

#include "ros/ros.h"
#include "intermediate_planner_interface/MakeIntermediateGoal.h"
#include "arena_plan_manager/plan_collector.h"

class IntermediatePlannerInterface{
public:
    IntermediatePlannerInterface();
    bool makeNewIntermediateGoal(intermediate_planner_interface::MakeIntermediateGoal::Request &req,
                                 intermediate_planner_interface::MakeIntermediateGoal::Response &rep);
private:
    PlanCollector::Ptr _plan_collector;
    ros::ServiceServer _getIntermediateGoal;
};

#endif //INTERMEDIATE_PLANNER_INTERFACE_INTERMEDIATEPLANNERINTERFACE_H