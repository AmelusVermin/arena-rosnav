#include "ros/ros.h"
#include "arena_services/AStar2.h"
#include "arena_path_search/astar2.h"


bool dummy_service(arena_services::DummyService::Request &req, arena_services::DummyService::Response &res){
    res.out = req.in + 1;
    return true;
}

int main(int argc, char **argv){
    ros::init(argc, argv, "arena_services");
    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("dummy_service", dummy_service);
    ROS_INFO("Ready to provide arena services");
    ros::spin();
    return 0;
}
