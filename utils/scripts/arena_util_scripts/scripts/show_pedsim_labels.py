#!/usr/bin/env python
import numpy as np
import rospy
import sys
import time
from nav_msgs.msg import OccupancyGrid
from pedsim_msgs.msg import AgentStates
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

class LabelPublisher:
    def __init__(self):
        rospy.init_node('label_publisher', anonymous=True)
        self.simulated_agents = None
        self.marker_pub = rospy.Publisher(rospy.get_namespace() + 'pedsim_labels', MarkerArray, queue_size=10)
        self.agent_states_sub = rospy.Subscriber(rospy.get_namespace() + "pedsim_simulator/simulated_agents", AgentStates, self.simulated_agents_callback)


    def simulated_agents_callback(self, agent_states_msg):
        self.simulated_agents = agent_states_msg


    def create_label_marker(self, id, x, y, text, scale = 1.0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "marker"
        marker.id = id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.MODIFY
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.0
        marker.scale.y = 0.0
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.points = []
        marker.text = text
        marker.lifetime = rospy.Duration.from_sec(0.1)
        return marker


    def publish_labels(self):
        while not rospy.is_shutdown():
            # wait for agent positions to be filled
            if self.simulated_agents is None:
                continue
            else:
                # iterate over agents
                markers = MarkerArray()
                for agent in self.simulated_agents.agent_states:
                    id = agent.id
                    pos_x = agent.pose.position.x
                    pos_y = agent.pose.position.y
                    text = agent.social_state
                    marker = self.create_label_marker(id, pos_x, pos_y, text)

                    markers.markers.append(marker)

                self.marker_pub.publish(markers)

            time.sleep(0.1)

    def run(self):
        self.publish_labels()


if __name__ == '__main__':
    publisher = LabelPublisher()
    try:
        publisher.run()
    except rospy.ROSInterruptException:
        pass