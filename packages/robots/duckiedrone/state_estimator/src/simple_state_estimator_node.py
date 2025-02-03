#!/usr/bin/env python3

import rospy

from duckietown.dtros import NodeType, DTROS
from sensor_msgs.msg import Range
from geometry_msgs.msg import Point, Pose, PoseWithCovariance
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


class SimpleStateEstimatorNode(DTROS):

    def __init__(self):
        super(SimpleStateEstimatorNode, self).__init__(
            node_name="state_estimator_node",
            node_type=NodeType.PERCEPTION
        )

        # publishers
        self._state_pub = rospy.Publisher("~state", Odometry, queue_size=1)
        # subscribers
        rospy.Subscriber("~altitude", Range, self._altitude_cb, queue_size=1)

    def _altitude_cb(self, msg):
        self._state_pub.publish(
            Odometry(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id="map",
                ),
                pose=PoseWithCovariance(
                    pose=Pose(
                        position=Point(
                            z=msg.range
                        )
                    )
                )
            )
        )


if __name__ == '__main__':
    node = SimpleStateEstimatorNode()
    rospy.spin()
