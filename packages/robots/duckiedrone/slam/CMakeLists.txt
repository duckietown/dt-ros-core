cmake_minimum_required(VERSION 2.8.3)
project(slam)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  duckietown_msgs
  std_msgs
  rospy
)

catkin_python_setup()

catkin_package()

# Install Python scripts
catkin_install_python(PROGRAMS
  src/slam_node/slam_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
