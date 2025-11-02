#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')
    package_share = get_package_share_directory('particle-filter-ros2')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pre_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_gazebo_pkg, 'launch', 'turtlebot3_house.launch.py')
            )
        )

    gz_topics_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_topics_bridge',
        output='screen',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    gz_service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_service_bridge',
        output='screen',
        arguments=[
            '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose@gz.msgs.Pose@gz.msgs.Boolean'
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', nav2_rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(pre_launch)

    ld.add_action(gz_topics_bridge)
    ld.add_action(gz_service_bridge)

    ld.add_action(rviz)
    ld.add_action(transform)

    return ld
