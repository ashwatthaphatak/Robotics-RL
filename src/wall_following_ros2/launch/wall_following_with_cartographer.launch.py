#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_wall_follow = get_package_share_directory('wall_following_ros2')
    pkg_tb3_cartographer = get_package_share_directory('turtlebot3_cartographer')

    wall_follow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_wall_follow, 'launch', 'wall_following.launch.py')
        )
    )

    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_cartographer, 'launch', 'cartographer.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    delayed_cartographer = TimerAction(
        period=8.0,
        actions=[cartographer_launch],
    )

    ld = LaunchDescription()
    ld.add_action(wall_follow_launch)
    ld.add_action(delayed_cartographer)
    return ld

