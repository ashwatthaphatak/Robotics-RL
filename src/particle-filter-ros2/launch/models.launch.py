#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('particle-filter-ros2')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    motion_model = Node(
        package='particle-filter-ros2',
        executable='motion_model',   # ← REQUIRED
        name='motion_model',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'motion.yaml'),
            {'use_sim_time': use_sim_time}
        ]
    )

    sensor_model = Node(
        package='particle-filter-ros2',
        executable='sensor_model',   # ← REQUIRED
        name='sensor_model',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'sensor.yaml'),
            {'use_sim_time': use_sim_time}
        ]
    )

    return LaunchDescription([
        motion_model,
        sensor_model
    ])
