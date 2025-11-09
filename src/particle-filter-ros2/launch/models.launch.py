#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('particle-filter-ros2')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    start_rviz = LaunchConfiguration('start_rviz', default='true')

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

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_models',
        output='screen',
        arguments=['-d', nav2_rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(start_rviz)
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulated clock if true.'
        ),
        DeclareLaunchArgument(
            'start_rviz',
            default_value='true',
            description='Launch RViz alongside the model nodes.'
        ),
        motion_model,
        sensor_model,
        rviz,
    ])
