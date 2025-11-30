#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_wall_follow = get_package_share_directory('wall_following_ros2')
    pkg_map_merge = get_package_share_directory('multirobot_map_merge')

    base_two_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_wall_follow,
                'launch',
                'two_robot_sarsa_with_cartographer.launch.py'
            )
        )
    )

    map_merge_config = os.path.join(
        pkg_map_merge,
        'config',
        'params.yaml'
    )

    # Merge the per-robot Cartographer maps into a single /merged_map.
    map_merge = Node(
        package='multirobot_map_merge',
        executable='map_merge',
        name='map_merge',
        output='screen',
        parameters=[
            map_merge_config,
            {
                'use_sim_time': True,
                'world_frame': 'map',
                'merged_map_topic': 'merged_map',
                'robot_map_topic': 'map',
                'robot_map_updates_topic': 'map_updates',
                'known_init_poses': True,
                '/robot1/map_merge/init_pose_x': -2.0,
                '/robot1/map_merge/init_pose_y': -3.25,
                '/robot1/map_merge/init_pose_z': 0.0,
                '/robot1/map_merge/init_pose_yaw': 0.0,
                '/robot2/map_merge/init_pose_x': 2.0,
                '/robot2/map_merge/init_pose_y': 3.25,
                '/robot2/map_merge/init_pose_z': 0.0,
                '/robot2/map_merge/init_pose_yaw': 0.0,
            }
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )

    ld = LaunchDescription()
    ld.add_action(base_two_robot)
    ld.add_action(map_merge)
    return ld
