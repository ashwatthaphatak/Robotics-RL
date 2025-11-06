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

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': os.path.join(package_share, 'maps', 'map.yaml'),
            'topic_name': 'map',
            'frame_id': 'map'
        }]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    pf_parameters = os.path.join(package_share, 'config', 'pf.yaml')
    particle_filter = Node(
        package='particle-filter-ros2',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[
            pf_parameters,
            {'use_sim_time': use_sim_time}
        ]
    )

    ld = LaunchDescription()
    ld.add_action(pre_launch)

    ld.add_action(gz_topics_bridge)
    ld.add_action(gz_service_bridge)

    ld.add_action(rviz)
    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)
    ld.add_action(static_tf)
    ld.add_action(particle_filter)

    return ld
