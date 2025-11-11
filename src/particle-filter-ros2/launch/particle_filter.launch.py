#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import sys


def generate_launch_description():
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')
    
    # Get particle filter package directory
    particle_filter_pkg = get_package_share_directory('particle-filter-ros2')
    maps_dir = os.path.join(particle_filter_pkg, 'maps')
    map_yaml = os.path.join(maps_dir, 'map.yaml')
    scripts_dir = os.path.join(particle_filter_pkg, 'scripts')
    particle_filter_script = os.path.join(scripts_dir, 'particle_filter.py')
    rviz_config = os.path.join(particle_filter_pkg, 'rviz', 'particle_filter.rviz')

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
        output='screen'
    )

    # rviz = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', rviz_config],
    #     output='screen'
    # )

    transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # MAP SERVER NODE
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_yaml,
            'use_sim_time': use_sim_time
        }]
    )
    
    # LIFECYCLE MANAGER
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

    # PARTICLE FILTER NODE - Run script from scripts directory
    particle_filter_node = Node(
        package='particle-filter-ros2',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    ld = LaunchDescription()
    ld.add_action(pre_launch)
    ld.add_action(gz_topics_bridge)
    ld.add_action(gz_service_bridge)
    ld.add_action(rviz)
    ld.add_action(transform)
    ld.add_action(map_server_node)
    ld.add_action(lifecycle_manager)
    ld.add_action(particle_filter_node)

    return ld