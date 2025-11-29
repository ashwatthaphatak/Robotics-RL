#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    # Default spawn matches TurtleBot3 house world
    # spawn used when the map was recorded.
    spawn_x = LaunchConfiguration('spawn_x', default='-2.0')
    spawn_y = LaunchConfiguration('spawn_y', default='5.5')
    spawn_yaw = LaunchConfiguration('spawn_yaw', default='0.0')
    # Default model name in Gazebo is the raw
    # TURTLEBOT3_MODEL (e.g., "burger"), not
    # "turtlebot3_burger".
    entity_name = LaunchConfiguration('entity_name', default='burger')

    pkg = get_package_share_directory('particle-filter-ros2')
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')

    # Allow overriding map via `map:=/full/path/to/map.yaml`
    default_map = os.path.join(pkg, 'maps', 'map.yaml')
    map_yaml = LaunchConfiguration('map', default=default_map)
    a_star_params = os.path.join(pkg, 'config', 'a_star.yaml')

    # Gazebo world with TurtleBot3 (house)
    gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_pkg, 'launch', 'turtlebot3_house.launch.py')
        ),
        # Pass spawn position into the underlying
        # world launch so Gazebo spawns the robot
        # at the same location we use elsewhere.
        launch_arguments={
            'x_pose': spawn_x,
            'y_pose': spawn_y,
        }.items(),
    )

    # Static TF map->odom so the `map` frame
    # always exists for RViz and other nodes,
    # even before the planner starts.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )

    # ROS <-> GZ bridges for sim time, scan, and cmd_vel
    gz_topics_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_topics_bridge',
        output='screen',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            # Match TurtleBot3 Gazebo bridge config:
            # /cmd_vel uses TwistStamped on Jazzy.
            '/cmd_vel@geometry_msgs/msg/TwistStamped@gz.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
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

    # Map server + lifecycle manager
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': use_sim_time}]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )

    # RViz (Nav2 default view has Map and Goal tools configured)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', nav2_rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    pose_setter = Node(
        package='particle-filter-ros2',
        executable='set_gz_pose',
        name='initial_pose_setter',
        output='screen',
        parameters=[{
            'entity_name': entity_name,
            'x': spawn_x,
            'y': spawn_y,
            'z': 0.01,
            'yaw': spawn_yaw,
        }]
    )

    # A* planner node
    a_star = Node(
        package='particle-filter-ros2',
        executable='a_star_planner',
        name='a_star_planner',
        output='screen',
        parameters=[a_star_params, {'use_sim_time': use_sim_time}],
    )

    # Expose spawn and sim parameters as launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    declare_spawn_x = DeclareLaunchArgument(
        'spawn_x',
        default_value='-2.0',
        description='Initial robot x (world/map)'
    )
    declare_spawn_y = DeclareLaunchArgument(
        'spawn_y',
        default_value='-0.5',
        description='Initial robot y (world/map)'
    )
    declare_spawn_yaw = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='0.0',
        description='Initial robot yaw (rad)'
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_spawn_x)
    ld.add_action(declare_spawn_y)
    ld.add_action(declare_spawn_yaw)
    ld.add_action(gazebo_world)
    ld.add_action(static_tf)
    ld.add_action(gz_topics_bridge)
    ld.add_action(gz_service_bridge)
    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)
    ld.add_action(rviz)
    ld.add_action(pose_setter)
    ld.add_action(a_star)
    return ld
