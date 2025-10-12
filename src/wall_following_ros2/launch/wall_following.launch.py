#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # ---------------- Launch arguments ----------------
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-3.25')

    mode        = LaunchConfiguration('mode',        default='train')       # train | test
    algorithm   = LaunchConfiguration('algorithm',   default='q_learning')  # q_learning | sarsa
    reward_mode = LaunchConfiguration('reward_mode', default='sparse')      # sparse | shaped
    goal_x      = LaunchConfiguration('goal_x',      default='2.6')
    goal_y      = LaunchConfiguration('goal_y',      default='3.1')
    goal_r      = LaunchConfiguration('goal_r',      default='0.5')

    declare_args = [
        DeclareLaunchArgument('mode',        default_value='train'),
        DeclareLaunchArgument('algorithm',   default_value='q_learning'),
        DeclareLaunchArgument('reward_mode', default_value='sparse'),
        DeclareLaunchArgument('goal_x',      default_value='2.6'),
        DeclareLaunchArgument('goal_y',      default_value='3.1'),
        DeclareLaunchArgument('goal_r',      default_value='0.5'),
    ]

    # ---------------- Paths ----------------
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')
    world = os.path.join(get_package_share_directory('wall_following_ros2'), 'worlds', 'largemaze.world')

    # ---------------- Gazebo + robot ----------------
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -s -v2 ', world]}.items()
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2 '}.items()
    )

    robot_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items()
    )

    set_env = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')
    )

    # ---------------- Bridges ----------------
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

    transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # ---------------- RL node ----------------
    wall_follower = ExecuteProcess(
        cmd=[
            'python3', '/home/aaphatak/ros2_ws/src/wall_following_ros2/src/q_td_run.py',
            '--mode', mode,
            '--algorithm', algorithm,
            '--reward_mode', reward_mode,
            '--goal_x', goal_x,
            '--goal_y', goal_y,
            '--goal_r', goal_r
        ],
        output='screen'
    )

    ld = LaunchDescription()
    for a in declare_args:
        ld.add_action(a)
    ld.add_action(gzserver)
    ld.add_action(gzclient)
    ld.add_action(spawn_tb3)
    ld.add_action(robot_state_pub)
    ld.add_action(set_env)
    ld.add_action(gz_topics_bridge)
    ld.add_action(gz_service_bridge)
    ld.add_action(rviz)
    ld.add_action(transform)
    ld.add_action(wall_follower)
    return ld
