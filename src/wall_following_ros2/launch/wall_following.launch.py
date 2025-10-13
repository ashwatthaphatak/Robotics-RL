#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, AppendEnvironmentVariable, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ─────────────────────────────────────────────
    # Arguments
    # ─────────────────────────────────────────────
    mode = LaunchConfiguration('mode')
    algo = LaunchConfiguration('algorithm')
    reward = LaunchConfiguration('reward_mode')
    reset = LaunchConfiguration('reset_mode')

    declare_mode = DeclareLaunchArgument(
        'mode', default_value='train',
        description='Mode: train or test')

    declare_algorithm = DeclareLaunchArgument(
        'algorithm', default_value='q_learning',
        description='Algorithm: q_learning or sarsa')

    declare_reward = DeclareLaunchArgument(
        'reward_mode', default_value='shaped',
        description='Reward mode: shaped or sparse')

    declare_reset = DeclareLaunchArgument(
        'reset_mode', default_value='once',
        description='Reset behavior: once, always, or none')

    # ─────────────────────────────────────────────
    # Path setup
    # ─────────────────────────────────────────────
    pkg_share = get_package_share_directory('wall_following_ros2')
    tb3_gz = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup = get_package_share_directory('nav2_bringup')

    world_path = os.path.join(pkg_share, 'worlds', 'largemaze.world')
    rviz_config = os.path.join(nav2_bringup, 'rviz', 'nav2_default_view.rviz')

    # 🧭 Path to your source scripts
    src_dir = os.path.join(
        os.path.expanduser('~'),
        'ros2_ws', 'src', 'wall_following_ros2', 'src'
    )

    # ─────────────────────────────────────────────
    # Gazebo setup
    # ─────────────────────────────────────────────
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -s -v2 {world_path}'}.items()
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2'}.items()
    )

    robot_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': '-2.0', 'y_pose': '-3.25'}.items()
    )

    set_env_vars = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gz, 'models')
    )

    # ─────────────────────────────────────────────
    # ROS–Gazebo bridges + viz
    # ─────────────────────────────────────────────
    bridge_topics = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
             '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
             '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'],
        output='screen'
    )

    bridge_service = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose@gz.msgs.Pose@gz.msgs.Boolean'],
        output='screen'
    )

    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen'
    )

    tf_static = ExecuteProcess(
        cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
             '0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # ─────────────────────────────────────────────
    # RL Nodes — executed directly from src/
    # ─────────────────────────────────────────────
    q_train = ExecuteProcess(
        cmd=['python3', os.path.join(src_dir, 'q_td_train.py')],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", algo, "' == 'q_learning' and '", mode, "' == 'train'"
        ]))
    )

    q_test = ExecuteProcess(
        cmd=['python3', os.path.join(src_dir, 'q_td_run.py')],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", algo, "' == 'q_learning' and '", mode, "' == 'test'"
        ]))
    )

    sarsa_train = ExecuteProcess(
        cmd=['python3', os.path.join(src_dir, 'sarsa_train.py')],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", algo, "' == 'sarsa' and '", mode, "' == 'train'"
        ]))
    )

    sarsa_test = ExecuteProcess(
        cmd=['python3', os.path.join(src_dir, 'sarsa_run.py')],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", algo, "' == 'sarsa' and '", mode, "' == 'test'"
        ]))
    )

    # ─────────────────────────────────────────────
    # Launch Description
    # ─────────────────────────────────────────────
    ld = LaunchDescription()

    # Arguments
    for arg in [declare_mode, declare_algorithm, declare_reward, declare_reset]:
        ld.add_action(arg)

    # Core setup
    for act in [set_env_vars, gzserver, gzclient, spawn_tb3, robot_state_pub]:
        ld.add_action(act)

    # Bridges & visualization
    for act in [bridge_topics, bridge_service, rviz, tf_static]:
        ld.add_action(act)

    # RL scripts
    for act in [q_train, q_test, sarsa_train, sarsa_test]:
        ld.add_action(act)

    return ld
