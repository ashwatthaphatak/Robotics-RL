#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    ExecuteProcess,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory('wall_following_ros2')
    tb3_gz = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    tb3_cartographer = get_package_share_directory('turtlebot3_cartographer')

    world_path = os.path.join(pkg_share, 'worlds', 'largemaze.world')
    rviz_config = os.path.join(nav2_bringup, 'rviz', 'nav2_default_view.rviz')

    src_dir = os.path.join(
        os.path.expanduser('~'),
        'ros2_ws', 'src', 'wall_following_ros2', 'src'
    )

    # Gazebo + GUI
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -s -v2 {world_path}'}.items()
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2'}.items(),
    )

    robot_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Spawn two TB3s at different corners of the maze
    spawn_tb3_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': '-2.0', 'y_pose': '-3.25'}.items()
    )
    spawn_tb3_2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': '2.0', 'y_pose': '3.25'}.items()
    )

    delayed_spawn_1 = TimerAction(period=5.0, actions=[spawn_tb3_1])
    delayed_spawn_2 = TimerAction(period=7.0, actions=[spawn_tb3_2])

    set_env_vars = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gz, 'models')
    )

    # NOTE: This uses the same bridge topics for both robots; adjust to per-robot
    # topics if your Gazebo world publishes distinct names per model.
    bridge_topics = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        output='screen'
    )

    bridge_service = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose@gz.msgs.Pose@gz.msgs.Boolean'
        ],
        output='screen'
    )

    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen',
    )

    # Two independent SARSA run nodes in different namespaces.
    # Ensure /robot1/scan,/robot1/cmd_vel and /robot2/scan,/robot2/cmd_vel are bridged.
    sarsa_1 = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(src_dir, 'sarsa_run.py'),
            '--max_linear', '0.5',
            '--ros-args', '-r', '__ns:=/robot1'
        ],
        output='screen'
    )
    sarsa_2 = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(src_dir, 'sarsa_run.py'),
            '--max_linear', '0.5',
            '--ros-args', '-r', '__ns:=/robot2'
        ],
        output='screen'
    )

    # Cartographer for mapping (single instance)
    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_cartographer, 'launch', 'cartographer.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )
    delayed_cartographer = TimerAction(period=10.0, actions=[cartographer_launch])

    # Static transform between synthetic frames for stitching maps later.
    tf_between_robots = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
            '4.0', '4.0', '0.0',   # x y z offset (tune to your initial spacing)
            '0', '0', '0',         # roll pitch yaw
            'robot1_root', 'robot2_root'
        ],
        output='screen'
    )

    ld = LaunchDescription()
    for act in [
        set_env_vars,
        gzserver,
        gzclient,
        robot_state_pub,
        delayed_spawn_1,
        delayed_spawn_2,
        bridge_topics,
        bridge_service,
        rviz,
        sarsa_1,
        sarsa_2,
        delayed_cartographer,
        tf_between_robots,
    ]:
        ld.add_action(act)

    return ld

