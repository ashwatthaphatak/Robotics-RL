#!/usr/bin/env python3
"""
Launch two TurtleBot3 robots, each with its own namespace, state publisher,
SARSA runner, and Cartographer instance. Bridges are set per-robot so scans
and cmd_vel stay separated. Adjust GZ topic names if your simulator differs.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory('wall_following_ros2')
    tb3_gz = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    tb3_cartographer = get_package_share_directory('turtlebot3_cartographer')

    world_path = os.path.join(pkg_share, 'worlds', 'largemaze.world')
    rviz_config = os.path.join(nav2_bringup, 'rviz', 'nav2_default_view.rviz')
    model_sdf = os.path.join(tb3_gz, 'models', 'turtlebot3_burger', 'model.sdf')

    src_dir = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'wall_following_ros2', 'src')

    # Gazebo headless server + optional GUI
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -s -v2 {world_path}'}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2'}.items(),
    )

    set_env_vars = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gz, 'models'))

    # Spawn two robots with unique names
    spawn_r1 = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_sim', 'create', '-name', 'robot1', '-file', model_sdf, '-x', '-2.0', '-y', '-3.0', '-z', '0.01'],
        output='screen',
    )
    spawn_r2 = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_sim', 'create', '-name', 'robot2', '-file', model_sdf, '-x', '2.0', '-y', '3.0', '-z', '0.01'],
        output='screen',
    )
    delayed_spawn_r1 = TimerAction(period=3.0, actions=[spawn_r1])
    delayed_spawn_r2 = TimerAction(period=5.0, actions=[spawn_r2])

    # Per-robot robot_state_publisher with frame_prefix
    rsp_r1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'frame_prefix': 'robot1'}.items(),
    )
    rsp_r2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'frame_prefix': 'robot2'}.items(),
    )

    # Bridges: adjust gz_topic_name if your sim publishes different topic paths
    bridge_r1 = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            # LiDAR
            '/world/default/model/robot1/link/base_scan/sensor/hls_lfcd_lds/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            # Odometry
            '/model/robot1/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            # TF
            '/model/robot1/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            # Cmd vel
            '/model/robot1/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            # Clock
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        output='screen',
        additional_env={'ROS_NAMESPACE': '/robot1'},
    )

    bridge_r2 = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/world/default/model/robot2/link/base_scan/sensor/hls_lfcd_lds/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/model/robot2/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/model/robot2/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/model/robot2/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        output='screen',
        additional_env={'ROS_NAMESPACE': '/robot2'},
    )

    # SARSA runners
    sarsa_r1 = ExecuteProcess(
        cmd=['python3', os.path.join(src_dir, 'sarsa_run.py'), '--ros-args', '-r', '__ns:=/robot1'],
        output='screen',
    )
    sarsa_r2 = ExecuteProcess(
        cmd=['python3', os.path.join(src_dir, 'sarsa_run.py'), '--ros-args', '-r', '__ns:=/robot2'],
        output='screen',
    )

    # Cartographer per robot (namespaced)
    carto_r1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_cartographer, 'launch', 'cartographer.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'use_rviz': 'false'}.items(),
    )
    carto_r2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_cartographer, 'launch', 'cartographer.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'use_rviz': 'false'}.items(),
    )

    # Delay Cartographer to allow robots to spawn & TF to settle
    delayed_carto_r1 = TimerAction(period=8.0, actions=[carto_r1])
    delayed_carto_r2 = TimerAction(period=9.0, actions=[carto_r2])

    # RViz (single instance)
    rviz = ExecuteProcess(cmd=['rviz2', '-d', rviz_config], output='screen')

    ld = LaunchDescription()
    for action in [
        set_env_vars,
        gzserver,
        gzclient,
        delayed_spawn_r1,
        delayed_spawn_r2,
        rsp_r1,
        rsp_r2,
        bridge_r1,
        bridge_r2,
        sarsa_r1,
        sarsa_r2,
        delayed_carto_r1,
        delayed_carto_r2,
        rviz,
    ]:
        ld.add_action(action)

    return ld

