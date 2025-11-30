#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    ExecuteProcess,
    TimerAction,
    GroupAction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('wall_following_ros2')
    tb3_gz = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    tb3_cartographer = get_package_share_directory('turtlebot3_cartographer')

    world_path = os.path.join(pkg_share, 'worlds', 'largemaze.world')
    rviz_config = os.path.join(nav2_bringup, 'rviz', 'nav2_default_view.rviz')
    model_sdf = os.path.join(tb3_gz, 'models', 'turtlebot3_burger', 'model.sdf')

    src_dir = os.path.join(
        os.path.expanduser('~'),
        'ros2_ws', 'src', 'wall_following_ros2', 'src'
    )

    set_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger')

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

    # Ensure Gazebo can load ros_gz_sim system plugins (e.g., set_entity_pose)
    plugin_path = os.path.join(os.environ.get('ROS_DISTRO_PREFIX', '/opt/ros/jazzy'), 'lib')
    set_system_plugins = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        plugin_path
    )
    # Use frame_prefix so TFs are isolated per robot
    robot_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'frame_prefix': 'robot1'}.items()
    )
    robot_state_pub_2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true', 'frame_prefix': 'robot2'}.items()
    )

    # Spawn two TB3s with explicit names so bridge remaps match
    spawn_tb3_1 = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_sim', 'create',
             '-name', 'robot1',
             '-file', model_sdf,
             '-x', '-2.0', '-y', '-3.25', '-z', '0.01'],
        output='screen',
    )
    spawn_tb3_2 = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_sim', 'create',
             '-name', 'robot2',
             '-file', model_sdf,
             '-x', '2.0', '-y', '3.25', '-z', '0.01'],
        output='screen',
    )

    delayed_spawn_1 = TimerAction(period=5.0, actions=[spawn_tb3_1])
    delayed_spawn_2 = TimerAction(period=7.0, actions=[spawn_tb3_2])

    set_env_vars = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gz, 'models')
    )

    # Bridges: explicit ROS topic names per robot -> Gazebo topics (including TF from diffdrive)
    bridge_r1 = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan:=/world/default/model/robot1/link/base_scan/sensor/hls_lfcd_lds/scan',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry:=/model/robot1/odometry',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V:=/model/robot1/tf',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model:=/model/robot1/joint_states',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist:=/model/robot1/cmd_vel',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        output='screen',
        additional_env={'ROS_NAMESPACE': 'robot1'}
    )

    bridge_r2 = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan:=/world/default/model/robot2/link/base_scan/sensor/hls_lfcd_lds/scan',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry:=/model/robot2/odometry',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V:=/model/robot2/tf',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model:=/model/robot2/joint_states',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist:=/model/robot2/cmd_vel',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        output='screen',
        additional_env={'ROS_NAMESPACE': 'robot2'}
    )

    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen',
    )

    # Two independent SARSA run nodes in different namespaces.
    sarsa_1 = GroupAction([
        PushRosNamespace('robot1'),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(src_dir, 'sarsa_run.py'),
                '--max_linear', '0.5',
            ],
            output='screen'
        )
    ])
    sarsa_2 = GroupAction([
        PushRosNamespace('robot2'),
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(src_dir, 'sarsa_run.py'),
                '--max_linear', '0.5',
            ],
            output='screen'
        )
    ])

    carto_config_dir = os.path.join(pkg_share, 'config')

    # Cartographer per robot (explicit nodes with remaps to /tf, /tf_static, /robotX/{scan,odom,map})
    carto_r1 = GroupAction([
        PushRosNamespace('robot1'),
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
            arguments=[
                '-configuration_directory', carto_config_dir,
                '-configuration_basename', 'cartographer_robot1.lua'
            ],
            remappings=[
                ('/tf', '/tf'),
                ('/tf_static', '/tf_static'),
                ('scan', '/robot1/scan'),
                ('odom', '/robot1/odom'),
                ('submap_query', '/robot1/submap_query'),
                ('submap_list', '/robot1/submap_list'),
                ('trajectory_node_list', '/robot1/trajectory_node_list'),
                ('landmark_poses_list', '/robot1/landmark_poses_list'),
            ],
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': True, 'resolution': 0.05, 'publish_period_sec': 1.0}],
            remappings=[
                ('/tf', '/tf'),
                ('/tf_static', '/tf_static'),
                ('/submap_list', '/robot1/submap_list'),
                ('/submap_query', '/robot1/submap_query'),
                ('/map', '/robot1/map'),
            ],
        ),
    ])

    carto_r2 = GroupAction([
        PushRosNamespace('robot2'),
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
            arguments=[
                '-configuration_directory', carto_config_dir,
                '-configuration_basename', 'cartographer_robot2.lua'
            ],
            remappings=[
                ('/tf', '/tf'),
                ('/tf_static', '/tf_static'),
                ('scan', '/robot2/scan'),
                ('odom', '/robot2/odom'),
                ('submap_query', '/robot2/submap_query'),
                ('submap_list', '/robot2/submap_list'),
                ('trajectory_node_list', '/robot2/trajectory_node_list'),
                ('landmark_poses_list', '/robot2/landmark_poses_list'),
            ],
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': True, 'resolution': 0.05, 'publish_period_sec': 1.0}],
            remappings=[
                ('/tf', '/tf'),
                ('/tf_static', '/tf_static'),
                ('/submap_list', '/robot2/submap_list'),
                ('/submap_query', '/robot2/submap_query'),
                ('/map', '/robot2/map'),
            ],
        ),
    ])

    delayed_cartographer = TimerAction(period=10.0, actions=[carto_r1])
    delayed_cartographer_2 = TimerAction(period=11.0, actions=[carto_r2])

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
        set_model,
        set_system_plugins,
        set_env_vars,
        gzserver,
        gzclient,
        robot_state_pub,
        robot_state_pub_2,
        delayed_spawn_1,
        delayed_spawn_2,
        bridge_r1,
        bridge_r2,
        rviz,
        sarsa_1,
        sarsa_2,
        delayed_cartographer,
        delayed_cartographer_2,
        tf_between_robots,
    ]:
        ld.add_action(act)

    return ld
