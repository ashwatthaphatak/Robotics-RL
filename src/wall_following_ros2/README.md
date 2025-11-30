Experimenting with RL for Robotics. Currently working with 

1. Manual Q Table
    - You can test by running this command `ros2 launch wall_following_ros2 wall_following.launch.py`
    - Make sure you build the package and source the `install/setup.sh` before running the launch. 

2. Q - Learning
    - To train (headless), run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=train algorithm:=q_learning gui:=false` 
    - To test, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=test algorithm:=q_learning`

3. SARSA 
    - To train, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=train algorithm:=sarsa reward_mode:=shaped`
    - To test, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=test algorithm:=sarsa reward_mode:=shaped`

4. Frontier exploration (m-explore logic) on the large maze
    - Build this workspace so `explore_lite` and Nav2 are available, then run:
      `ros2 launch wall_following_ros2 maze_explore.launch.py`
    - This launches gz sim with `worlds/largemaze.world`, starts Nav2 with slam_toolbox, and uses `explore_lite` to drive the TurtleBot3 around autonomously.
    - Toggle GUI/RViz with `gui:=false` / `rviz:=false`, or override Nav2/Explore params via `params_file:=...` and `explore_params:=...`.
<<<<<<< ours
<<<<<<< ours
    - For a more stable robot, the launch defaults to TurtleBot3 waffle; you can change with `tb3_model:=burger|waffle|waffle_pi` and set spawn pose via `x_pose:=... y_pose:=... yaw_pose:=...` if you want to avoid starting near walls.
<<<<<<< ours
    - A dedicated slam_toolbox instance (online_sync) is launched before Nav2 to provide the `map -> odom` transform; you can point it to a custom config with `slam_params_file:=...`. Nav2 localization is off by default here to avoid conflicts; set `use_localization:=true` if you want localization instead of live SLAM.
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
