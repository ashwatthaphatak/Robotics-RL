Experimenting with RL for Robotics. Currently working with 

1. Manual Q Table
    - You can test by running this command `ros2 launch wall_following_ros2 wall_following.launch.py`
    - Make sure you build the package and source the `install/setup.sh` before running the launch. 

2. Q - Learning
    - To train, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=train algorithm:=q_learning`
    - To test, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=test algorithm:=q_learning`

3. SARSA 
    - To train, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=train algorithm:=sarsa`
    - To test, run - `ros2 launch wall_following_ros2 wall_following.launch.py mode:=test algorithm:=sarsa`

