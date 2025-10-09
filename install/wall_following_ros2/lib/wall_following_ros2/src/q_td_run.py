#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class QLearningWallFollower(Node):
    def __init__(self):
        super().__init__('q_learning_wall_follower')

        self.setup_80_state_system()

        self.setup_ros_communications()

        self.print_q_table()
        
        self.get_logger().info("80-State Wall Follower Node Started - Manual Policy Active!")
        
    def setup_80_state_system(self):
        """Define the 80-state system according to professor's slides"""
        self.L_STATES = ['CLOSE', 'FAR']
        self.F_STATES = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        self.RF_STATES = ['CLOSE', 'FAR']
        self.R_STATES = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']

        self.DISTANCE_BOUNDARIES = {
            'TOO_CLOSE': (0.0, 0.3),
            'CLOSE': (0.3, 0.6), 
            'MEDIUM': (0.6, 1.2),
            'FAR': (1.2, 2.5),
            'TOO_FAR': (2.5, float('inf'))
        }

        self.L_SECTOR = (80, 100)
        self.F_SECTOR = (355, 5)
        self.RF_SECTOR = (310, 320)
        self.R_SECTOR = (260, 280)

        self.actions = {
            'FORWARD': (0.2, 0.0),
            'LEFT': (0.2, 0.785),
            'RIGHT': (0.2, -0.785),
            'STOP': (0.0, 0.0)
        }
        self.action_names = list(self.actions.keys())

        self.q_table = self.create_manual_q_table()
        
        self.get_logger().info("80-State system setup complete!")
    
    def create_manual_q_table(self):
        """Create a detailed manual Q-table covering all 80 states"""
        q_table = {}
        
        # Generate all 80 state combinations
        for l in self.L_STATES:
            for f in self.F_STATES:
                for rf in self.RF_STATES:
                    for r in self.R_STATES:
                        state = (l, f, rf, r)
                        
                        # DEFAULT POLICY: Conservative forward movement
                        base_policy = [3, 1, 1, 2]  # [FORWARD, LEFT, RIGHT, STOP]
                        
                        # ===== CATEGORY 1: IDEAL WALL FOLLOWING (8 states) =====
                        if (l == 'FAR' and f == 'FAR' and rf == 'FAR' and r == 'MEDIUM'):
                            # Perfect wall following conditions
                            base_policy = [10, 2, 2, 1]  # Strong forward preference
                        
                        elif (l == 'FAR' and f == 'FAR' and rf == 'FAR' and r == 'CLOSE'):
                            # Slightly closer than ideal but still good
                            base_policy = [8, 3, 1, 2]   # Forward with slight left tendency
                        
                        # ===== CATEGORY 2: TOO CLOSE TO RIGHT WALL (10 states) =====
                        elif r == 'TOO_CLOSE':
                            if f == 'FAR' and rf == 'FAR':
                                # Simple correction: turn away from wall
                                base_policy = [2, 10, -5, 3]
                            elif f == 'CLOSE' and rf == 'FAR':
                                # Front getting close while too close to wall
                                base_policy = [1, 8, -8, 6]
                            elif f == 'TOO_CLOSE':
                                # Emergency: front obstacle + too close to wall
                                base_policy = [-10, 8, -10, 10]
                            elif rf == 'CLOSE':
                                # Corner detection while too close
                                base_policy = [1, 9, -7, 4]
                            else:
                                # General too close case
                                base_policy = [2, 9, -6, 4]
                        
                        # ===== CATEGORY 3: TOO FAR FROM RIGHT WALL (10 states) =====
                        elif r == 'TOO_FAR':
                            if f == 'FAR' and rf == 'FAR':
                                # Simple correction: turn toward wall
                                base_policy = [2, -5, 10, 3]
                            elif f == 'CLOSE' and rf == 'FAR':
                                # Front obstacle while too far from wall
                                base_policy = [1, -8, 8, 6]
                            elif f == 'TOO_CLOSE':
                                # Emergency: front obstacle + too far from wall
                                base_policy = [-10, -10, 8, 10]
                            elif rf == 'CLOSE':
                                # Corner detection while too far
                                base_policy = [1, -7, 9, 4]
                            else:
                                # General too far case
                                base_policy = [2, -6, 9, 4]
                        
                        # ===== CATEGORY 4: FRONT OBSTACLE SCENARIOS (16 states) =====
                        elif f == 'TOO_CLOSE':
                            # Emergency stop required
                            if r == 'TOO_CLOSE':
                                base_policy = [-10, 6, -10, 10]  # Stop, slight left preference
                            elif r == 'TOO_FAR':
                                base_policy = [-10, -10, 6, 10]  # Stop, slight right preference
                            else:
                                base_policy = [-10, 5, 5, 10]    # Stop, either turn direction
                        
                        elif f == 'CLOSE':
                            # Caution: slow down or turn
                            if rf == 'CLOSE' and r == 'CLOSE':
                                base_policy = [1, 4, 4, 8]       # Narrow passage, prefer stop
                            elif rf == 'FAR' and r == 'MEDIUM':
                                base_policy = [2, 6, 3, 7]       # Turn left preference
                            elif rf == 'CLOSE' and r == 'MEDIUM':
                                base_policy = [2, 3, 6, 7]       # Turn right preference
                            else:
                                base_policy = [3, 5, 5, 6]       # General caution
                        
                        # ===== CATEGORY 5: CORNER AND TURN DETECTION (12 states) =====
                        elif (rf == 'CLOSE' and r == 'FAR'):
                            # Right corner detected - need to turn
                            if f == 'FAR':
                                base_policy = [4, 8, 2, 3]       # Turn left to follow wall
                            elif f == 'CLOSE':
                                base_policy = [2, 7, 1, 5]       # More aggressive turn
                            else:
                                base_policy = [3, 7, 2, 4]       # General corner handling
                        
                        elif (rf == 'FAR' and r == 'CLOSE' and f == 'FAR'):
                            # Coming out of corner - straighten out
                            base_policy = [7, 3, 5, 2]           # Forward with slight right
                        
                        # ===== CATEGORY 6: NARROW PASSAGE SCENARIOS (8 states) =====
                        elif (l == 'CLOSE' and r == 'CLOSE'):
                            # Both walls close - center in hallway
                            if f == 'FAR':
                                base_policy = [6, 3, 3, 4]       # Centered forward
                            elif f == 'CLOSE':
                                base_policy = [4, 4, 4, 6]       # Slow down in narrow space
                            else:
                                base_policy = [5, 3, 3, 5]       # General narrow passage
                        
                        # ===== CATEGORY 7: LEFT WALL INTERACTION (8 states) =====
                        elif l == 'CLOSE':
                            # Left wall is close - avoid drifting left
                            if f == 'FAR' and r == 'MEDIUM':
                                base_policy = [7, 1, 4, 2]       # Slight right tendency
                            elif f == 'CLOSE':
                                base_policy = [3, 2, 5, 6]       # Right turn preference
                            else:
                                base_policy = [5, 2, 3, 3]       # General avoid left
                        
                        # ===== CATEGORY 8: RIGHT-FRONT OBSTACLE (8 states) =====
                        elif rf == 'CLOSE':
                            # Obstacle on right-front
                            if f == 'FAR' and r == 'MEDIUM':
                                base_policy = [5, 6, 2, 3]       # Turn left to avoid
                            elif f == 'CLOSE':
                                base_policy = [2, 7, 1, 6]       # Strong left turn
                            else:
                                base_policy = [4, 5, 2, 4]       # General avoid right-front
                        
                        # Store the policy for this state
                        q_table[state] = base_policy
        
        self.get_logger().info(f"Comprehensive Q-table created with {len(q_table)} unique policies")
        return q_table

    def print_q_table(self):
        """Print the Q-table in a readable format"""
        self.get_logger().info("=== Q-TABLE ===")
        self.get_logger().info("Format: State (L, F, RF, R) -> [FORWARD, LEFT, RIGHT, STOP]")
        self.get_logger().info("-" * 80)
        
        for state, q_values in self.q_table.items():
            l, f, rf, r = state
            best_action_idx = q_values.index(max(q_values))
            best_action = self.action_names[best_action_idx]
            
            self.get_logger().info(
                f"({l:>9}, {f:>9}, {rf:>5}, {r:>8}) -> "
                f"{q_values} | Best: {best_action}"
            )
        
        self.get_logger().info("-" * 80)
        self.get_logger().info(f"Total states: {len(self.q_table)}")
    
    def setup_ros_communications(self):
        """Setup ROS subscribers and publishers"""

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.cmd_publisher = self.create_publisher(
            Twist,
            '/cmd_vel', 
            10)
        
        self.get_logger().info("ROS communications setup complete!")
    
    def get_sector_average(self, ranges, start_deg, end_deg):
        """Calculate average distance in a LiDAR sector"""
        num_readings = len(ranges)

        if start_deg > end_deg:
            start_idx = int(start_deg * num_readings / 360)
            end_idx = num_readings
            sector1 = ranges[start_idx:end_idx]
            sector2 = ranges[0:int(end_deg * num_readings / 360)]
            sector_readings = sector1 + sector2
        else:
            start_idx = int(start_deg * num_readings / 360)
            end_idx = int(end_deg * num_readings / 360)
            sector_readings = ranges[start_idx:end_idx]

        valid_readings = [r for r in sector_readings if r != float('inf') and not math.isnan(r)]
        
        if valid_readings:
            return sum(valid_readings) / len(valid_readings)
        return float('inf')
    
    def distance_to_state(self, distance, sector_type='general'):
        """Convert distance measurement to state category"""
        boundaries = self.DISTANCE_BOUNDARIES

        if sector_type == 'front':
            available_states = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR']
        elif sector_type == 'right':
            available_states = ['TOO_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'TOO_FAR']
        elif sector_type == 'left' or sector_type == 'right_front':
            available_states = ['CLOSE', 'FAR']
        else:
            available_states = ['CLOSE', 'FAR']
        
        for state in available_states:
            min_dist, max_dist = boundaries[state]
            if min_dist <= distance < max_dist:
                return state

        if sector_type == 'right':
            return 'TOO_FAR'
        elif sector_type == 'front':
            return 'FAR'
        else:
            return 'FAR'
    
    def determine_state(self, scan_data):
        """Convert LiDAR data to one of 80 states"""
        l_dist = self.get_sector_average(scan_data, *self.L_SECTOR)
        f_dist = self.get_sector_average(scan_data, *self.F_SECTOR)  
        rf_dist = self.get_sector_average(scan_data, *self.RF_SECTOR)
        r_dist = self.get_sector_average(scan_data, *self.R_SECTOR)
        
        l_state = self.distance_to_state(l_dist, 'left')
        f_state = self.distance_to_state(f_dist, 'front')
        rf_state = self.distance_to_state(rf_dist, 'right_front') 
        r_state = self.distance_to_state(r_dist, 'right')
        
        return (l_state, f_state, rf_state, r_state)
    
    def execute_action(self, action_name):
        """Convert action name to Twist command"""
        twist = Twist()
        linear, angular = self.actions[action_name]
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_publisher.publish(twist)
    
    def scan_callback(self, msg):
        """Main LiDAR callback - processes scan data and chooses actions"""
        current_state = self.determine_state(msg.ranges)

        action_scores = self.q_table[current_state]
        best_action_index = action_scores.index(max(action_scores))
        best_action = self.action_names[best_action_index]

        self.execute_action(best_action)

        if not hasattr(self, 'scan_count'):
            self.scan_count = 0
        self.scan_count += 1
        
        if self.scan_count % 50 == 0:
            l_dist = self.get_sector_average(msg.ranges, *self.L_SECTOR)
            f_dist = self.get_sector_average(msg.ranges, *self.F_SECTOR)
            rf_dist = self.get_sector_average(msg.ranges, *self.RF_SECTOR)
            r_dist = self.get_sector_average(msg.ranges, *self.R_SECTOR)
            
            self.get_logger().info(
                f"State: {current_state} | "
                f"Distances: L={l_dist:.2f}, F={f_dist:.2f}, RF={rf_dist:.2f}, R={r_dist:.2f} | "
                f"Action: {best_action}"
            )

def main():
    rclpy.init()
    node = QLearningWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
