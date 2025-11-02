#!/usr/bin/env python3
import os
import math
import cv2
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ament_index_python.packages import get_package_share_directory

class SensorModel(Node):
    def __init__(self):
        super().__init__('sensor_model')

        # Parameters
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('sigma_hit', 0.2)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.2)
        self.declare_parameter('z_max', 3.5)
        self.declare_parameter('beam_subsample', 20)
        self.declare_parameter('distance_field_output', '')
        self.declare_parameter('distance_field_invert', True)

        self._share_dir = get_package_share_directory('particle-filter-ros2')
        default_map = os.path.join(self._share_dir, 'maps', 'map.yaml')

        p = self.get_parameter
        self.map_yaml = self._resolve_map_yaml(p('map_yaml').value, default_map)
        self.sigma_hit = p('sigma_hit').value
        self.z_hit = p('z_hit').value
        self.z_rand = p('z_rand').value
        self.z_max = p('z_max').value
        self.beam_subsample = p('beam_subsample').value
        self.distance_field_output = self._resolve_output_path(
            p('distance_field_output').value,
            os.path.join(os.path.dirname(self.map_yaml), 'distance_field.png')
        )
        self.distance_field_invert = bool(p('distance_field_invert').value)

        self.dist_field = None

        self.load_map(self.map_yaml)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('Sensor model node started.')

    def _resolve_map_yaml(self, value, default_map):
        candidate = os.path.expanduser(value) if value else ''
        if not candidate:
            return default_map

        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate

        search_roots = [
            self._share_dir,
            os.path.join(self._share_dir, 'config'),
        ]
        for root in search_roots:
            maybe = os.path.normpath(os.path.join(root, candidate))
            if os.path.exists(maybe):
                return maybe

        self.get_logger().warn(
            f"map_yaml parameter '{value}' not found; falling back to default {default_map}"
        )
        return default_map

    def _resolve_output_path(self, value, default_output):
        candidate = os.path.expanduser(value) if value else ''
        if not candidate:
            return default_output

        if os.path.isabs(candidate):
            output_dir = os.path.dirname(candidate)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            return candidate

        resolved = os.path.normpath(os.path.join(self._share_dir, candidate))
        output_dir = os.path.dirname(resolved)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return resolved

    def load_map(self, map_yaml):
        if not map_yaml or not os.path.exists(map_yaml):
            self.get_logger().error(f"Map YAML not found: {map_yaml}")
            return

        with open(map_yaml, 'r') as f:
            info = yaml.safe_load(f)

        image_path = info['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml), image_path)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f"Failed to read map image: {image_path}")
            return

        if info['negate']:
            img = 255 - img
        _, binary = cv2.threshold(img, info['occupied_thresh']*255, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        self.dist_field = dist * info['resolution']
        if np.max(dist) > 0:
            normalized = (dist/np.max(dist)*255).astype(np.uint8)
        else:
            normalized = dist.astype(np.uint8)
        if self.distance_field_invert:
            normalized = 255 - normalized
        cv2.imwrite(self.distance_field_output, normalized)
        self.get_logger().info(f"Saved likelihood field → {self.distance_field_output}")

    def scan_callback(self, scan):
        if self.dist_field is None:
            return

        total_loglik = 0.0
        count = 0
        for r in scan.ranges[::int(self.beam_subsample)]:
            if math.isinf(r) or r >= self.z_max:
                continue
            pz = self.prob_hit(r)
            total_loglik += math.log(max(pz, 1e-9))
            count += 1
        if count:
            self.get_logger().info(f"Avg log-likelihood: {total_loglik/count:.3f}")

    def prob_hit(self, dist):
        norm = 1.0 / (self.sigma_hit * math.sqrt(2*math.pi))
        return self.z_hit * norm * math.exp(-0.5 * (dist/self.sigma_hit)**2) + \
               self.z_rand * (1.0/self.z_max)

def main(args=None):
    rclpy.init(args=args)
    node = SensorModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
