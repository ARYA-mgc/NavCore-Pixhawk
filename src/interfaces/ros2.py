#!/usr/bin/env python3
# ROS2 interface.
# Because someone said we needed more middleware.

import logging
import numpy as np
from typing import Optional

log = logging.getLogger("ros2_interface")

# Lazy ROS2 import — system works without it
_ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String, Header
    from geometry_msgs.msg import (
        Point, Quaternion, Vector3, Twist,
        PoseWithCovariance, TwistWithCovariance, Pose
    )
    from builtin_interfaces.msg import Time
    _ROS2_AVAILABLE = True
except ImportError:
    log.debug("ROS2 (rclpy) not available — ros2_interface disabled")


class INSRos2Publisher:
    # Publish ESKF state to ROS2.

    def __init__(self, node_name: str = "navcore_ins",
                 odom_topic: str = "/ins/odom",
                 imu_topic: str = "/ins/imu",
                 health_topic: str = "/ins/health",
                 frame_id: str = "odom",
                 child_frame_id: str = "base_link"):

        self._active = False
        self._node = None

        if not _ROS2_AVAILABLE:
            log.info("ROS2 interface disabled (rclpy not found)")
            return

        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(node_name)

            self._odom_pub = self._node.create_publisher(
                Odometry, odom_topic, 10)
            self._imu_pub = self._node.create_publisher(
                Imu, imu_topic, 10)
            self._health_pub = self._node.create_publisher(
                String, health_topic, 10)

            self._frame_id = frame_id
            self._child_frame_id = child_frame_id
            self._active = True

            log.info(f"ROS2 publisher active: {odom_topic}, {imu_topic}")
        except Exception as e:
            log.warning(f"ROS2 init failed: {e} — interface disabled")
            self._active = False

    def publish_odom(self, state: dict, covariance: np.ndarray):
        # Format and publish Odometry message.
        if not self._active:
            return

        msg = Odometry()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = self._child_frame_id
        msg.header.stamp = self._node.get_clock().now().to_msg()

        # Position
        pos = state["pos"]
        msg.pose.pose.position = Point(x=float(pos[0]),
                                        y=float(pos[1]),
                                        z=float(pos[2]))

        # Orientation (quaternion)
        q = state["quat"]
        msg.pose.pose.orientation = Quaternion(w=float(q[0]),
                                                x=float(q[1]),
                                                y=float(q[2]),
                                                z=float(q[3]))

        # Velocity
        vel = state["vel"]
        msg.twist.twist.linear = Vector3(x=float(vel[0]),
                                          y=float(vel[1]),
                                          z=float(vel[2]))

        # Covariance (ROS uses 6x6 pose covariance: x,y,z,rx,ry,rz)
        pose_cov = np.zeros((6, 6))
        pose_cov[0:3, 0:3] = covariance[0:3, 0:3]   # position
        pose_cov[3:6, 3:6] = covariance[6:9, 6:9]   # attitude
        msg.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0:3, 0:3] = covariance[3:6, 3:6]   # velocity
        msg.twist.covariance = twist_cov.flatten().tolist()

        self._odom_pub.publish(msg)

    def publish_imu(self, accel: np.ndarray, gyro: np.ndarray,
                    quat: np.ndarray):
        # Format and publish IMU message.
        if not self._active:
            return

        msg = Imu()
        msg.header.frame_id = self._child_frame_id
        msg.header.stamp = self._node.get_clock().now().to_msg()

        msg.angular_velocity = Vector3(x=float(gyro[0]),
                                        y=float(gyro[1]),
                                        z=float(gyro[2]))
        msg.linear_acceleration = Vector3(x=float(accel[0]),
                                           y=float(accel[1]),
                                           z=float(accel[2]))
        msg.orientation = Quaternion(w=float(quat[0]),
                                     x=float(quat[1]),
                                     y=float(quat[2]),
                                     z=float(quat[3]))

        self._imu_pub.publish(msg)

    def publish_health(self, health_json: str):
        # Format and publish health status.
        if not self._active:
            return

        msg = String()
        msg.data = health_json
        self._health_pub.publish(msg)

    def shutdown(self):
        # Shutdown ROS2 node gracefully.
        if self._active and self._node:
            self._node.destroy_node()
            log.info("ROS2 node destroyed")
            self._active = False
