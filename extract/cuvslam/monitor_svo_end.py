#!/usr/bin/env python3
"""Monitor SVO playback end, call get_all_poses, then exit the launch."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8
from zed_msgs.msg import SvoStatus
from isaac_ros_visual_slam_interfaces.srv import GetAllPoses
import argparse
import sys
import os
import time

class SvoEndMonitor(Node):
    def __init__(self, status_topic, poses_file):
        super().__init__('svo_end_monitor')
        self.poses_file = poses_file
        self.cli = self.create_client(GetAllPoses, '/visual_slam/get_all_poses')
        self.get_logger().info(f'Waiting for SVO end on {status_topic}')
        self._handled = False
        self.status_sub = None
        self._pending_call_at = None
        # Topic may appear as either UInt8 or zed_msgs/SvoStatus; try the richer type first, then fall back.
        try:
            self.status_sub = self.create_subscription(
                SvoStatus, status_topic, self.on_status_svo_status, qos_profile_sensor_data)
            self.get_logger().info(f'Subscribed to {status_topic} as SvoStatus')
        except Exception as exc:  # fall back if the topic is already typed as UInt8
            self.get_logger().warn(f'SvoStatus subscription failed ({exc}); falling back to UInt8')
            self.status_sub = self.create_subscription(
                UInt8, status_topic, self.on_status_uint8, qos_profile_sensor_data)

    def on_status_uint8(self, msg: UInt8):
        if msg.data == 2 and not self._handled:  # SVO playback ended (UInt8 encoding)
            self._handled = True
            if self.status_sub is not None:
                self.destroy_subscription(self.status_sub)
                self.status_sub = None
            self._pending_call_at = time.monotonic() + 0.5
            self.get_logger().info('SVO reached end; will call get_all_poses after short delay')

    def on_status_svo_status(self, msg: SvoStatus):
        if msg.status == SvoStatus.STATUS_END and not self._handled:
            self._handled = True
            if self.status_sub is not None:
                self.destroy_subscription(self.status_sub)
                self.status_sub = None
            self._pending_call_at = time.monotonic() + 0.5
            self.get_logger().info('SVO reached end (SvoStatus); will call get_all_poses after short delay')

    def call_and_save(self):
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('get_all_poses not available')
            rclpy.shutdown()
            return

        req = GetAllPoses.Request()
        req.max_count = 10000
        self.get_logger().info('Calling get_all_poses (max_count=10000)...')
        future = self.cli.call_async(req)
        self.get_logger().info('Waiting for get_all_poses response...')
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        self.get_logger().info('get_all_poses wait finished')

        if not future.done():
            self.get_logger().error('get_all_poses call timed out')
            future.cancel()
        elif future.exception() is not None:
            self.get_logger().error(f'get_all_poses exception: {future.exception()}')
        elif future.result() is None:
            self.get_logger().error('get_all_poses returned no result')
        else:
            try:
                with open(self.poses_file, 'w') as f:
                    f.write(str(future.result()))
                self.get_logger().info(f'Poses saved to {self.poses_file}')
            except OSError as e:
                self.get_logger().error(f'Failed to write poses: {e}')

        self.shutdown()

    def shutdown(self):
        self.get_logger().info('Monitor shutting down')
        rclpy.shutdown()
        sys.exit(0)

def main():
    """Entry point: wait for EOF on SVO status, then fetch poses and exit."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--status-topic', required=True)
    parser.add_argument('--poses-file', default='/tmp/poses_response.txt')
    args = parser.parse_args()

    rclpy.init()
    node = SvoEndMonitor(args.status_topic, args.poses_file)
    try:
        while rclpy.ok():
            # Spin in the main thread so shutdown stays clean and deterministic.
            rclpy.spin_once(node, timeout_sec=0.1)
            if node._pending_call_at is not None and time.monotonic() >= node._pending_call_at:
                node._pending_call_at = None
                node.call_and_save()
                break
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0

if __name__ == '__main__':
    sys.exit(main())