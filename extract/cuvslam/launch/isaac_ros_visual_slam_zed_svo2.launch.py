# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# ---------------------------------------------------------------------------
# MODIFIED for the freehand-US probe-tracking comparison.
#
# Apache-2.0 section 4(b) requires modified files to carry prominent notice of
# the change, so: this file began as an NVIDIA Isaac ROS launch template and was
# substantially rewritten here. NVIDIA's copyright and licence above are
# retained and continue to apply; the modifications are
# Copyright (c) 2026 Zohar Oddes and Dana Solav, MIT licensed (see LICENSE).
#
# What was changed, and why each matters to the published results:
#   * Plays back a recorded SVO2 file through the ZED ROS 2 wrapper rather than
#     opening a live camera, with `svo.replay_rate: 0.4` and
#     `svo.svo_real_time: False`.
#   * `use_sim_time: True` throughout, with `svo.publish_svo_clock: True`, so
#     the original acquisition timestamps are preserved rather than replaced by
#     wall-clock times.
#   * The ZED wrapper's own positional tracking is DISABLED
#     (`pos_tracking.enabled: False`) -- cuVSLAM does the tracking, and leaving
#     both on would be measuring the wrong thing.
#   * Grayscale rectified stereo published (`video.publish_gray: True`), which
#     is what the visual-SLAM node consumes.
#   * QoS depths and cuVSLAM buffer sizes raised (`image.qos_depth: 300`,
#     `sensors.qos_depth: 500`, `image_buffer_size`, `imu_buffer_size`) to stop
#     the SLAM node dropping frames under load. Together with the 0.4 replay
#     rate this is the frame-loss mitigation the paper reports as a limitation
#     for any real-time claim.
#   * Registers an end-of-file monitor that requests the optimised trajectory
#     when playback finishes (see ../monitor_svo_end.py).
#   * Static transforms carry THIS RIG's measured ZED stereo baseline. They are
#     not defaults; change them for other hardware.
# ---------------------------------------------------------------------------
"""Launch SVO2 playback on ZED with Visual SLAM and EOF monitoring."""
import os

# The container workspace root. Hardcoded absolute paths in the original are
# replaced by this, so the launch file works in a container mounted elsewhere.
ISAAC_ROS_WS = os.environ.get("ISAAC_ROS_WS", "/workspaces/isaac_ros-dev")
import launch
from launch.actions import (DeclareLaunchArgument, RegisterEventHandler, 
                            OpaqueFunction)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os
import subprocess
import time



def generate_launch_description():
    svo_file_path = LaunchConfiguration('svo_file_path')
    svo_loop = LaunchConfiguration('svo_loop')
    svo_replay_rate = LaunchConfiguration('svo_replay_rate')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_imu_fusion = LaunchConfiguration('enable_imu_fusion')
    camera_name = LaunchConfiguration('zed_camera_model')
    enable_ipc = LaunchConfiguration('enable_ipc')

    # Load ZED wrapper config file
    zed_config_path = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'config',
        'common_stereo.yaml'
    )

    # ZED Camera composable node configured for SVO2 playback
    zed_camera_node = ComposableNode(
        name='zed',
        package='zed_components',
        plugin='stereolabs::ZedCamera',
        parameters=[
            zed_config_path,
            {
                'svo.svo_path': svo_file_path,
                'svo.svo_real_time': False,
                'svo.replay_rate': svo_replay_rate,
                'svo.loop': svo_loop,
                'svo.publish_svo_clock': True,
                
                'general.camera_model': camera_name,
                'video.publish_gray': True,
                'pos_tracking.publish_tf': False,
                'sensors.publish_imu': True,
                'pos_tracking.enabled': False,
                
                'image.qos_history': 1, # '1': KEEP_LAST - '2': KEEP_ALL
                'image.qos_depth': 300, # Queue size if using KEEP_LAST
                'image.qos_reliability': 1, # '1': RELIABLE - '2': BEST_EFFORT -
                'image.qos_durability': 1, # '1': TRANSIENT_LOCAL - '2': VOLATILE 
                
                'sensors.qos_history': 1, # '1': KEEP_LAST - '2': KEEP_ALL
                'sensors.qos_depth': 500, # Queue size if using KEEP_LAST
                'sensors.qos_reliability': 1, # '1': RELIABLE - '2': BEST_EFFORT -
                'sensors.qos_durability': 1, # '1': TRANSIENT_LOCAL - '2': VOLATILE 

            }
        ],
        extra_arguments=[{'use_intra_process_comms': enable_ipc}],
    )
    # Visual SLAM node tuned for the SVO playback pipeline
    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_imu_fusion': enable_imu_fusion,
            'use_sim_time': use_sim_time,
            # 'gyro_noise_density': 0.000244,
            # 'gyro_random_walk': 0.000019393,
            # 'accel_noise_density': 0.001862,
            # 'accel_random_walk': 0.003,
            # 'calibration_frequency': 200.0,
            'override_publishing_stamp': False,   
            'multicam_mode': 1,
            'sync_matching_threshold_ms': 5.0, # Maximum value of acceptable timestamp deltas between the camera images to still consider their timestamp identical. Should be < 0.5 / fps_value.
            'image_jitter_threshold_ms': 34.0,  # at 60hz, expected nominal frame interval ~16.6ms
            'base_frame': 'base_link',
            'imu_frame': 'zed_imu_link',
            'map_frame': 'map',
            'odom_frame': 'odom',
            'enable_slam_visualization': False,
            'enable_landmarks_view': False,
            'enable_observations_view': False,
            'enable_localization_n_mapping': True,
            'num_cameras': 2,
            'enable_debug_mode': False,
            'debug_dump_path': os.path.join(ISAAC_ROS_WS, 'visual_slam_debug') + '/',
            'image_buffer_size': 1000, # Buffer size used by the image synchronizer.
            'imu_buffer_size': 500, # Buffer size used by the IMU sequencer.
            'image_qos': 'DEFAULT',
            'imu_qos': 'DEFAULT',
            'publish_map_to_odom_tf': False,
            'publish_odom_to_base_tf': False,
            'broadcast_map_to_base_frame': False,
            'verbosity': 1,
            'path_max_size': 1,
            'slam_max_map_size': 1000,
            # 'slam_throttling_time_ms': 100
        }],
        remappings=[
            ('visual_slam/image_0', '/zed/left/gray/rect/image'),
            ('visual_slam/camera_info_0', '/zed/left/gray/rect/camera_info'),
            ('visual_slam/image_1', '/zed/right/gray/rect/image'),
            ('visual_slam/camera_info_1', '/zed/right/gray/rect/camera_info'),
            ('visual_slam/imu', '/zed/imu/data'),
        ],
        extra_arguments=[{'use_intra_process_comms': enable_ipc}],
    )

    # Shared composable container for both ZED and SLAM nodes (isolated per node for debugging)
    composable_nodes_container = ComposableNodeContainer(
        name='zed_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        arguments=['--ros-args', '--log-level', 'info'],
        composable_node_descriptions=[
            visual_slam_node,
            zed_camera_node,
            # Static TF to connect base_link to camera_link
            ComposableNode(
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                name='tf_base_to_camera',
                parameters=[{
                    'child_frame_id': 'zed_camera_link',
                    'frame_id': 'base_link',
                    'translation.x': 0.0,
                    'translation.y': 0.0,
                    'translation.z': 0.0,
                    'rotation.x': 0.0,
                    'rotation.y': 0.0,
                    'rotation.z': 0.0,
                    'rotation.w': 1.0,
                }],
            ),
            # Static TF for camera center (0, 0, 0.016 from camera_link)
            ComposableNode(
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                name='tf_camera_to_center',
                parameters=[{
                    'child_frame_id': 'zed_camera_center',
                    'frame_id': 'zed_camera_link',
                    'translation.x': 0.0,
                    'translation.y': 0.0,
                    'translation.z': 0.016,
                    'rotation.x': 0.0,
                    'rotation.y': 0.0,
                    'rotation.z': 0.0,
                    'rotation.w': 1.0,
                }],
            ),
            # Static TF for left camera frame
            ComposableNode(
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                name='tf_center_to_left',
                parameters=[{
                    'child_frame_id': 'zed_left_camera_frame',
                    'frame_id': 'zed_camera_center',
                    'translation.x': -0.01,
                    'translation.y': 0.02486265,
                    'translation.z': 0.0,
                    'rotation.x': 0.0,
                    'rotation.y': 0.0,
                    'rotation.z': 0.0,
                    'rotation.w': 1.0,
                }],
            ),
            # Static TF for left camera optical frame (rotation)
            ComposableNode(
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                name='tf_left_to_left_optical',
                parameters=[{
                    'child_frame_id': 'zed_left_camera_frame_optical',
                    'frame_id': 'zed_left_camera_frame',
                    'translation.x': 0.0,
                    'translation.y': 0.0,
                    'translation.z': 0.0,
                    'rotation.x': 0.5,
                    'rotation.y': -0.5,
                    'rotation.z': 0.5,
                    'rotation.w': -0.5,
                }],
            ),
            # Static TF for right camera frame
            ComposableNode(
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                name='tf_center_to_right',
                parameters=[{
                    'child_frame_id': 'zed_right_camera_frame',
                    'frame_id': 'zed_camera_center',
                    'translation.x': -0.01,
                    'translation.y': -0.02486265,
                    'translation.z': 0.0,
                    'rotation.x': 0.0,
                    'rotation.y': 0.0,
                    'rotation.z': 0.0,
                    'rotation.w': 1.0,
                }],
            ),
            # Static TF for right camera optical frame (rotation)
            ComposableNode(
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                name='tf_right_to_right_optical',
                parameters=[{
                    'child_frame_id': 'zed_right_camera_frame_optical',
                    'frame_id': 'zed_right_camera_frame',
                    'translation.x': 0.0,
                    'translation.y': 0.0,
                    'translation.z': 0.0,
                    'rotation.x': 0.5,
                    'rotation.y': -0.5,
                    'rotation.z': 0.5,
                    'rotation.w': -0.5,
                }],
            ),
        ],
        output='screen'
    )
    # Monitor SVO status; on EOF call get_all_poses then exit
    svo_monitor = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(ISAAC_ROS_WS, 'extract/cuvslam/monitor_svo_end.py'),
            '--status-topic', '/zed/status/svo',
            '--poses-file', '/tmp/poses_response.txt'
        ],
        output='screen',
        on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())]
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'svo_file_path',
            description='Path to the SVO2 file to playback (REQUIRED)'),
        DeclareLaunchArgument(
            'svo_loop',
            default_value='False',
            description='Loop the SVO instead of stopping at EOF'),
        DeclareLaunchArgument(
            'svo_replay_rate',
            default_value='0.4',
            description='Replay rate for SVO when not in realtime mode (0.1-5.0)'),
        DeclareLaunchArgument(
            'zed_camera_model',
            default_value='zedxm',
            description='ZED camera model: zedxm, zedx, zed2i, zed2, zedm, zed'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulated time (set to True for rosbag/svo2 playback)'),
        DeclareLaunchArgument(
            'enable_imu_fusion',
            default_value='True',
            description='Enable IMU fusion in Visual SLAM'),
        DeclareLaunchArgument(
            # Stereolabs recommends disabling IPC when the ZED wrapper runs
            # alongside Isaac ROS, to avoid conflicts with the NITROS
            # transport: https://www.stereolabs.com/docs/isaac-ros/setting_up_isaac_ros
            'enable_ipc',
            default_value='False',
            description='Enable Intra-Process Communication (IPC).'),
        composable_nodes_container,
        svo_monitor,
    ])
