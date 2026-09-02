#!/usr/bin/env python3
"""Stage B: convert a native RealSense recording into a ROS 2 bag.

ENV: `ros-convert` -- ROS 2 Humble sourced FIRST, then the venv holding
     pyrealsense2. That order is load-bearing: sourcing ROS 2 puts its
     site-packages on PYTHONPATH, which Python honours regardless of whether
     the venv includes system site-packages. See ../../docs/setup.md.

Reads a `.bag` written by the RealSense Viewer and writes a rosbag2 (sqlite3)
directory publishing `/camera/{left,right}/image_raw`,
`/camera/{left,right}/camera_info` and `/imu`, preserving the original hardware
timestamps in both the message headers and the bag index. That preservation is
the whole point: the evaluation aligns this trajectory to a motion-capture
reference by its timestamps, so a converter that restamped with wall-clock
times would destroy the measurement.

Images are `mono8` at 1280x800, from the two infrared streams. IMU keeps its
native ~400 Hz.

Two independent playback pipelines are opened over the same file -- one
enabling only the infrared streams, one only the accelerometer and gyroscope.
This looks redundant and is not: enabling IMU in the frameset pipeline throttles
it to the camera rate, which would throw away more than nine tenths of the
inertial samples. The accelerometer is read in a first pass and interpolated
onto the gyroscope timestamps in a second, which is what `--unite-imu 2`, the
default and the setting used for the published results, means.

Ported from the original converter with five fixes. One of them changes the
output, and is the reason this converter now reproduces the reference bags
exactly:

  * **The IMU passes waited only one second for a frame and treated the first
    timeout as end-of-stream.** librealsense raises RuntimeError for both a
    transient timeout and a genuine end of playback, so one slow read silently
    ended the whole IMU collection -- yielding about 30 Hz of IMU instead of
    400, roughly a twentieth of the inertial data, with the converter reporting
    success. The camera loop had already been given a five-second timeout "under
    heavy system load"; the IMU loops had not. `_imu_wait()` now retries. With
    it the converter reproduces the reference bags exactly: same message counts,
    bag timestamps equal to the nanosecond, identical payloads.
  * Four IMU methods that the default configuration never reaches were removed
    (195 lines). `--unite-imu 2` takes the two-pass path instead.
  * `_interpolate_accel` sorted the entire accelerometer history and scanned it
    linearly for every gyroscope sample; the pruning that would have bounded
    that only ran in one of the removed branches. With ~22k of each that is
    ~22k sorts over ~22k keys. It now sorts once and bisects.
  * Bare `except: pass` handlers, which silently dropped IMU data and
    pipeline-teardown failures, now log.
  * A `ros2 bag info` shell-out with an unquoted interpolated path was removed.
  * The `--no-compression` flag, documented as "not implemented yet" and never
    read, was removed.
"""

from sensor_msgs.msg import CameraInfo, Imu
from std_msgs.msg import Header
import pyrealsense2 as rs
import cv_bridge
import logging
import numpy as np
import os
import argparse
import rosbag2_py

log = logging.getLogger('realsense_bag_to_rosbag2')
from rclpy.serialization import serialize_message


class DirectRealSenseConverter:
    def __init__(self, realsense_bag_path, output_bag_path, unite_imu_method=2):
        self.bag_path = realsense_bag_path
        self.output_bag_path = output_bag_path
        self.unite_imu_method = unite_imu_method  # 0=none, 1=copy, 2=linear_interpolation
        self.frame_count = 0
        self.imu_count = 0
        self.first_bag_timestamp = None
        self.last_bag_timestamp = None
        
        # Validate input bag file
        if not os.path.exists(realsense_bag_path):
            raise FileNotFoundError(f"RealSense bag file not found: {realsense_bag_path}")
        
        # Initialize CV bridge
        self.bridge = cv_bridge.CvBridge()
        
        # IMU fusion data structures
        self.last_accel_data = None  # For copy method
        self.accel_history = {}  # For interpolation method: timestamp -> accel_data
        
        # Initialize RealSense pipeline
        self._setup_realsense_pipeline()
        
        # Initialize ROS2 bag writer
        self._setup_bag_writer()
        
        print(f"Direct converter initialized")
        print(f"Input: {realsense_bag_path}")
        print(f"Output: {output_bag_path}")
        
        # Print IMU fusion method
        method_names = {0: "None (separate messages)", 1: "Copy (last accel attached to gyro)", 2: "Linear Interpolation"}
        print(f"IMU fusion method: {unite_imu_method} - {method_names.get(unite_imu_method, 'Unknown')}")

    def _setup_realsense_pipeline(self):
        """Setup RealSense pipeline for bag playback - camera streams only
        
        IMPORTANT: We do NOT enable IMU streams here because the camera framerate
        (30 Hz) would limit IMU to 30 Hz as well due to frameset synchronization.
        Instead, we use a separate pipeline for reading IMU data.
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable playback from bag file
        self.config.enable_device_from_file(self.bag_path, repeat_playback=False)
        
        # Enable ONLY infrared streams for camera data
        # Note: RealSense uses stream indices 1 and 2 for stereo pair (IR0=1, IR1=2)
        self.config.enable_stream(rs.stream.infrared, 1)  # Stereo IR 1 (left)
        self.config.enable_stream(rs.stream.infrared, 2)  # Stereo IR 2 (right)
        
        # Start pipeline
        self.pipeline_profile = self.pipeline.start(self.config)
        self.device = self.pipeline_profile.get_device()
        
        # Setup playback
        self.playback_device = self.device.as_playback()
        self.playback_device.set_real_time(False)
        
        # Get camera calibration
        self._setup_camera_calibration()
        
        # Setup separate pipeline for IMU data
        self._setup_imu_pipeline()

    def _setup_camera_calibration(self):
        """Extract camera calibration from RealSense streams"""
        try:
            streams = self.pipeline_profile.get_streams()
            ir_streams = [s for s in streams if s.stream_type() == rs.stream.infrared]
            
            if len(ir_streams) >= 2:
                self.left_ir_profile = ir_streams[0].as_video_stream_profile()
                self.right_ir_profile = ir_streams[1].as_video_stream_profile()
                
                self.left_intrinsics = self.left_ir_profile.get_intrinsics()
                self.right_intrinsics = self.right_ir_profile.get_intrinsics()
                
                self.extrinsics = self.left_ir_profile.get_extrinsics_to(self.right_ir_profile)
                self.baseline = abs(self.extrinsics.translation[0])
                
                print(f"Camera calibration loaded - Baseline: {self.baseline:.3f}m")
            else:
                raise ValueError("Need at least 2 infrared streams")
        except Exception as e:
            print(f"Warning: Failed to get camera calibration: {e}")
            # Set defaults
            self.baseline = 0.050

    def _setup_imu_pipeline(self):
        """Setup separate pipeline for reading IMU data at full 400 Hz
        
        We use a separate pipeline without infrared streams so IMU is not
        limited by the camera framerate.
        """
        self.imu_pipeline = rs.pipeline()
        self.imu_config = rs.config()
        
        # Enable playback from SAME bag file
        self.imu_config.enable_device_from_file(self.bag_path, repeat_playback=False)
        
        # Enable ONLY IMU streams (no infrared)
        self.imu_config.enable_stream(rs.stream.accel, 0, rs.format.motion_xyz32f, 400)
        self.imu_config.enable_stream(rs.stream.gyro, 0, rs.format.motion_xyz32f, 400)
        
        # Start IMU pipeline
        self.imu_pipeline_profile = self.imu_pipeline.start(self.imu_config)
        self.imu_device = self.imu_pipeline_profile.get_device()
        
        # Setup playback
        self.imu_playback_device = self.imu_device.as_playback()
        self.imu_playback_device.set_real_time(False)
        
        print("IMU pipeline initialized at full 400 Hz")

    def _setup_bag_writer(self):
        """Setup ROS2 bag writer with proper timestamp preservation"""
        # Remove existing bag if it exists
        if os.path.exists(self.output_bag_path):
            import shutil
            shutil.rmtree(self.output_bag_path)
        
        # Create storage options
        storage_options = rosbag2_py.StorageOptions(
            uri=self.output_bag_path,
            storage_id='sqlite3'
        )
        
        converter_options = rosbag2_py.ConverterOptions('', '')
        
        # Create bag writer
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(storage_options, converter_options)
        
        # Create topic metadata
        topics = [
            rosbag2_py.TopicMetadata(
                name='/camera/left/image_raw',
                type='sensor_msgs/msg/Image',
                serialization_format='cdr'
            ),
            rosbag2_py.TopicMetadata(
                name='/camera/right/image_raw', 
                type='sensor_msgs/msg/Image',
                serialization_format='cdr'
            ),
            rosbag2_py.TopicMetadata(
                name='/camera/left/camera_info',
                type='sensor_msgs/msg/CameraInfo',
                serialization_format='cdr'
            ),
            rosbag2_py.TopicMetadata(
                name='/camera/right/camera_info',
                type='sensor_msgs/msg/CameraInfo', 
                serialization_format='cdr'
            ),
            rosbag2_py.TopicMetadata(
                name='/imu',
                type='sensor_msgs/msg/Imu',
                serialization_format='cdr'
            )
        ]
        
        # Create topics
        for topic in topics:
            self.writer.create_topic(topic)
        
        print("ROS2 bag writer initialized with topics:")
        for topic in topics:
            print(f"  {topic.name} ({topic.type})")
    
    def _create_camera_info_message(self, intrinsics, header, camera_index=0):
        """Create CameraInfo message from RealSense intrinsics"""
        # ROS2 api msg/CameraInfo - https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width = intrinsics.width
        camera_info.height = intrinsics.height
        
        # Intrinsic matrix K
        camera_info.k = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        # Rectification matrix R (identity for left, rotation for right if needed)
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        # Projection matrix P
        Tx = -intrinsics.fx * self.baseline if camera_index == 1 else 0.0 # 0 for left, -fx*baseline for right
        # Tx = 0.0 # zero for both cameras - This is modified during launching isaac ros vslam! no need to adjust here!
        Ty = 0.0 # zero for both cameras
        camera_info.p = [
            intrinsics.fx, 0.0, intrinsics.ppx, Tx,
            0.0, intrinsics.fy, intrinsics.ppy, Ty,
            0.0, 0.0, 1.0, 0.0
        ]
        # Given a 3D point [X Y Z]', the projection (x, y) of the point onto
        # the rectified image is given by:
        # [u v w]' = P * [X Y Z 1]'
        #        x = u / w
        #        y = v / w
        # This holds for both images of a stereo pair.



        # Distortion model and coefficients
        # Ensure distortion_model is a string (sensor_msgs requires str)
        try:
            camera_info.distortion_model = str(intrinsics.distortion.model)
        except Exception:
            camera_info.distortion_model = 'plumb_bob'

        # Ensure we have exactly 5 distortion coefficients (pad/truncate as needed)
        try:
            coeffs = list(intrinsics.coeffs)
        except Exception:
            coeffs = [0.0] * 5

        # Pad or truncate to length 5
        if len(coeffs) < 5:
            coeffs = coeffs + [0.0] * (5 - len(coeffs))
        elif len(coeffs) > 5:
            coeffs = coeffs[:5]

        camera_info.d = [float(c) for c in coeffs]
        
        return camera_info

    def _convert_timestamp(self, realsense_timestamp_ms):
        """Convert RealSense timestamp to ROS2 timestamp and bag timestamp"""
        # Convert milliseconds to seconds
        timestamp_sec_float = realsense_timestamp_ms / 1000.0
        
        # Create ROS2 header timestamp
        timestamp_sec = int(timestamp_sec_float)
        timestamp_nanosec = int((timestamp_sec_float - timestamp_sec) * 1_000_000_000)
        
        # Create header timestamp
        from builtin_interfaces.msg import Time
        header_stamp = Time()
        header_stamp.sec = timestamp_sec
        header_stamp.nanosec = timestamp_nanosec
        
        # Create bag timestamp (nanoseconds since epoch)
        bag_timestamp = int(timestamp_sec_float * 1_000_000_000)
        
        return header_stamp, bag_timestamp

    def _write_message(self, topic_name, message, bag_timestamp):
        """Write message to bag with preserved timestamp"""
        serialized_data = serialize_message(message)
        self.writer.write(topic_name, serialized_data, bag_timestamp)
        # Track the written span so _validate_output has a denominator for its
        # rate checks.
        if self.first_bag_timestamp is None or bag_timestamp < self.first_bag_timestamp:
            self.first_bag_timestamp = bag_timestamp
        if self.last_bag_timestamp is None or bag_timestamp > self.last_bag_timestamp:
            self.last_bag_timestamp = bag_timestamp

    def convert(self):
        """Main conversion loop - process camera and IMU from separate pipelines"""
        print("\n=== Starting Direct Conversion ===")
        print("Processing camera frames and IMU data from separate pipelines...")
        
        try:
            # Process all camera frames
            print("\nPhase 1: Processing camera frames...")
            
            while True:
                try:
                    # Use longer timeout (5 seconds) under heavy system load
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    
                    # Process camera data - get frames by stream index to avoid mixing left/right
                    ir_frames = {}  # Dictionary: stream_index -> frame
                    has_camera_data = False
                    
                    # Extract IR frames by their stream index (1=left, 2=right)
                    for i in range(2):  # Try both positions 0 and 1
                        try:
                            ir_frame = frames.get_infrared_frame(i)
                            if ir_frame:
                                stream_idx = ir_frame.get_profile().stream_index()
                                ir_frames[stream_idx] = ir_frame
                                has_camera_data = True
                        except Exception as exc:
                            log.debug("could not read frame metadata: %s", exc)
                    
                    # Ensure we have both streams (1=left, 2=right)
                    if has_camera_data and 1 in ir_frames and 2 in ir_frames:
                        self._process_camera_frames(frames, ir_frames)
                
                except RuntimeError as e:
                    # Any timeout means stream has ended
                    print(f"  Camera stream ended (processed {self.frame_count} frames)")
                    break
            
            # Process all IMU frames - two passes: accel first, then gyro
            print("\nPhase 2: Processing IMU data...")
            
            if self.unite_imu_method == 2:
                # For interpolation: collect all accel first, then process gyro
                self._preprocess_accel_frames()
                self._process_gyro_frames()
            else:
                # For other methods: process framesets normally with timeout tolerance
                imu_frame_errors = 0
                max_errors = 10
                
                while imu_frame_errors < max_errors:
                    try:
                        # Use longer timeout (5 seconds) under heavy system load
                        frames = self.imu_pipeline.wait_for_frames(timeout_ms=5000)
                        self._process_imu_frames(frames)
                        
                        # Reset error counter on successful frame
                        imu_frame_errors = 0
                    
                    except RuntimeError as e:
                        # Increment counter and check limit
                        imu_frame_errors += 1
                        error_msg = str(e).lower()
                        
                        # Print diagnostic on first timeout
                        if imu_frame_errors == 1:
                            print(f"  Frame timeout (system may be busy) — retrying... [Error: {str(e)[:60]}]")
                        elif imu_frame_errors < max_errors:
                            pass  # Silent retries after first one
                        
                        # If we hit max consecutive timeouts, assume stream ended
                        if imu_frame_errors >= max_errors:
                            print(f"  IMU stream ended after {max_errors} consecutive timeouts (processed {self.imu_count} messages)")
                            break
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback
            traceback.print_exc()
        
        # Close the pipelines
        try:
            self.pipeline.stop()
        except Exception as exc:
            log.warning("failed to close a playback pipeline: %s", exc)
        try:
            self.imu_pipeline.stop()
        except Exception as exc:
            log.warning("failed to close a playback pipeline: %s", exc)
        
        # Close the bag writer
        self.writer.close()
        
        self._validate_output()

        print(f"\n=== Conversion Complete ===")
        print(f"Processed {self.frame_count} camera frames")
        print(f"Processed {self.imu_count} IMU messages")
        print(f"Output bag: {self.output_bag_path}")
        
        # Show bag info
        print(f"\nBag info:")
        # (`ros2 bag info <output>` was shelled out here, with the path
        #  interpolated unquoted. Run it yourself if you want the summary.)

    def _process_camera_frames(self, frames, ir_frames):
        """Process camera frames with timestamp preservation
        
        ir_frames is a dict mapping stream_index (1=left, 2=right) to frame objects
        """
        # Get original RealSense timestamp
        realsense_timestamp_ms = frames.get_timestamp()
        header_stamp, bag_timestamp = self._convert_timestamp(realsense_timestamp_ms)
        
        # Create header
        header = Header()
        header.stamp = header_stamp
        header.frame_id = "camera_link"
        
        # Process left camera (stream index 1)
        if 1 in ir_frames:
            ir_frame = ir_frames[1]
            # Verify this is actually the left camera
            if ir_frame.get_profile().stream_index() != 1:
                print(f"WARNING: Stream index mismatch for left camera! Expected 1, got {ir_frame.get_profile().stream_index()}")
            # Convert frame to numpy array
            ir_image = np.asanyarray(ir_frame.get_data())
            
            # Ensure proper format
            if len(ir_image.shape) == 1:
                frame_height = ir_frame.get_height()
                frame_width = ir_frame.get_width()
                ir_image = ir_image.reshape(frame_height, frame_width)
            
            if ir_image.dtype != np.uint8:
                ir_image = ir_image.astype(np.uint8)
            
            if not ir_image.flags['C_CONTIGUOUS']:
                ir_image = np.ascontiguousarray(ir_image)
            
            # Create image message
            image_msg = self.bridge.cv2_to_imgmsg(ir_image, encoding="mono8")
            image_msg.header = header
            
            # Create camera info message for left camera
            header.frame_id = "camera_infra1_optical_frame"
            camera_info = self._create_camera_info_message(self.left_intrinsics, header, camera_index=0)
            self._write_message('/camera/left/image_raw', image_msg, bag_timestamp)
            self._write_message('/camera/left/camera_info', camera_info, bag_timestamp)
        
        # Process right camera (stream index 2)
        if 2 in ir_frames:
            ir_frame = ir_frames[2]
            # Verify this is actually the right camera
            if ir_frame.get_profile().stream_index() != 2:
                print(f"WARNING: Stream index mismatch for right camera! Expected 2, got {ir_frame.get_profile().stream_index()}")
            # Convert frame to numpy array
            ir_image = np.asanyarray(ir_frame.get_data())
            
            # Ensure proper format
            if len(ir_image.shape) == 1:
                frame_height = ir_frame.get_height()
                frame_width = ir_frame.get_width()
                ir_image = ir_image.reshape(frame_height, frame_width)
            
            if ir_image.dtype != np.uint8:
                ir_image = ir_image.astype(np.uint8)
            
            if not ir_image.flags['C_CONTIGUOUS']:
                ir_image = np.ascontiguousarray(ir_image)
            
            # Create image message
            image_msg = self.bridge.cv2_to_imgmsg(ir_image, encoding="mono8")
            image_msg.header = header
            
            # Create camera info message for right camera
            header.frame_id = "camera_infra2_optical_frame"
            camera_info = self._create_camera_info_message(self.right_intrinsics, header, camera_index=1)
            self._write_message('/camera/right/image_raw', image_msg, bag_timestamp)
            self._write_message('/camera/right/camera_info', camera_info, bag_timestamp)
        
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            print(f"Processed {self.frame_count} camera frames, {self.imu_count} IMU messages...")

    def _build_accel_index(self):
        """Sort the accelerometer history once, for bisecting.

        The original sorted the whole history and scanned it linearly on every
        gyroscope sample. The pruning that would have bounded that only ran in
        a branch the default configuration never takes, so with ~22k samples of
        each it was ~22k sorts over ~22k keys. Sorting once and bisecting gives
        the same answer.
        """
        self._accel_ts = sorted(self.accel_history)

    def _interpolate_accel(self, target_timestamp_ms):
        """Accelerometer reading at a gyroscope sample time, linearly interpolated.

        Returns the nearest sample when the target falls outside the history,
        which is what the original did, and None if there is no history at all.
        """
        import bisect

        if not self.accel_history:
            return None
        if getattr(self, "_accel_ts", None) is None:
            self._build_accel_index()
        ts = self._accel_ts

        i = bisect.bisect_left(ts, target_timestamp_ms)
        if i < len(ts) and ts[i] == target_timestamp_ms:
            return self.accel_history[ts[i]]
        lower = ts[i - 1] if i > 0 else None
        upper = ts[i] if i < len(ts) else None
        if lower is None:
            return self.accel_history[upper] if upper is not None else None
        if upper is None:
            return self.accel_history[lower]

        a, b = self.accel_history[lower], self.accel_history[upper]
        span = upper - lower
        if span == 0:
            return a
        f = (target_timestamp_ms - lower) / span
        return {k: a[k] + f * (b[k] - a[k]) for k in ("x", "y", "z")}

    # librealsense raises RuntimeError both for a genuine end-of-playback and
    # for a transient timeout, and the two are indistinguishable from the
    # exception alone. The camera loop already allows 5 s "under heavy system
    # load"; the IMU loops used 1 s and treated the first timeout as the end of
    # the stream, so one slow read silently truncated the whole IMU collection.
    #
    # That is what made re-converting produce ~30 Hz of IMU instead of 400: not
    # a change in the data, just a machine busy enough to miss a 1 s deadline.
    IMU_TIMEOUT_MS = 5000
    IMU_MAX_CONSECUTIVE_TIMEOUTS = 3

    def _imu_wait(self, pipeline):
        """Pull the next motion frameset, retrying transient timeouts.

        Raises RuntimeError only after several consecutive timeouts, which is
        then taken to mean the playback really has ended.
        """
        last = None
        for attempt in range(self.IMU_MAX_CONSECUTIVE_TIMEOUTS):
            try:
                return pipeline.wait_for_frames(timeout_ms=self.IMU_TIMEOUT_MS)
            except RuntimeError as exc:
                last = exc
                log.debug("IMU wait timed out (attempt %d/%d)",
                          attempt + 1, self.IMU_MAX_CONSECUTIVE_TIMEOUTS)
        raise last

    def _preprocess_accel_frames(self):
        """Pre-process and collect ALL accel frames into history
        
        Used for interpolation method: store all accel data before processing gyro.
        """
        print("  Pre-processing accel data...")
        accel_count = 0
        
        # Restart IMU pipeline for second pass
        self.imu_pipeline.stop()
        
        imu_pipeline = rs.pipeline()
        imu_config = rs.config()
        
        imu_config.enable_device_from_file(self.bag_path, repeat_playback=False)
        imu_config.enable_stream(rs.stream.accel, 0, rs.format.motion_xyz32f, 400)
        
        profile = imu_pipeline.start(imu_config)
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)
        
        try:
            while True:
                frames = self._imu_wait(imu_pipeline)
                
                for i in range(frames.size()):
                    frame = frames[i]
                    if frame.get_profile().stream_type() == rs.stream.accel:
                        accel_motion_frame = frame.as_motion_frame()
                        ts_ms = accel_motion_frame.get_timestamp()
                        accel_data = accel_motion_frame.get_motion_data()
                        
                        self.accel_history[ts_ms] = {
                            'x': float(accel_data.x),
                            'y': float(accel_data.y),
                            'z': float(accel_data.z),
                        }
                        accel_count += 1
        except RuntimeError:
            pass
        
        imu_pipeline.stop()
        print(f"  Collected {accel_count} accel samples for interpolation")

    def _process_gyro_frames(self):
        """Process gyro frames and interpolate accel data
        
        Read gyro frames and create unified IMU messages with interpolated accel.
        """
        print("  Processing gyro data with interpolated accel...")
        
        # Restart pipeline for gyro-only pass
        imu_pipeline = rs.pipeline()
        imu_config = rs.config()
        
        imu_config.enable_device_from_file(self.bag_path, repeat_playback=False)
        imu_config.enable_stream(rs.stream.gyro, 0, rs.format.motion_xyz32f, 400)
        
        profile = imu_pipeline.start(imu_config)
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)
        
        try:
            while True:
                frames = self._imu_wait(imu_pipeline)
                
                for i in range(frames.size()):
                    frame = frames[i]
                    if frame.get_profile().stream_type() == rs.stream.gyro:
                        gyro_motion_frame = frame.as_motion_frame()
                        gyro_timestamp_ms = gyro_motion_frame.get_timestamp()
                        gyro_data = gyro_motion_frame.get_motion_data()
                        
                        header_stamp, bag_timestamp = self._convert_timestamp(gyro_timestamp_ms)
                        
                        # Create unified IMU message
                        imu_msg = Imu()
                        imu_msg.header.stamp = header_stamp
                        imu_msg.header.frame_id = "imu_link"
                        
                        # Add gyro data (always available)
                        imu_msg.angular_velocity.x = float(gyro_data.x)
                        imu_msg.angular_velocity.y = float(gyro_data.y)
                        imu_msg.angular_velocity.z = float(gyro_data.z)
                        
                        # Interpolate accel to gyro timestamp
                        accel_interp = self._interpolate_accel(gyro_timestamp_ms)
                        if accel_interp:
                            imu_msg.linear_acceleration.x = accel_interp['x']
                            imu_msg.linear_acceleration.y = accel_interp['y']
                            imu_msg.linear_acceleration.z = accel_interp['z']
                        
                        # Set covariances
                        imu_msg.linear_acceleration_covariance[0] = -1.0
                        imu_msg.angular_velocity_covariance[0] = -1.0
                        imu_msg.orientation_covariance[0] = -1.0
                        
                        self._write_message('/imu', imu_msg, bag_timestamp)
                        self.imu_count += 1
        except RuntimeError:
            pass
        
        imu_pipeline.stop()


    # Expected stream rates for a D455 recording made by the RealSense Viewer:
    # infrared stereo at 30 Hz, accelerometer and gyroscope at 400 Hz.
    EXPECTED_IMAGE_HZ = 30.0
    EXPECTED_IMU_HZ = 400.0
    RATE_TOLERANCE = 0.5          # accept half the nominal rate before failing

    def _validate_output(self):
        """Check the written bag actually contains the streams it should.

        This exists because the failure it catches was real and silent: a
        one-second IMU timeout, treated as end-of-stream, dropped roughly 94 %
        of the inertial data while the converter reported success. That specific
        cause is fixed (see `_imu_wait`), but the check stays, because any
        future truncation of a playback stream would look exactly the same --
        a bag that opens fine, plays fine, and quietly starves the
        visual-inertial SLAM.

        Nothing downstream noticed. The driver script checked only that the
        output files existed, so a bag holding a fraction of its IMU was
        reported as a successful conversion.

        Raises rather than warns: a truncated bag is not a degraded result, it
        is a different measurement.
        """
        if self.frame_count == 0:
            raise RuntimeError("no camera frames were written; the output bag is empty")
        if self.first_bag_timestamp is None:
            log.warning("no messages written; skipping rate validation")
            return

        duration_s = (self.last_bag_timestamp - self.first_bag_timestamp) / 1e9
        if duration_s <= 0:
            log.warning("written span is zero; skipping rate validation")
            return

        image_hz = self.frame_count / duration_s
        imu_hz = self.imu_count / duration_s
        log.info("written bag spans %.1f s: %d image sets (%.1f Hz), %d IMU (%.1f Hz)",
                 duration_s, self.frame_count, image_hz, self.imu_count, imu_hz)

        problems = []
        if image_hz < self.EXPECTED_IMAGE_HZ * self.RATE_TOLERANCE:
            problems.append(
                "images at %.1f Hz over %.1f s, expected about %.0f Hz (%d written)"
                % (image_hz, duration_s, self.EXPECTED_IMAGE_HZ, self.frame_count)
            )
        if imu_hz < self.EXPECTED_IMU_HZ * self.RATE_TOLERANCE:
            problems.append(
                "IMU at %.1f Hz over %.1f s, expected about %.0f Hz (%d written). The "
                "IMU playback delivered only a fraction of its samples; see "
                "extract/README.md."
                % (imu_hz, duration_s, self.EXPECTED_IMU_HZ, self.imu_count)
            )
        if problems:
            raise RuntimeError(
                "the converted bag does not contain the expected streams:\n  - "
                + "\n  - ".join(problems)
            )

def main():
    parser = argparse.ArgumentParser(description='Direct RealSense to ROS2 Bag Converter with True Timestamp Preservation')
    parser.add_argument('bag_file', help='Path to RealSense .bag file')
    parser.add_argument('-o', '--output', default='realsense_converted',
                        help='Output bag directory name')
    parser.add_argument('--unite-imu', type=int, default=2, choices=[0, 1, 2],
                        help='IMU fusion method: 0=none (separate messages), 1=copy (last accel), 2=linear_interpolation (default: 2)')
    
    args = parser.parse_args()
    
    try:
        # Create converter
        converter = DirectRealSenseConverter(
            realsense_bag_path=args.bag_file,
            output_bag_path=args.output,
            unite_imu_method=args.unite_imu
        )
        
        # Run conversion
        converter.convert()
        
        print(f"\n✓ Conversion completed successfully!")
        print(f"✓ Original timestamps preserved in both message headers and bag timestamps")
        print(f"✓ IMU messages combined using method {args.unite_imu}")
        print(f"✓ Ready for Isaac ROS Visual SLAM")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())