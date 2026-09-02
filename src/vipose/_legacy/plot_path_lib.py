"""
plot_path_lib.py - Modular library for hand-eye calibration and synchronization.

This module extracts core functions from plot_path.py for reuse in batch processing.
Provides parameterized versions of all calibration, synchronization, and visualization functions.
"""

import os
import sys
import pandas as pd
import numpy as np

# Configure matplotlib - check for display first
import matplotlib

from synchronize_vector_objective import compare_objectives
if 'DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ:
    matplotlib.use('TkAgg')  # Force TkAgg interactive backend if display available
    import matplotlib.pyplot as plt
    plt.ion()  # Enable interactive mode
else:
    matplotlib.use('Agg')  # Use Agg backend for headless environments
    import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation
from SynchronizationNew import *
import scipy.optimize

# Disable Qt platform plugin issues with OpenCV
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2


def _has_display():
    """Check if X11 display is available."""
    return 'DISPLAY' in os.environ


class CalibrationConfig:
    """Configuration for calibration workflow."""
    def __init__(self):
        self.velocity_threshold = 1e-4  # mm/ms threshold for stagnant pose detection
        self.window_size = 4  # frames for smoothing window
        self.discard_fraction = 0.0  # first 10% of data for lag estimation
        
        # ===== Hand-Eye Calibration Method Selection =====
        """ Option 0 - calibrateRobotWorldHandEye - AX=ZB https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#ga7874b7e33b597c56994974a7ee532285 """
        self.calibration_option = 0; self.calib_method = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH  # ok'ish, fast
        # self.calibration_option = 0; self.calib_method = cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI  # Not good
        
        """ Option 1 - calibrateHandEye - AX=XB https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b """
        # self.calibration_option = 1; self.calib_method = cv2.CALIB_HAND_EYE_TSAI
        # self.calibration_option = 1; self.calib_method = cv2.CALIB_HAND_EYE_PARK
        # self.calibration_option = 1; self.calib_method = cv2.CALIB_HAND_EYE_HORAUD  # also good
        # self.calibration_option = 1; self.calib_method = cv2.CALIB_HAND_EYE_ANDREFF  # simultaneous rotation and translation (not good)
        # self.calibration_option = 1; self.calib_method = cv2.CALIB_HAND_EYE_DANIILIDIS  # simultaneous rotation and translation (good)
        
        self.marker_coordinate_option = 2  # 0, 1, or 2 for marker intrinsic frame computation
        self.interactive = True  # Always enable plots when display available
        

class CalibrationResult:
    """Container for calibration results."""
    def __init__(self):
        self.T_zed2rb = None  # ZED to rigid body transformation
        self.T_zed02vicon = None  # ZED0 to VICON transformation
        self.T_zed2marker_intrinsic = None  # ZED to marker-intrinsic frame (persistent)
        self.marker_centroid = None  # Centroid of marker layout in VICON frame
        self.x_axis = None  # X-axis direction in marker-intrinsic frame
        self.y_axis = None  # Y-axis direction in marker-intrinsic frame
        self.z_axis = None  # Z-axis direction in marker-intrinsic frame
        self.df_zed_in_vicon = None  # ZED poses in VICON frame
        self.df_zed_as_marker = None  # ZED poses as markers in VICON frame
        self.df_rb_interpolated = None  # VICON rigid body data (interpolated)
        self.opt_lag = None  # Optimized temporal lag
        self.calib_method = None  # Which calibration method was used
        # Distance statistics
        self.distances_zed2zed_marker = None  # ZED to ZED-marker distances
        self.distances_zed2rb = None  # ZED to RB distances
        self.distances_zed_marker2rb = None  # ZED-marker to RB distances
        

def process_zed_vicon_calibration(zed_csv_path, mocap_csv_path=None, config=None, verbose=True, 
                                   pre_cropped_time_range=None, pre_computed_lag=None):
    """
    Main calibration workflow: load data, synchronize, and compute hand-eye calibration.
    
    Args:
        zed_csv_path: Path to ZED tracking CSV
        mocap_csv_path: Path to VICON mocap CSV (if None, inferred from zed_csv_path)
        config: CalibrationConfig instance (uses defaults if None)
        verbose: Print progress messages
        pre_cropped_time_range: dict with keys 't_min' and 't_max' (ms)
            - If provided, crop ZED data to this window before synchronization
            - Used in unified multi-pipeline calibration to ensure all pipelines use same VICON window
            - If None, uses full ZED data (backward compatible, independent mode)
        pre_computed_lag: float (optional)
            - If provided, skip temporal lag estimation and use this value directly (ms)
            - Used in unified multi-pipeline mode after Stages 1-2 refinement
            - If None, estimate lag via cross-correlation and optimization (default)
        
    Returns:
        CalibrationResult instance with all computed transformations
    """
    if config is None:
        config = CalibrationConfig()
    
    result = CalibrationResult()
    result.calib_method = config.calib_method  # Store which calibration method was used for reference
    result.calibration_option = config.calibration_option
    
    # 1. Load and preprocess ZED data
    if verbose:
        print(f"\n=== Loading ZED data from {zed_csv_path} ===")
    test_name = os.path.basename(zed_csv_path).split("_")[0]
    
    df_ZED = get_df_ZED(zed_csv_path)
    df_ZED = preprocess_zed_data(
        df_ZED,
        ZED_replace_stagnant_with_interpolation=False,
        ZED_remove_stagnant_poses=False,
        ZED_smooth_data=False,
        ZED_semantic_rebase=False,
        # rebase_at_first_dict={"Spatial_Memory": ["KNOWN_MAP", "cuVSLAM"]},
        rebase_at_first_dict={"Spatial_Memory": ["MAP_UPDATE", "cuVSLAM"]}, #< mAKE SURE THIS MATCHES THE REBASE CONDITION USED IN plot_path_unified.py
        velocity_threshold=config.velocity_threshold,
        window_size=config.window_size
    )
    if verbose:
        print(f"Loaded {len(df_ZED)} frames from ZED data")
    
    # Apply pre-cropping if provided (unified multi-pipeline mode)
    if pre_cropped_time_range is not None:
        if verbose:
            print(f"Applying pre-cropped time window: {pre_cropped_time_range['t_min']:.2f} - {pre_cropped_time_range['t_max']:.2f} ms")
        df_ZED = crop_tracking_data_to_window(
            df_ZED,
            pre_cropped_time_range['t_min'],
            pre_cropped_time_range['t_max'],
            verbose=verbose
        )
        if verbose:
            print(f"After cropping: {len(df_ZED)} frames remain")
    
    # 2. Load and process VICON mocap data
    if mocap_csv_path is None:
        mocap_csv_path = os.path.join(os.path.dirname(zed_csv_path), f"{test_name}_mocap.csv")
    
    df_mocap = None
    df_rb = None
    
    if os.path.exists(mocap_csv_path):
        if verbose:
            print(f"Loading VICON mocap data from {mocap_csv_path}")
        df_mocap = load_mocap(mocap_csv_path)
        df_rb, _ = calculate_vicon_rigid_body(df_mocap, print_stats=verbose)
        if verbose:
            print(f"Loaded {len(df_mocap.Marker[0].X)} frames from VICON mocap data")
        
        if config.interactive and _has_display():
            plot_vicon_rigid_body(df_rb, title='VICON Rigid Body Trajectory and Orientation')
    else:
        if verbose:
            print(f"Warning: VICON mocap CSV file '{mocap_csv_path}' not found. Skipping VICON data.")
        return result  # Return early if no VICON data

    # 3. Synchronize ZED and VICON data
    if verbose:
        print("\n=== Synchronizing ZED and VICON data ===")
    
    if config.interactive and _has_display():
        plot_zed_vicon_orientation(df_ZED, df_rb, title="ZED [Oz] and VICON rigid-body [Ovicon] Orientations Over Time - Before Synchronization")
        plot_absolute_rotation_angle(df_ZED, df_rb, title="Absolute Rotation Angle and Rate (ZED & VICON) - Before Synchronization")
    
    # Estimate temporal offset (skip if pre_computed_lag is provided)
    if pre_computed_lag is not None:
        result.opt_lag = pre_computed_lag
        if verbose:
            print(f"Using pre-computed lag: {result.opt_lag} ms (skipping lag estimation)")
    else:
        # Estimate temporal offset via cross-correlation
        initial_lag = estimate_temporal_offset_zed_vicon(df_ZED, df_rb, discard_fraction=config.discard_fraction)
        if verbose:
            print(f"Initial estimated lag: {initial_lag} ms")
        
        # Optimize lag estimation
        optimize_lag = True  # Set to True to enable optimization of lag estimation
        if optimize_lag:
            itter_count = 0
            def objective_function(lag):
                nonlocal itter_count
                itter_count += 1
                rms = synchronize_zed_vicon_trial(df_ZED, df_rb, lag, discard_fraction=config.discard_fraction)
                if verbose and itter_count % 5 == 0:
                    print(f"  Iteration {itter_count}: lag={lag} ms, RMS error={rms:.8f} rad/ms")
                return rms
            
            opt_lag = scipy.optimize.fmin(func=objective_function, x0=initial_lag, disp=0, maxiter=50, xtol=1e-6)
        else:   
            opt_lag = initial_lag
        result.opt_lag = opt_lag[0] if isinstance(opt_lag, np.ndarray) else opt_lag
        if verbose:
            print(f"Optimized lag: {result.opt_lag} ms")

    # compare_objectives(df_ZED, df_rb, result.opt_lag)

    # Synchronize VICON to ZED timestamps
    df_mocap_interpolated = synchronize_zed_vicon(df_ZED, df_rb, df_mocap, result.opt_lag)
    df_rb_interpolated, _ = calculate_vicon_rigid_body(df_mocap_interpolated, print_stats=verbose)
    result.df_rb_interpolated = df_rb_interpolated
    
    # plot df_rb_interpolated.Residuals to check quality of rigid body fitting
    if config.interactive and _has_display():
        #plot here
        plt.figure(figsize=(10, 4))
        plt.plot(df_rb_interpolated.Time_ms, df_rb_interpolated.Residuals, label='Rigid Body Fit Residuals (interpolated)')
        # plt.plot(df_rb.Time_ms, df_rb.Residuals, label='Rigid Body Fit Residuals')
        plt.xlabel('Time (ms)')
        plt.ylabel('Residual (mm)')
        plt.title('Rigid Body Fitting Residuals')
        plt.legend()
        plt.show()

    if config.interactive and _has_display():
        plot_absolute_rotation_angle(df_ZED, df_rb_interpolated, title="Absolute Rotation Angle and Rate (ZED & VICON) - After Synchronization")
    
    # 4. Hand-Eye Calibration
    if verbose:
        print("\n=== Performing hand-eye calibration ===")
    
    T_rb2vicon = df_to_T(df_rb_interpolated)
    T_zed2zed0 = df_to_T(df_ZED)
    
    # Extract rotations and translations
    T_vicon2rb = [np.linalg.inv(T) for T in T_rb2vicon] # B matrix
    t_rb2vicon, R_rb2vicon = split_rt(T_rb2vicon)
    t_vicon2rb, R_vicon2rb = split_rt(T_vicon2rb)
    
    T_zed02zed = [np.linalg.inv(T) for T in T_zed2zed0] # A matrix
    t_zed02zed, R_zed02zed = split_rt(T_zed02zed)

    excitation = motion_axis_excitation(T_rb2vicon)
    e = excitation["axis_scatter_eigs"]
    print(f"  axis excitation: eigs=[{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}]  "
      f"e2/e1={e[1]/e[0]:.4f}  aniso={excitation['axis_anisotropy']:.2f}  "
      f"span={excitation['rotation_span_deg']:.2f} deg  n={excitation['n_used']}")

    # Filter invalid rotation matrices
    valid_indices = []
    for i, (R1, R2) in enumerate(zip(R_zed02zed, R_vicon2rb)):
        det1 = np.linalg.det(R1)
        det2 = np.linalg.det(R2)
        if not (np.isnan(det1) or np.isnan(det2) or np.abs(det1) < 1e-10 or np.abs(det2) < 1e-10):
            valid_indices.append(i)
    
    if verbose:
        print(f"Using {len(valid_indices)}/{len(R_zed02zed)} valid rotation matrices")
    
    # Keep only valid data
    R_zed02zed_valid = [R_zed02zed[i] for i in valid_indices]
    t_zed02zed_valid = [t_zed02zed[i] for i in valid_indices]
    R_vicon2rb_valid = [R_vicon2rb[i] for i in valid_indices]
    t_vicon2rb_valid = [t_vicon2rb[i] for i in valid_indices]
    t_rb2vicon_valid = [t_rb2vicon[i] for i in valid_indices]
    R_rb2vicon_valid = [R_rb2vicon[i] for i in valid_indices]
    
    # Calibrate using OpenCV
    try:
        if config.calibration_option == 0:
            # Option 0: calibrateRobotWorldHandEye (AX=ZB)
            [R_vicon2zed0, t_vicon2zed0, R_rb2zed, t_rb2zed] = cv2.calibrateRobotWorldHandEye(
                R_zed02zed_valid, t_zed02zed_valid,
                R_vicon2rb_valid, t_vicon2rb_valid,
                method=config.calib_method
            )
            T_vicon2zed0 = np.vstack((np.hstack((R_vicon2zed0, t_vicon2zed0.reshape(3, 1))), [0, 0, 0, 1]))
            T_zed02vicon = np.linalg.inv(T_vicon2zed0)
            
            T_rb2zed = np.vstack((np.hstack((R_rb2zed, t_rb2zed.reshape(3, 1))), [0, 0, 0, 1]))
            T_zed2rb = np.linalg.inv(T_rb2zed)
        else:
            # Option 1: calibrateHandEye (AX=XB)
            [R_zed2rb, t_zed2rb] = cv2.calibrateHandEye(
                R_rb2vicon_valid, t_rb2vicon_valid,
                R_zed02zed_valid, t_zed02zed_valid,
                method=config.calib_method
            )
            T_zed2rb = np.vstack((np.hstack((R_zed2rb, t_zed2rb.reshape(3, 1))), [0, 0, 0, 1]))
            
            # Compute ZED0 -> VICON
            T_candidates = []
            for i in range(len(T_rb2vicon)):
                M_i = T_rb2vicon[i] @ T_zed2rb @ T_zed02zed[i]
                T_candidates.append(M_i)
            T_zed02vicon = average_pose(T_candidates)
        
        result.T_zed2rb = T_zed2rb
        result.T_zed02vicon = T_zed02vicon
        
        if verbose:
            print("Hand-eye calibration successful")
    except Exception as e:
        if verbose:
            print(f"Calibration failed: {e}")
        return result
    
    # 5. Compute ZED poses in VICON frame
    T_zed_in_vicon = [T_zed02vicon @ T for T in T_zed2zed0]
    result.df_zed_in_vicon = T_to_df(T_zed_in_vicon).assign(Time_ms=df_ZED["Time_ms"], Frame=df_ZED["Frame"])
    
    T_zed_as_marker = [T @ T_zed2rb for T in T_rb2vicon]
    result.df_zed_as_marker = T_to_df(T_zed_as_marker).assign(Time_ms=df_rb_interpolated["Time_ms"], Frame=df_rb_interpolated["Frame"])
    
    # 6. Compute marker-intrinsic coordinate system
    if verbose:
        print("\n=== Computing marker-intrinsic coordinate system ===")
    
    mocap_markers = df_mocap_interpolated.Marker
    
    m_top_pos = np.array([mocap_markers[0].X[0], mocap_markers[0].Y[0], mocap_markers[0].Z[0]])
    m_bl_pos = np.array([mocap_markers[1].X[0], mocap_markers[1].Y[0], mocap_markers[1].Z[0]])
    m_br_pos = np.array([mocap_markers[2].X[0], mocap_markers[2].Y[0], mocap_markers[2].Z[0]])
    m_fr_pos = np.array([mocap_markers[3].X[0], mocap_markers[3].Y[0], mocap_markers[3].Z[0]])
    m_fl_pos = np.array([mocap_markers[4].X[0], mocap_markers[4].Y[0], mocap_markers[4].Z[0]])
    
    marker_centroid = np.mean([m_top_pos, m_bl_pos, m_br_pos, m_fr_pos, m_fl_pos], axis=0)
    result.marker_centroid = marker_centroid
    
    # Compute axes based on selected option
    marker_coordinate_opt = config.marker_coordinate_option
    if marker_coordinate_opt == 2:
        # Y-axis: from back midpoint to front midpoint
        back_midpoint = (m_bl_pos + m_br_pos) / 2
        front_midpoint = (m_fl_pos + m_fr_pos) / 2
        y_vec = front_midpoint - back_midpoint
        y_axis = y_vec / np.linalg.norm(y_vec)
        
        # X-axis: from back-left to back-right
        x_vec = np.cross(y_axis, m_top_pos-back_midpoint)
        # x_vec = m_br_pos - m_bl_pos
        x_axis = x_vec / np.linalg.norm(x_vec)
        
        # Z-axis: cross(X, Y)
        z_vec = np.cross(x_axis, y_axis)
        z_axis = z_vec / np.linalg.norm(z_vec)
    # deleted
    # else: 
    #     # Default to option 2 for simplicity
    #     back_midpoint = (m_bl_pos + m_br_pos) / 2
    #     front_midpoint = (m_fl_pos + m_fr_pos) / 2
    #     y_vec = front_midpoint - back_midpoint
    #     y_axis = y_vec / np.linalg.norm(y_vec)
        
    #     x_vec = m_br_pos - m_bl_pos
    #     x_axis = x_vec / np.linalg.norm(x_vec)
        
    #     z_vec = np.cross(x_axis, y_axis)
    #     z_axis = z_vec / np.linalg.norm(z_vec)
    
    result.x_axis = x_axis
    result.y_axis = y_axis
    result.z_axis = z_axis
    
    # Construct transformation to marker-intrinsic frame
    R_vicon2marker_intrinsic = np.column_stack([x_axis, y_axis, z_axis]).T
    R_rb2marker_intrinsic = R_vicon2marker_intrinsic
    T_rb2marker_intrinsic = np.vstack((np.hstack((R_rb2marker_intrinsic, np.zeros((3, 1)))), [0, 0, 0, 1]))
    
    T_zed2marker_intrinsic = T_rb2marker_intrinsic @ T_zed2rb
    result.T_zed2marker_intrinsic = T_zed2marker_intrinsic
    
    if verbose:
        print(f"Marker centroid (VICON): {marker_centroid}")
        print(f"T_zed2marker_intrinsic:\n{T_zed2marker_intrinsic}")
    
    # 7. Generate visualization plots
    if verbose:
        print("\n=== Generating validation plots ===")
    
    loop_closed_ind = df_ZED[df_ZED['Spatial_Memory'] == "LOOP_CLOSED"].index
    
    # Generate visualization plots with interactive display
    loop_closed_ind = df_ZED[df_ZED['Spatial_Memory'] == "LOOP_CLOSED"].index
    
    if verbose:
        print("\n=== Generating interactive validation plots ===")
    
    # Plot 1: Velocity
    plot_velocity(result.df_zed_in_vicon, result.df_zed_as_marker, loop_closed_ind, 
                 title="Camera Frame Velocity in VICON Frame of Reference")
    plt.draw()
    plt.pause(0.001)
    
    # Plot 2: Distance ZED vs ZED-marker
    _, distances_zed2zed_marker = plot_df_to_df_distance(result.df_zed_in_vicon, result.df_zed_as_marker, loop_closed_ind,
                          title="Projection Translational Error")
    result.distances_zed2zed_marker = distances_zed2zed_marker
    if verbose:
        print(f"Distance between ZED and ZED-marker in VICON Frame of Reference stats (mm): "
            f"min={distances_zed2zed_marker.min():.6f}, "
            f"max={distances_zed2zed_marker.max():.6f}, "
            f"mean={distances_zed2zed_marker.mean():.6f}, "
            f"std={distances_zed2zed_marker.std():.6f}")
    plt.draw()
    plt.pause(0.001)
    
    # Plot 3: Distance ZED vs RB
    _, distances_zed2rb = plot_df_to_df_distance(result.df_zed_in_vicon, df_rb_interpolated, loop_closed_ind,
                          title="Distance between ZED and RB in VICON Frame of Reference")
    result.distances_zed2rb = distances_zed2rb
    if verbose:
        print(f"Distance between ZED and RB in VICON Frame of Reference stats (mm): "
            f"min={distances_zed2rb.min():.6f}, "
            f"max={distances_zed2rb.max():.6f}, "
            f"mean={distances_zed2rb.mean():.6f}, "
            f"std={distances_zed2rb.std():.6f}")
    plt.draw()
    plt.pause(0.001)
    
    # Plot 4: Distance ZED-marker vs RB
    _, distances_zed_marker2rb = plot_df_to_df_distance(result.df_zed_as_marker, df_rb_interpolated, loop_closed_ind,
                          title="Distance between ZED-marker and RB in VICON Frame of Reference")
    result.distances_zed_marker2rb = distances_zed_marker2rb
    if verbose:
        print(f"Distance between ZED-marker and RB in VICON Frame of Reference stats (mm): "
            f"min={distances_zed_marker2rb.min():.6f}, "
            f"max={distances_zed_marker2rb.max():.6f}, "
            f"mean={distances_zed_marker2rb.mean():.6f}, "
            f"std={distances_zed_marker2rb.std():.6f}")
    plt.draw()
    plt.pause(0.001)

     # Plot 5: Plot Rotation angle between ZED in vicon and ZED as marker
    result.angles = plot_df_to_df_angles(result.df_zed_in_vicon, result.df_zed_as_marker,
                          title="Rotation Angle between ZED and ZED-marker in VICON Frame of Reference")
    plt.draw()
    plt.pause(0.001)
    if verbose:
        print(f"Rotation angle between ZED and ZED-marker in VICON Frame of Reference stats (degrees): "
            f"min={np.array(result.angles).min():.6f}, "
            f"max={np.array(result.angles).max():.6f}, "
            f"mean={np.array(result.angles).mean():.6f}, "
            f"std={np.array(result.angles).std():.6f}")

    
    # Plot 6: Orientation
    temp_plt = plot_zed_vicon_orientation(result.df_zed_in_vicon, result.df_zed_as_marker)
    temp_plt.suptitle("ZED and ZED-marker Orientation Over Time - In VICON frame")
    temp_plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    



    # Plot 6: ZED path in VICON frame of reference with interactive slider
    try:
        plot_zed_path_in_vicon_frame_of_reference(result.df_zed_in_vicon, df_mocap_interpolated, 
                                                  T_zed_as_marker, df_rb_interpolated, 
                                                  result.df_zed_as_marker, 
                                                #   title="ZED Path in VICON Frame of Reference")
                                                    title=r"$\left\|{}^{V}\mathbf{T}_{W}\,{}^{W}\mathbf{T}_{C}(t)-{}^{V}\mathbf{T}_{M}(t)\,{}^{M}\mathbf{T}_{C}\right\|$")
    except Exception as e:
        if verbose:
            print(f"Interactive plot error: {e} (skipping this plot)")
    
    # Plot 7: VICON rigid body
    plot_vicon_rigid_body(df_rb_interpolated, title="VICON Rigid Body (Fitted) Trajectory and Orientation - Interpolated")
    # plt.draw()
    # plt.pause(0.001)


   
    ################# Visualization: Plot markers and ZED frame in marker-intrinsic frame #################
    # Call wrapper function to compute and visualize marker-intrinsic coordinate system
    # marker_viz_result = plot_markers_and_zed_in_marker_frame(
    #     df_mocap_interpolated.Marker,
    #     T_zed2rb,
    #     marker_coordinate_opt=config.marker_coordinate_option,
    #     verbose=verbose
    # )
    
    ####################### End of additional ##########################
    return result
    



# ============= Unified Multi-Pipeline Calibration Utilities =============

def compute_common_vicon_window(offsets_ms, time_ranges_local):
    """
    Compute the common VICON time window across all pipelines.
    
    All pipelines are synchronized to VICON using their respective offsets.
    This function finds the intersection of their valid time ranges in VICON absolute time,
    then back-projects to each pipeline's local time.
    
    Args:
        offsets_ms (dict): {pipeline_num: offset_ms}
            - offset_ms: temporal offset between pipeline and VICON (ms)
            - Positive: VICON ahead of pipeline; Negative: pipeline ahead of VICON
        time_ranges_local (dict): {pipeline_num: (t_min_local, t_max_local)}
            - Local time range for each pipeline tracking data
    
    Returns:
        dict with keys:
            - 'vicon_window': (vicon_t_min, vicon_t_max) — common VICON time window
            - 'local_windows': {pipeline_num: (t_min_local, t_max_local)}
            - 'valid': bool — True if intersection is non-empty
            - 'stats': dict with details (useful for logging)
    
    Raises:
        ValueError: If pipelines have no time overlap in VICON frame
    """
    # Convert each pipeline's local time range to VICON absolute time
    vicon_ranges = {}
    for pipeline_num, (t_min_local, t_max_local) in time_ranges_local.items():
        offset = offsets_ms[pipeline_num]
        # Convert to VICON: subtract offset (offset is lag of pipeline w.r.t. VICON)
        vicon_t_min = t_min_local - offset
        vicon_t_max = t_max_local - offset
        vicon_ranges[pipeline_num] = (vicon_t_min, vicon_t_max)
    
    # Find intersection in VICON time
    vicon_t_min_common = max(vr[0] for vr in vicon_ranges.values())
    vicon_t_max_common = min(vr[1] for vr in vicon_ranges.values())
    
    valid = vicon_t_min_common <= vicon_t_max_common
    
    if not valid:
        stats = {
            'pipeline_vicon_ranges': vicon_ranges,
            'intersection_min': vicon_t_min_common,
            'intersection_max': vicon_t_max_common,
            'overlap': vicon_t_max_common - vicon_t_min_common
        }
        raise ValueError(
            f"No temporal overlap among pipelines in VICON frame. "
            f"Common window would be empty: [{vicon_t_min_common:.2f}, {vicon_t_max_common:.2f}]. "
            f"Details: {stats}"
        )
    
    # Back-project common VICON window to each pipeline's local time
    local_windows = {}
    for pipeline_num, offset in offsets_ms.items():
        t_min_local = vicon_t_min_common + offset
        t_max_local = vicon_t_max_common + offset
        local_windows[pipeline_num] = (t_min_local, t_max_local)
    
    stats = {
        'pipeline_vicon_ranges': vicon_ranges,
        'vicon_window': (vicon_t_min_common, vicon_t_max_common),
        'vicon_duration_ms': vicon_t_max_common - vicon_t_min_common,
        'local_windows': local_windows,
        'local_durations_ms': {
            pnum: (tw[1] - tw[0])
            for pnum, tw in local_windows.items()
        }
    }
    
    return {
        'vicon_window': (vicon_t_min_common, vicon_t_max_common),
        'local_windows': local_windows,
        'valid': valid,
        'stats': stats
    }


def crop_tracking_data_to_window(df_tracking, t_min_local, t_max_local, verbose=True):
    """
    Crop tracking data to a specified local time window.
    
    Args:
        df_tracking: DataFrame with 'Time_ms' column (ZED or other tracking)
        t_min_local: Minimum time (ms) in local frame
        t_max_local: Maximum time (ms) in local frame
        verbose: Print diagnostics
    
    Returns:
        Cropped DataFrame (reset index and update Time_ms from new first frame), with metadata:
            - 'crop_window': (t_min_local, t_max_local)
            - 'n_frames_original': int
            - 'n_frames_cropped': int
    """
    mask = (df_tracking['Time_ms'] >= t_min_local) & (df_tracking['Time_ms'] <= t_max_local)
    n_original = len(df_tracking)
    n_cropped = mask.sum()
    
    df_cropped = df_tracking[mask].reset_index(drop=True)
    if not df_cropped.empty:
        df_cropped['Time_ms_uncropped'] = df_cropped['Time_ms']  # Store original time for reference
        # df_cropped['Time_ms'] -= df_cropped['Time_ms'].iloc[0] # Rebase time to start at 0 for cropped data
    
    if verbose:
        print(f"Cropped tracking data: {n_cropped}/{n_original} frames retained "
              f"(window: {t_min_local:.2f} - {t_max_local:.2f} ms)")
    
    # Store metadata
    df_cropped.attrs['crop_window'] = (t_min_local, t_max_local)
    df_cropped.attrs['n_frames_original'] = n_original
    df_cropped.attrs['n_frames_cropped'] = n_cropped
    
    return df_cropped


def save_calibration_results(result, output_path):
    """
    Save calibration results to text file.
    
    Args:
        result: CalibrationResult instance
        output_path: Path to output text file
    """
    with open(output_path, 'w') as f:
        f.write("Hand-eye calibration results\n\n")
        f.write(f"Calibration method: {result.calib_method}\n\n")
        f.write(f"Calibration option: {result.calibration_option}\n\n")
        f.write(f"========== VICON-based Calibration (test-dependent) ==========\n")
        f.write(f"T_zed2rb (ZED to Rigid Body in VICON frame):\n{result.T_zed2rb}\n\n")
        f.write(f"T_zed02vicon (ZED0 to VICON):\n{result.T_zed02vicon}\n\n")
        if result.T_zed02vicon is not None:
            f.write(f"T_vicon2zed0 (VICON to ZED0):\n{np.linalg.inv(result.T_zed02vicon)}\n\n")
        f.write(f"========== Marker-Intrinsic Calibration (persistent across tests) ==========\n")
        f.write(f"Marker centroid (VICON frame):\n{result.marker_centroid}\n\n")
        f.write(f"Marker-intrinsic X-axis (VICON frame):\n{result.x_axis}\n\n")
        f.write(f"Marker-intrinsic Y-axis (VICON frame):\n{result.y_axis}\n\n")
        f.write(f"Marker-intrinsic Z-axis (VICON frame):\n{result.z_axis}\n\n")
        f.write(f"T_zed2marker_intrinsic (ZED to marker-intrinsic frame - PERSISTENT):\n{result.T_zed2marker_intrinsic}\n\n")
        f.write(f"Optimized temporal lag: {result.opt_lag} ms\n\n")
        f.write(f"========== Distance Validation Statistics (mm) ==========\n")
        if result.distances_zed2zed_marker is not None:
            f.write(f"Distance between ZED and ZED-marker in VICON Frame:\n")
            f.write(f"  min={result.distances_zed2zed_marker.min():.6f}\n")
            f.write(f"  max={result.distances_zed2zed_marker.max():.6f}\n")
            f.write(f"  mean={result.distances_zed2zed_marker.mean():.6f}\n")
            f.write(f"  std={result.distances_zed2zed_marker.std():.6f}\n\n")
        if result.distances_zed2rb is not None:
            f.write(f"Distance between ZED and RB in VICON Frame:\n")
            f.write(f"  min={result.distances_zed2rb.min():.6f}\n")
            f.write(f"  max={result.distances_zed2rb.max():.6f}\n")
            f.write(f"  mean={result.distances_zed2rb.mean():.6f}\n")
            f.write(f"  std={result.distances_zed2rb.std():.6f}\n\n")
        if result.distances_zed_marker2rb is not None:
            f.write(f"Distance between ZED-marker and RB in VICON Frame:\n")
            f.write(f"  min={result.distances_zed_marker2rb.min():.6f}\n")
            f.write(f"  max={result.distances_zed_marker2rb.max():.6f}\n")
            f.write(f"  mean={result.distances_zed_marker2rb.mean():.6f}\n")
            f.write(f"  std={result.distances_zed_marker2rb.std():.6f}\n")
        if result.angles is not None:
            f.write(f"Rotation angle between ZED and ZED-marker in VICON Frame (degrees):\n")
            f.write(f"  min={np.array(result.angles).min():.6f}\n")
            f.write(f"  max={np.array(result.angles).max():.6f}\n")
            f.write(f"  mean={np.array(result.angles).mean():.6f}\n")
            f.write(f"  std={np.array(result.angles).std():.6f}\n\n")
    save_calibration_distances_csv(result, output_path.replace('.txt', '_distances.csv'))
    save_calibration_results_angles_csv(result, output_path.replace('.txt', '_angles.csv'))

def save_calibration_distances_csv(result, output_path):
    """
    Save per-frame distance metrics to CSV file.
    
    Args:
        result: CalibrationResult instance
        output_path: Path to output CSV file
    """
    distances_dict = {}
    
    # Add ZED to ZED-marker distances
    if result.distances_zed2zed_marker is not None:
        distances_dict['Distance_ZED_to_ZED_marker'] = result.distances_zed2zed_marker
    
    # Add ZED to RB distances
    if result.distances_zed2rb is not None:
        distances_dict['Distance_ZED_to_RB'] = result.distances_zed2rb
    
    # Add ZED-marker to RB distances
    if result.distances_zed_marker2rb is not None:
        distances_dict['Distance_ZED_marker_to_RB'] = result.distances_zed_marker2rb
    
    if distances_dict:
        df_distances = pd.DataFrame(distances_dict)
        df_distances.to_csv(output_path, index_label='Frame')

def save_calibration_results_angles_csv(result, output_path):
    """
    Save per-frame angle metrics to CSV file.
    
    Args:
        result: CalibrationResult instance
        output_path: Path to output CSV file
    """
    if result.angles is not None:
        df_angles = pd.DataFrame({'Rotation_Angle_Degrees': result.angles})
        df_angles.to_csv(output_path, index_label='Frame')  


def generate_ascii_timeline(offsets_ms, time_ranges_local, vicon_window, local_windows):
    """
    Generate detailed ASCII art timeline visualization for logging.
    
    Args:
        frame_stats: dict of {pipeline_num: (n_cropped, n_original)} for frame retention percentages
    """
    output = []
    output.append("\n╔" + "═"*78 + "╗")
    output.append("║" + " "*78 + "║")
    output.append("║" + "TEMPORAL SYNCHRONIZATION TIMELINE (Unified Calibration)".center(78) + "║")
    output.append("║" + " "*78 + "║")
    output.append("╚" + "═"*78 + "╝\n")
    
    vicon_t_min, vicon_t_max = vicon_window
    vicon_duration = vicon_t_max - vicon_t_min
    
    output.append(" "*20 + "VICON REFERENCE FRAME")
    output.append("  ┌" + "─"*67 + "┐")
    output.append(f"  │ [{vicon_t_min:>8.2f} ━━━━━━━━━━━━━━━━━━━━━━━━ {vicon_t_max:>8.2f}] ms" + " "*14 + "│")
    output.append(f"  │  ◀────────────────── {vicon_duration:.2f} ms duration ────────────────►" + " "*8 + "│")
    output.append("  └" + "─"*67 + "┘\n")
    
    pipeline_names = {1: "ZED SDK", 2: "cuVSLAM - Isaac SVO2", 3: "cuVSLAM - Isaac RealSense"}
    
    for pipeline_num in sorted(offsets_ms.keys()):
        lag = offsets_ms[pipeline_num]
        t_min_local, t_max_local = local_windows[pipeline_num]
        
        lag_seconds = abs(lag) / 1000.0
        lag_indicator = f"ZED ahead by {lag_seconds:.2f}s" if lag < 0 else f"VICON ahead by {lag_seconds:.2f}s"
        
        # Calculate frame retention from actual frame counts
        retention_pct = (local_windows[pipeline_num][1] - local_windows[pipeline_num][0]) / (time_ranges_local[pipeline_num][1] - time_ranges_local[pipeline_num][0]) * 100 if pipeline_num in time_ranges_local else 100.0
        
        bar_length = 20
        filled = int(bar_length * retention_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        output.append("════════════════════════════════════════════════════════════════════════════════")
        output.append(f"\nPIPELINE {pipeline_num} ({pipeline_names[pipeline_num]})")
        output.append(f"  Lag: {lag:>9.2f} ms ({lag_indicator})")
        output.append(f"  Coverage: {bar} {retention_pct:>5.1f}%")
        output.append(f"  Local Time:     [{t_min_local:>8.2f} ═════════════════════════════ {t_max_local:>8.2f}] ms")
        output.append(f"  Maps to VICON window ✓\n")
    
    output.append("════════════════════════════════════════════════════════════════════════════════\n")
    output.append("KEY OBSERVATIONS:")
    output.append(f"  ✓ All pipelines converge to same VICON window: [{vicon_t_min:.2f}, {vicon_t_max:.2f}] ms")
    output.append(f"  ✓ Unified calibration ensures consistent reference frame across all pipelines")
    output.append("  ✓ Frame retention varies by pipeline - check coverage bars\n")
    
    return "\n".join(output)

def save_ascii_timeline(offsets_ms, time_ranges_local, vicon_window, local_windows, 
                        output_dir, test_num, verbose=True):
    """Save ASCII timeline to log file."""
    ascii_output = generate_ascii_timeline(offsets_ms, time_ranges_local, vicon_window, 
                                          local_windows)
    
    log_file = (output_dir) / f"unified_timeline_test{test_num}.txt"
    with open(log_file, 'w') as f:
        f.write(ascii_output)
    
    if verbose:
        print(ascii_output)
        print(f"Timeline saved to {log_file}")



__all__ = [
    'CalibrationConfig', 'CalibrationResult', 'process_zed_vicon_calibration',
    'compute_common_vicon_window', 'crop_tracking_data_to_window',
    'generate_ascii_timeline', 'save_ascii_timeline',
    'save_calibration_results', 'save_calibration_distances_csv', 'save_calibration_results_angles_csv'
]




def motion_axis_excitation(poses, weight_by_angle=True, min_angle_deg=0.5):
    """
    Rotation-axis diversity of a pose sequence: the physical measure of whether
    a motion provided multi-axis excitation for hand-eye calibration.

    Needs only the REFERENCE (Vicon) side, and the result is invariant to
    whether you pass ^V T_M or ^M T_V -- inverting every pose flips each axis
    sign (n n^T is unchanged), and changing frame rotates all axes together
    (which leaves the scatter eigenvalues unchanged).  So there is no
    convention to get wrong here.

    poses : (N,4,4) or (N,3,3) absolute poses, one consistent frame.

    Returns
    -------
    axis_scatter_eigs : 3 eigenvalues, descending, summing to 1.
                        [1/3, 1/3, 1/3] = isotropic (ideal excitation)
                        [1, 0, 0]       = all rotation about one axis (degenerate)
    axis_anisotropy   : eigs[0] / eigs[2].  ~1-5 healthy, >1e3 degenerate.
    """
    P = np.asarray(poses, dtype=float)
    R = P[:, :3, :3] if P.shape[-2:] == (4, 4) else P
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (N,4,4) or (N,3,3), got {P.shape}")

    # Relative rotations w.r.t. the first pose.  If all of these share a single
    # axis then so does every pairwise relative rotation, so this is a complete
    # test for the single-axis degeneracy -- no need for consecutive differences.
    rotvec = Rotation.from_matrix(R[0].T @ R).as_rotvec()
    ang = np.linalg.norm(rotvec, axis=1)

    keep = ang > np.radians(min_angle_deg)   # axis is undefined for tiny angles
    if keep.sum() < 3:
        return {"axis_scatter_eigs": np.full(3, np.nan),
                "axis_anisotropy": np.nan, "rotation_span_deg": np.nan,
                "mean_rotation_deg": np.nan, "n_used": int(keep.sum())}

    axes = rotvec[keep] / ang[keep, None]
    w = ang[keep] if weight_by_angle else np.ones(keep.sum())

    S = np.einsum("n,ni,nj->ij", w, axes, axes) / w.sum()
    eigs = np.linalg.eigvalsh(S)[::-1]
    return {
        "axis_scatter_eigs": eigs,
        "axis_anisotropy": float(eigs[0] / max(eigs[-1], 1e-15)),
        "rotation_span_deg": float(np.degrees(ang.max())),
        "mean_rotation_deg": float(np.degrees(ang[keep].mean())),
        "n_used": int(keep.sum()),
    }