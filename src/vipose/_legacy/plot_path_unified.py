#!/usr/bin/env python3
"""
plot_path_unified.py - Unified wrapper for pipeline-wise calibration comparison.

Discovers CSV outputs from all three pipelines in the unified workflow results,
loads corresponding VICON mocap data, and runs calibration+visualization for each pipeline.
Organizes results by pipeline with test-based subdirectories.

usage (from ZED_Project root):
    .venv/bin/python3 plot_path_unified.py --tests 1 --pipelines 3
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from scipy.optimize import fmin
from plot_path_lib import (
    CalibrationConfig, process_zed_vicon_calibration, save_ascii_timeline, save_calibration_results,
    compute_common_vicon_window, crop_tracking_data_to_window
)
from SynchronizationNew import get_df_ZED, load_mocap, calculate_vicon_rigid_body, estimate_temporal_offset_zed_vicon, preprocess_zed_data, synchronize_zed_vicon_trial
from stage3_integration import refine_offsets_stage3

class PipelineConfig:
    """Configuration for discovering and processing pipelines."""
    
    PIPELINE_NAMES = {
        1: "pipeline1_zed_sdk",
        2: "pipeline2_isaac_svo2",
        3: "pipeline3_isaac_realsense"
    }
    
    CSV_PATTERNS = {
        1: "{test}_ZED_SDK_trajectory.csv",
        2: "{test}_cuVSLAM_results.csv",
        3: "{test}_realsense_cuVSLAM_results.csv"
    }
    
    def __init__(self, results_base_dir, data_base_dir):
        """
        Initialize pipeline configuration.
        
        Args:
            results_base_dir: Base directory for pipeline results (e.g., Zohar_Experiment/23_10_25/results/)
            data_base_dir: Base directory for mocap data (e.g., Zohar_Experiment/23_10_25/data/)
        """
        self.results_base_dir = Path(results_base_dir)
        self.data_base_dir = Path(data_base_dir)
    
    def get_pipeline_csv(self, pipeline_num, test_num):
        """
        Get path to pipeline CSV for given test number.
        
        Args:
            pipeline_num: Pipeline number (1, 2, or 3)
            test_num: Test number (1-5)
            
        Returns:
            Path to CSV file if it exists, None otherwise
        """
        pipeline_dir = self.results_base_dir / self.PIPELINE_NAMES[pipeline_num] / f"test{test_num}"
        csv_pattern = self.CSV_PATTERNS[pipeline_num]
        csv_filename = csv_pattern.format(test=f"test{test_num}")
        csv_path = pipeline_dir / csv_filename
        
        if csv_path.exists():
            print(f"Found CSV for pipeline {pipeline_num}, test {test_num}: {csv_path}")
            return csv_path
        else:
            return None
    
    def get_mocap_csv(self, test_num):
        """
        Get path to VICON mocap CSV for given test number.
        
        Args:
            test_num: Test number (1-5)
            
        Returns:
            Path to mocap CSV if it exists, None otherwise
        """
        mocap_path = self.data_base_dir / f"test{test_num}" / f"test{test_num}_mocap.csv"
        
        if mocap_path.exists():
            return mocap_path
        else:
            return None
    
    def get_output_dir(self, pipeline_num, test_num):
        """
        Get output directory for calibration results.
        
        Args:
            pipeline_num: Pipeline number (1, 2, or 3)
            test_num: Test number (1-5)
            
        Returns:
            Path to output directory (creates if doesn't exist)
        """
        output_dir = self.results_base_dir / "comparison" / self.PIPELINE_NAMES[pipeline_num] / f"test{test_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_comparison_dir(self):
        """Get base directory for comparison results."""
        comparison_dir = self.results_base_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        return comparison_dir
    


def process_test_unified_calibration(pipeline_config, test_num, pipelines=[1, 2, 3], verbose=True):
    """
    Process all pipelines for a single test with unified VICON time window.
    
    Ensures all pipelines are calibrated against the same VICON time window by:
    1. Loading all pipeline tracking data and shared VICON mocap
    2. Estimating temporal offset for each pipeline with three-stage refinement
    3. Computing common time window intersection across all pipelines
    4. Cropping all pipeline data to common window
    4a. Saving cropped tracking data to CSV files
    4b. Refining temporal offsets on cropped data with three-stage refinement
    5. Running calibration on cropped data
    
    Args:
        pipeline_config: PipelineConfig instance
        test_num: Test number (1-5)
        pipelines: List of pipeline numbers (default [1, 2, 3])
        verbose: Print progress messages
        
    Returns:
        List of result metadata dictionaries (one per pipeline, same format as independent mode)
    """
    all_results = []
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"UNIFIED CALIBRATION MODE - Test {test_num}")
        print(f"{'#'*70}")
    
    # ===== STEP 1: Load all tracking data and mocap =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 1: Loading all tracking data and VICON mocap")
        print(f"{'='*70}")
    
    tracking_data = {}  # {pipeline_num: df_ZED}
    csv_paths = {}      # {pipeline_num: path}
    
    # Load mocap (shared)
    mocap_path = pipeline_config.get_mocap_csv(test_num)
    if mocap_path is None:
        msg = f"VICON mocap CSV not found for test {test_num}"
        if verbose:
            print(f"Error: {msg}")
        # Return failed results for all pipelines
        for pipeline_num in pipelines:
            all_results.append({
                "pipeline": pipeline_num,
                "test": test_num,
                "status": "failed",
                "error": msg,
                "mode": "unified"
            })
        return all_results
    
    if verbose:
        print(f"Loading VICON mocap from: {mocap_path}")
    
    try:
        df_mocap = load_mocap(str(mocap_path))
        df_rb, uncropped_rb_residuals = calculate_vicon_rigid_body(df_mocap, print_stats=verbose)
        # print rb fitting stats here if verbose, for the uncropped mocap data, including the number of frames, mean residuals, std and max residuals.
        if verbose:
            print(f"VICON Rigid Body Fitting Stats (Uncropped):")
            print(f"  Frames: {len(uncropped_rb_residuals)}")
            print(f"  Mean Residual: {uncropped_rb_residuals.mean():.4f}")
            print(f"  Std Residual: {uncropped_rb_residuals.std():.4f}")
            print(f"  Max Residual: {uncropped_rb_residuals.max():.4f}")

        if verbose:
            print(f"Loaded VICON: {len(df_mocap.Frame)} frames")
    except Exception as e:
        msg = f"Failed to load VICON mocap: {e}"
        if verbose:
            print(f"Error: {msg}")
        for pipeline_num in pipelines:
            all_results.append({
                "pipeline": pipeline_num,
                "test": test_num,
                "status": "failed",
                "error": msg,
                "mode": "unified"
            })
        return all_results
    
    # Load tracking data for all pipelines
    for pipeline_num in pipelines:
        csv_path = pipeline_config.get_pipeline_csv(pipeline_num, test_num)
        
        if csv_path is None:
            if verbose:
                print(f"Warning: Pipeline {pipeline_num} CSV not found for test {test_num}")
            continue
        
        try:
            df_ZED = get_df_ZED(str(csv_path))
            df_ZED = preprocess_zed_data(
                    df_ZED,
                    ZED_replace_stagnant_with_interpolation=False,
                    ZED_remove_stagnant_poses=False,
                    ZED_smooth_data=False,
                    ZED_semantic_rebase=False,
                    rebase_at_first_dict={"Spatial_Memory": ["MAP_UPDATE", "cuVSLAM"]},
                    # rebase_at_first_dict={"Spatial_Memory": ["LOOP_CLOSED", "cuVSLAM"]}, $<<< Make sure this matches the rebase condition used in plot_path_lib.py
                    velocity_threshold=1e-4,
                    window_size=None
                )
            tracking_data[pipeline_num] = df_ZED
            csv_paths[pipeline_num] = csv_path
            if verbose:
                print(f"Pipeline {pipeline_num}: {len(df_ZED)} frames, "
                      f"time range [{df_ZED['Time_ms'].min():.2f}, {df_ZED['Time_ms'].max():.2f}] ms")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load pipeline {pipeline_num} CSV: {e}")
            continue
    
    if not tracking_data:
        msg = "No valid tracking data found for any pipeline"
        if verbose:
            print(f"Error: {msg}")
        for pipeline_num in pipelines:
            all_results.append({
                "pipeline": pipeline_num,
                "test": test_num,
                "status": "failed",
                "error": msg,
                "mode": "unified"
            })
        return all_results
    
    # ===== STEP 2: Estimate temporal offsets with three-stage refinement =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 2: Estimating temporal offsets (three-stage refinement)")
        print(f"{'='*70}")
    
    offsets_ms = {}  # Final optimized offset
    offset_stages = {}  # Store all three refinement stages per pipeline
    pipelines_to_skip = []  # Collect pipelines that fail
    
    for pipeline_num, df_ZED in tracking_data.items():
        try:
            # Stage 1: Cross-correlation on full data
            offset_stage1 = estimate_temporal_offset_zed_vicon(df_ZED, df_rb, discard_fraction=0.0)
            
            # Stage 2: fmin optimization on full data
            def objective_full(offset_candidate):
                rms_error = synchronize_zed_vicon_trial(df_ZED, df_rb, offset_candidate, discard_fraction=0.0)
                return rms_error
            
            offset_stage2 = fmin(objective_full, offset_stage1, maxiter=20, xtol=1e-5, disp=False)[0]
            delta_stage2 = offset_stage2 - offset_stage1
            
            offsets_ms[pipeline_num] = offset_stage2
            offset_stages[pipeline_num] = {
                'stage1_crosscorr_ms': float(offset_stage1),
                'stage2_optimized_ms': float(offset_stage2),
                'stage2_delta_ms': float(delta_stage2)
            }
            
            if verbose:
                print(f"Pipeline {pipeline_num}:")
                print(f"  Stage 1 (cross-correlation): {offset_stage1:.2f} ms")
                print(f"  Stage 2 (optimized full):    {offset_stage2:.2f} ms (Δ {delta_stage2:+.2f} ms)")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to estimate offset for pipeline {pipeline_num}: {e}")
            # Mark this pipeline to skip
            pipelines_to_skip.append(pipeline_num)
    
    # Remove pipelines that failed
    for pipeline_num in pipelines_to_skip:
        if pipeline_num in tracking_data:
            del tracking_data[pipeline_num]
    
    if not offsets_ms:
        msg = "Failed to estimate offsets for all pipelines"
        if verbose:
            print(f"Error: {msg}")
        for pipeline_num in pipelines:
            all_results.append({
                "pipeline": pipeline_num,
                "test": test_num,
                "status": "failed",
                "error": msg,
                "mode": "unified"
            })
        return all_results
    
    # ===== STEP 3: Compute common time window =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 3: Computing common VICON time window")
        print(f"{'='*70}")
    
    time_ranges_local = {
        pipeline_num: (df_ZED['Time_ms'].min(), df_ZED['Time_ms'].max())
        for pipeline_num, df_ZED in tracking_data.items()
    }

   
    try:
        # window_result = compute_common_vicon_window(offsets_ms, time_ranges_local)
        offsets_ms, window_result, stage3_info = refine_offsets_stage3(
                       offsets_ms=offsets_ms,
                       time_ranges_local=time_ranges_local,
                       csv_paths={pn: str(pipeline_config.get_pipeline_csv(pn, test_num))
                                  for pn in tracking_data},
                       mocap_path=str(pipeline_config.get_mocap_csv(test_num)),
                       compute_window=compute_common_vicon_window,
                       config=CalibrationConfig(),
                       verbose=verbose,
                   )
        
        if not window_result['valid']:
            raise ValueError("Common time window is empty")

        for pipeline_num, info in stage3_info.items():
            offset_stages.setdefault(pipeline_num, {}).update(info)

        vicon_window = window_result['vicon_window']
        local_windows = window_result['local_windows']
        stats = window_result['stats']
        
        if verbose:
            print(f"Common VICON window: [{vicon_window[0]:.2f}, {vicon_window[1]:.2f}] ms "
                  f"(duration: {stats['vicon_duration_ms']:.2f} ms)")
            print("Back-projected local windows:")
            for pipeline_num, (t_min, t_max) in local_windows.items():
                duration = t_max - t_min
                print(f"  Pipeline {pipeline_num}: [{t_min:.2f}, {t_max:.2f}] ms (duration: {duration:.2f} ms)")
    
    except Exception as e:
        msg = f"Failed to compute common time window: {e}"
        if verbose:
            print(f"Error: {msg}")
        for pipeline_num in tracking_data.keys():
            all_results.append({
                "pipeline": pipeline_num,
                "test": test_num,
                "status": "failed",
                "error": msg,
                "mode": "unified"
            })
        return all_results
    
    save_ascii_timeline(offsets_ms, time_ranges_local, vicon_window, local_windows, pipeline_config.get_comparison_dir(), test_num, verbose=verbose)
    
    # ===== STEP 3b: Calculate cropped VICON rigid body residuals =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 3b: Computing cropped VICON rigid body stats")
        print(f"{'='*70}")
    
    try:
        import numpy as np
        
        # Crop mocap data to common VICON time window
        # df_mocap is a MocapData object with Frame, Marker, Time_ms attributes
        time_ms = np.array(df_mocap.Time_ms)
        
        # Find frame indices within vicon window
        valid_indices = np.where(
            (time_ms >= vicon_window[0]) & 
            (time_ms <= vicon_window[1])
        )[0]
        
        if len(valid_indices) > 0:
            # Get start and end indices
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1] + 1
            
            # Create a new MocapData object with cropped data
            from SynchronizationNew import MocapData, Marker
            
            cropped_frames = df_mocap.Frame[start_idx:end_idx]
            cropped_time_ms = df_mocap.Time_ms[start_idx:end_idx]
            
            # Crop marker data
            cropped_markers = []
            for marker in df_mocap.Marker:
                cropped_marker = Marker(
                    marker.name,
                    marker.X[start_idx:end_idx],
                    marker.Y[start_idx:end_idx],
                    marker.Z[start_idx:end_idx]
                )
                cropped_markers.append(cropped_marker)
            
            # Create cropped MocapData
            df_mocap_cropped = MocapData(
                cropped_frames,
                df_mocap.Frame_Rate,
                cropped_markers,
                Time_ms=cropped_time_ms
            )
            
            # Recalculate rigid body for cropped data
            df_rb_cropped, cropped_rb_residuals = calculate_vicon_rigid_body(df_mocap_cropped, print_stats=False)
            
            if verbose:
                print(f"VICON Rigid Body Fitting Stats (Cropped to common window):")
                print(f"  Frames: {len(cropped_rb_residuals)}")
                print(f"  Mean Residual: {cropped_rb_residuals.mean():.4f}")
                print(f"  Std Residual: {cropped_rb_residuals.std():.4f}")
                print(f"  Max Residual: {cropped_rb_residuals.max():.4f}")
            
            # Save cropped rigid body stats to JSON file
            cropped_rb_stats = {
                "test": test_num,
                "vicon_window_ms": {
                    "start": float(vicon_window[0]),
                    "end": float(vicon_window[1]),
                    "duration": float(vicon_window[1] - vicon_window[0])
                },
                "rigid_body_fitting_stats": {
                    "frames": int(len(cropped_rb_residuals)),
                    "mean_residual": float(cropped_rb_residuals.mean()),
                    "std_residual": float(cropped_rb_residuals.std()),
                    "max_residual": float(cropped_rb_residuals.max())
                }
            }
            
            comparison_dir = pipeline_config.get_comparison_dir()
            stats_file = comparison_dir / f"test{test_num}_cropped_rb_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(cropped_rb_stats, f, indent=2)
            
            if verbose:
                print(f"Saved cropped RB stats to {stats_file}")
        else:
            if verbose:
                print(f"Warning: No mocap data in VICON window [{vicon_window[0]:.2f}, {vicon_window[1]:.2f}] ms")
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to compute cropped VICON rigid body stats: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== STEP 4: Crop tracking data =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 4: Cropping tracking data to common window")
        print(f"{'='*70}")
    
    cropped_data = {}
    for pipeline_num, (t_min, t_max) in local_windows.items():
        if pipeline_num not in tracking_data:
            continue
        df_ZED = tracking_data[pipeline_num]
        try:
            df_cropped = crop_tracking_data_to_window(df_ZED, t_min, t_max, verbose=verbose)
            # # Add time column (ms) relative to first timestamp
            # if 'Timestamp' in df_cropped.columns:
            #     df_cropped['time_ms'] = df_cropped['timestamp'] - df_cropped['timestamp'].iloc[0]
            cropped_data[pipeline_num] = df_cropped
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to crop pipeline {pipeline_num}: {e}")
            # Skip this pipeline
            continue
    
    if not cropped_data:
        msg = "Failed to crop data for all pipelines"
        if verbose:
            print(f"Error: {msg}")
        for pipeline_num in tracking_data.keys():
            all_results.append({
                "pipeline": pipeline_num,
                "test": test_num,
                "status": "failed",
                "error": msg,
                "mode": "unified"
            })
        return all_results
    
    # ===== STEP 4a: Save cropped tracking data to CSV =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 4a: Saving cropped tracking data to CSV files")
        print(f"{'='*70}")
    
    for pipeline_num, df_cropped in cropped_data.items():
        try:
            output_dir = pipeline_config.get_output_dir(pipeline_num, test_num)
            
            # Construct output filename with _unified suffix
            csv_patterns = {
                1: "test{test}_ZED_SDK_trajectory_unified.csv",
                2: "test{test}_cuVSLAM_results_unified.csv",
                3: "test{test}_realsense_cuVSLAM_results_unified.csv"
            }
            
            filename = csv_patterns[pipeline_num].format(test=test_num)
            csv_path = output_dir / filename
            
            # Save cropped data to CSV
            df_cropped.to_csv(csv_path, index=False)
            
            if verbose:
                print(f"Pipeline {pipeline_num}: Saved {len(df_cropped)} cropped frames to {filename}")
        
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save cropped CSV for pipeline {pipeline_num}: {e}")
    
    # ===== STEP 5: Run calibration on cropped data =====
    if verbose:
        print(f"\n{'='*70}")
        print("Step 5: Running calibration on cropped data (same VICON window)")
        print(f"{'='*70}")
    
    for pipeline_num in cropped_data.keys():
        if verbose:
            print(f"\n--- Calibrating Pipeline {pipeline_num} ---")
        
        result_meta = {
            "pipeline": pipeline_num,
            "test": test_num,
            "status": "pending",
            "error": None,
            "csv_path": str(csv_paths[pipeline_num]),
            "mocap_path": str(mocap_path),
            "output_dir": None,
            "calibration_result_file": None,
            "timestamp": datetime.now().isoformat(),
            "mode": "unified",
            "temporal_offset_ms": float(offsets_ms[pipeline_num]),
            "offset_refinement_stages": offset_stages.get(pipeline_num, {}),
            "vicon_window": vicon_window,
            "local_window": local_windows[pipeline_num]
        }
        
        output_dir = pipeline_config.get_output_dir(pipeline_num, test_num)
        result_meta["output_dir"] = str(output_dir)
        
        try:
            config = CalibrationConfig()
            config.interactive = True
            
            # Call calibration with pre-cropped window
            # Note: We pass cropped data, but need to re-load from CSV and then apply cropping
            # Actually, we should create a wrapper or write the cropped data temporarily
            # For now, let's use the cropping directly in calibration
            pre_crop_window = {
                't_min': local_windows[pipeline_num][0],
                't_max': local_windows[pipeline_num][1]
            }
            
            calib_result = process_zed_vicon_calibration(
                str(csv_paths[pipeline_num]),
                mocap_csv_path=str(mocap_path),
                config=config,
                verbose=verbose,
                pre_cropped_time_range=pre_crop_window,
                pre_computed_lag=offsets_ms[pipeline_num]
            )
            
            # Save calibration results
            if calib_result.T_zed2rb is not None:
                result_file = output_dir / f"pipeline{pipeline_num}_test{test_num}_calibration_results_unified.txt"
                save_calibration_results(calib_result, str(result_file))
                result_meta["calibration_result_file"] = str(result_file)
                result_meta["status"] = "success"
                
                # Save metadata as JSON
                metadata_file = output_dir / f"pipeline{pipeline_num}_test{test_num}_metadata_unified.json"
                with open(metadata_file, 'w') as f:
                    json.dump(result_meta, f, indent=2)
                
                if verbose:
                    print(f"Results saved to {result_file}")
            else:
                msg = "Calibration failed: T_zed2rb is None"
                result_meta["status"] = "failed"
                result_meta["error"] = msg
                if verbose:
                    print(f"Error: {msg}")
        
        except Exception as e:
            result_meta["status"] = "failed"
            result_meta["error"] = str(e)
            if verbose:
                print(f"Exception during calibration: {e}")
                import traceback
                traceback.print_exc()
        
        all_results.append(result_meta)
    
    return all_results


def process_pipeline_for_test(pipeline_config, pipeline_num, test_num, verbose=True):
    """
    Process a single pipeline for a given test.
    
    Args:
        pipeline_config: PipelineConfig instance
        pipeline_num: Pipeline number (1, 2, or 3)
        test_num: Test number (1-5)
        verbose: Print progress messages
        
    Returns:
        Dictionary with results metadata and status
    """
    result_meta = {
        "pipeline": pipeline_num,
        "test": test_num,
        "status": "pending",
        "error": None,
        "csv_path": None,
        "mocap_path": None,
        "output_dir": None,
        "calibration_result_file": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Get paths
    csv_path = pipeline_config.get_pipeline_csv(pipeline_num, test_num)
    mocap_path = pipeline_config.get_mocap_csv(test_num)
    output_dir = pipeline_config.get_output_dir(pipeline_num, test_num)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing Pipeline {pipeline_num}, Test {test_num}")
        print(f"{'='*70}")
    
    # Validate inputs
    if csv_path is None:
        msg = f"Pipeline {pipeline_num} CSV not found for test {test_num}"
        if verbose:
            print(f"Error: {msg}")
        result_meta["status"] = "failed"
        result_meta["error"] = msg
        return result_meta
    
    if mocap_path is None:
        msg = f"VICON mocap CSV not found for test {test_num}"
        if verbose:
            print(f"Error: {msg}")
        result_meta["status"] = "failed"
        result_meta["error"] = msg
        return result_meta
    
    result_meta["csv_path"] = str(csv_path)
    result_meta["mocap_path"] = str(mocap_path)
    result_meta["output_dir"] = str(output_dir)
    
    if verbose:
        print(f"ZED CSV: {csv_path}")
        print(f"Mocap CSV: {mocap_path}")
        print(f"Output dir: {output_dir}")
    
    # Run calibration
    try:
        config = CalibrationConfig()
        config.interactive = True  # Keep interactive mode as requested
        
        calib_result = process_zed_vicon_calibration(
            str(csv_path),
            mocap_csv_path=str(mocap_path),
            config=config,
            verbose=verbose
        )
        
        # Save calibration results
        if calib_result.T_zed2rb is not None:
            result_file = output_dir / f"pipeline{pipeline_num}_test{test_num}_calibration_results.txt"
            save_calibration_results(calib_result, str(result_file))
            result_meta["calibration_result_file"] = str(result_file)
            result_meta["status"] = "success"
            
            # Save metadata as JSON
            metadata_file = output_dir / f"pipeline{pipeline_num}_test{test_num}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(result_meta, f, indent=2)
            
            if verbose:
                print("")
                print(f"Results saved to {result_file}")
        else:
            msg = "Calibration failed: T_zed2rb is None"
            result_meta["status"] = "failed"
            result_meta["error"] = msg
            if verbose:
                print(f"Error: {msg}")
    
    except Exception as e:
        result_meta["status"] = "failed"
        result_meta["error"] = str(e)
        if verbose:
            print(f"Exception during calibration: {e}")
            import traceback
            traceback.print_exc()
    
    return result_meta


def process_test_all_pipelines(pipeline_config, test_num, pipelines=[1, 2, 3], verbose=True, unified=False):
    """
    Process all specified pipelines for a given test.
    
    Args:
        pipeline_config: PipelineConfig instance
        test_num: Test number (1-5)
        pipelines: List of pipeline numbers to process (default [1, 2, 3])
        verbose: Print progress messages
        unified: bool
            - If True: Use unified mode (all pipelines calibrated against same VICON window)
            - If False: Use independent mode (each pipeline independently calibrated)
        
    Returns:
        List of result metadata dictionaries
    """
    if unified:
        # Unified mode: all pipelines share common VICON window
        return process_test_unified_calibration(pipeline_config, test_num, pipelines=pipelines, verbose=verbose)
    else:
        # Independent mode: each pipeline independently calibrated (legacy behavior)
        results = []
        for pipeline_num in pipelines:
            result = process_pipeline_for_test(pipeline_config, pipeline_num, test_num, verbose=verbose)
            results.append(result)
        return results


def create_comparison_summary(results, output_dir):
    """
    Create summary of all calibration results.
    
    Args:
        results: List of result metadata dictionaries
        output_dir: Directory to save summary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_pipelines": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }
    
    summary_file = Path(output_dir) / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    """Main entry point for unified pipeline calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run hand-eye calibration across all pipelines for specified tests"
    )
    parser.add_argument(
        "--results-dir",
        default="/home/zohar/Documents/ZED/ZED_Project/Zohar_Experiment/23_10_25/results",
        help="Base directory for pipeline results"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zohar/Documents/ZED/ZED_Project/Zohar_Experiment/23_10_25/data",
        help="Base directory for mocap data"
    )
    parser.add_argument(
        "--tests",
        default="1",
        help="Comma-separated list of test numbers (e.g., '1' or '1,2,3')"
    )
    parser.add_argument(
        "--pipelines",
        default="1,2,3",
        help="Comma-separated list of pipeline numbers (e.g., '1,2,3')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (enabled by default)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        default=False,
        help="Use unified calibration mode (all pipelines calibrated against same VICON window). "
             "Default is independent mode (each pipeline calibrated independently)"
    )
    parser.add_argument(
        "--noplot",
        action="store_true",
        default=False,
        help="Disable interactive plot display. Plots are still generated for data extraction, "
             "but automatically closed to prevent blocking"
    )
    
    args = parser.parse_args()
    
    # Set non-interactive matplotlib backend if --noplot is enabled
    if args.noplot:
        import matplotlib
        matplotlib.use('Agg')
    
    # Parse test and pipeline numbers
    test_nums = [int(t.strip()) for t in args.tests.split(",")]
    pipeline_nums = [int(p.strip()) for p in args.pipelines.split(",")]
    
    # Handle verbose flag (quiet overrides verbose)
    verbose = args.verbose and not args.quiet
    
    if verbose:
        mode_str = "UNIFIED" if args.unified else "INDEPENDENT"
        print(f"Calibration mode: {mode_str}")
        print(f"Results directory: {args.results_dir}")
        print(f"Data directory: {args.data_dir}")
        print(f"Tests: {test_nums}")
        print(f"Pipelines: {pipeline_nums}")
    
    # Initialize pipeline configuration
    pipeline_config = PipelineConfig(args.results_dir, args.data_dir)
    
    # Process tests
    all_results = []
    for test_num in test_nums:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"Processing Test {test_num}")
            print(f"{'#'*70}")
        
        test_results = process_test_all_pipelines(
            pipeline_config,
            test_num,
            pipelines=pipeline_nums,
            verbose=verbose,
            unified=args.unified
        )
        all_results.extend(test_results)
        
        # Auto-close plots after each test if noplot mode is enabled
        if args.noplot:
            import matplotlib.pyplot as plt
            plt.close('all')
    
    # Create summary
    comparison_dir = Path(args.results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary = create_comparison_summary(all_results, comparison_dir)


    
    if verbose:
        print(f"\n{'='*70}")
        print("Summary:")
        print(f"  Total pipelines processed: {summary['total_pipelines']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"{'='*70}\n")
    
    # Handle plot display based on noplot flag
    import matplotlib.pyplot as plt
    
    if args.noplot:
        # Close all remaining plots
        plt.close('all')
        if verbose:
            print("\n" + "="*70)
            print("Plot generation complete (--noplot: plots auto-closed)")
            print("="*70 + "\n")
    else:
        # Show all plots and keep them open
        print("\n" + "="*70)
        print("PLOTS ARE DISPLAYED - Keep terminal open")
        print("Press Enter to close plots and exit...")
        print("="*70 + "\n")
        
        # Turn off interactive mode and show plots blocking
        try:
            plt.ioff()
            plt.show(block=True)
        except Exception as e:
            print(f"Plot display error (harmless): {e}")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
