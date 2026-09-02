from os import error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.widgets import Slider
from scipy.interpolate import CubicSpline
class Marker:
    def __init__(self, name, x, y, z):
        self.name = name
        self.X = x
        self.Y = y
        self.Z = z


class MocapData:
    def __init__(self, frames, frame_rate, markers, Time_ms=None):
        # if frames is None:
        #     frames = [] # Initialize with an empty list if no frames are provided
        self.Frame = frames
        # if frame_rate is None:
        #     frame_rate = []
        self.Marker = markers  # List of Marker objects
        if Time_ms is not None:
            self.Time_ms = Time_ms
        else:
            self.Time = (frames - frames[0]) / frame_rate
            self.Time_ms = self.Time * 1000

        # self.Time = (frames-frames[0]) / frame_rate  # Assuming frames are indexed from 0
        # self.Time_ms = self.Time * 1000  # Convert to milliseconds
        self.Frame_Rate = frame_rate

        # Initialize additional attributes for rigid body calculations
        self.Translation_X = None
        self.Translation_Y = None
        self.Translation_Z = None
        self.Rotation_X = None
        self.Rotation_Y = None
        self.Rotation_Z = None

    def fit_rigid_body(self):
        """
        Fits a rigid‐body transformation from all visible markers for each frame,
        using the SVD (Kabsch) algorithm. Handles marker occlusion by using only
        valid (non-NaN) markers while maintaining consistent translation estimates.
        Returns an object with:
          .angles()        → (N×3) array of rotation vectors (absolute orientation)
          .translations()  → (N×3) array of translation vectors (absolute position)
          .pos_res()       → (N,) array of RMS residuals
        """
        N = len(self.Marker) 
        min_markers = 3  # minimum markers needed for rigid body estimation
        
        # need at least min_markers markers
        if N < min_markers:
            raise ValueError(f"At least {min_markers} markers required to fit rigid body, got {N} after excluding last 2")

        # build reference marker set from first frame (assume first frame has all markers)
        ref = np.stack([[m.X[0], m.Y[0], m.Z[0]] for m in self.Marker[:N]])
        ref_centroid = ref.mean(axis=0)
        ref_centered = ref - ref_centroid

        angles = []
        translations = []
        residuals = []

        for i in range(len(self.Frame)):
            # Get current marker positions and identify valid (non-NaN) markers
            pts_full = np.stack([[m.X[i], m.Y[i], m.Z[i]] for m in self.Marker[:N]])
            
            # Find valid markers (non-NaN in all coordinates)
            valid_mask = ~np.isnan(pts_full).any(axis=1)
            
            if valid_mask.sum() < min_markers:
                # Not enough valid markers - assign NaN values
                print(f"Warning: Only {valid_mask.sum()} valid markers in frame {i}, assigning NaN")
                angles.append(np.array([np.nan, np.nan, np.nan]))
                translations.append(np.array([np.nan, np.nan, np.nan]))
                residuals.append(np.nan)
                continue
        
            # Extract valid markers only
            pts_valid = pts_full[valid_mask]
            ref_valid = ref[valid_mask]
            
            # Center the valid markers using their own centroids
            pts_centroid = pts_valid.mean(axis=0)
            ref_valid_centroid = ref_valid.mean(axis=0)
            
            pts_centered = pts_valid - pts_centroid
            ref_centered_valid = ref_valid - ref_valid_centroid

            # SVD/Kabsch on valid markers only
            H = pts_centered.T @ ref_centered_valid
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt

            # rotation vector (absolute orientation)
            rotvec = Rotation.from_matrix(R).as_rotvec()
            angles.append(rotvec)
            
            # Translation: absolute position of rigid body center in VICON coordinates
            # Correct for centroid shift due to missing markers
            ref_full_centroid = ref.mean(axis=0)  # Original centroid with all markers
            centroid_correction = ref_full_centroid - ref_valid_centroid  # Offset due to missing markers
            transformed_correction = R @ centroid_correction  # Apply current rotation to the correction
            translation = pts_centroid + transformed_correction  # Corrected position of full rigid body center
            translations.append(translation)

            # compute residual (RMS) using only valid markers
            ref_trans = (R @ ref_centered_valid.T).T + pts_centroid
            rms = np.sqrt(np.mean(np.sum((pts_valid - ref_trans)**2, axis=1)))
            residuals.append(rms)

        class RigidBodyFit:
            def __init__(self, angles, residuals, translations):
                self._angles = angles
                self._res = residuals
                self._t = translations

            def angles(self):
                return self._angles

            def translations(self):
                return self._t

            def pos_res(self):
                return self._res
            
            def T(self):
                """
                Returns a list of transformation matrices for each frame.
                Each matrix is a 4x4 transformation matrix combining rotation and translation.
                """
                T_matrices = []
                for i in range(len(self._angles)):
                    R = Rotation.from_rotvec(self._angles[i]).as_matrix()
                    t = self._t[i]
                    T_matrix = np.eye(4)
                    T_matrix[:3, :3] = R
                    T_matrix[:3, 3] = t
                    T_matrices.append(T_matrix)
                return T_matrices

        return RigidBodyFit(
            np.array(angles),
            np.array(residuals),
            np.array(translations)
        )    
    def df_rb2vc(self):
        """
        Calculate the transformation from the local frame (RB) to the global frame (VC)
        for each time step (frame) based on marker positions. Handles marker occlusion
        by using valid markers and interpolating missing ones when possible.

        Returns:
            pd.DataFrame: DataFrame with fields Translation_X, Translation_Y, Translation_Z,
                        Rotation_X, Rotation_Y, Rotation_Z, Frame, and Time_ms.
        """
        n_frames = len(self.Frame)
        markers = self.Marker

        if len(markers) < 5:
            raise ValueError("At least 5 markers are required to define the RB frame.")

        translations = []
        rotations = []

        for i in range(n_frames):
            # Get marker positions at the current frame
            marker_positions = []
            for j in range(5):  # First 5 markers
                pos = np.array([markers[j].X[i], markers[j].Y[i], markers[j].Z[i]])
                marker_positions.append(pos)
            
            # Check for NaN values
            valid_markers = []
            valid_indices = []
            for j, pos in enumerate(marker_positions):
                if not np.isnan(pos).any():
                    valid_markers.append(pos)
                    valid_indices.append(j)
            
            if len(valid_markers) < 3:
                print(f"Warning: Only {len(valid_markers)} valid markers in frame {i} for df_rb2vc")
                # Use previous frame's transformation or identity
                if i > 0:
                    translations.append(translations[-1])
                    rotations.append(rotations[-1])
                else:
                    translations.append(np.array([0.0, 0.0, 0.0]))
                    rotations.append(np.array([0.0, 0.0, 0.0]))
                continue
            
            # If we have fewer than 5 markers, we need to adapt the axis definition
            if len(valid_markers) == 5:
                # All markers available - use original method
                p0, p1, p2, p3, p4 = marker_positions
                ez = (p1 - p4) / np.linalg.norm(p1 - p4)
                ey = np.cross(p2 - p4, ez)
                ey /= np.linalg.norm(ey)
                ex = np.cross(ey, ez)
                t_rb_to_vc = p4
                
            else:
                # Fewer markers available - use robust axis definition
                valid_pos = np.array(valid_markers)
                
                # Use first two markers for one axis if available
                if len(valid_markers) >= 4:
                    # Try to use markers that correspond to original p1, p4 if available
                    if 1 in valid_indices and 4 in valid_indices:
                        idx1, idx4 = valid_indices.index(1), valid_indices.index(4)
                        ez = (valid_pos[idx1] - valid_pos[idx4])
                        ez /= np.linalg.norm(ez)
                    else:
                        # Use first two available markers
                        ez = (valid_pos[1] - valid_pos[0])
                        ez /= np.linalg.norm(ez)
                    
                    # Find another marker to define the plane
                    if 2 in valid_indices and 4 in valid_indices:
                        idx2, idx4 = valid_indices.index(2), valid_indices.index(4)
                        ey = np.cross(valid_pos[idx2] - valid_pos[idx4], ez)
                    else:
                        # Use third available marker
                        ey = np.cross(valid_pos[2] - valid_pos[0], ez)
                    
                    ey /= np.linalg.norm(ey)
                    ex = np.cross(ey, ez)
                    
                    # Use centroid or specific marker as origin
                    if 4 in valid_indices:
                        t_rb_to_vc = valid_pos[valid_indices.index(4)]
                    else:
                        t_rb_to_vc = np.mean(valid_pos, axis=0)  # Use centroid
                        
                else:
                    # Very few markers - use PCA or simple approach
                    centroid = np.mean(valid_pos, axis=0)
                    if len(valid_markers) >= 3:
                        # Use first two markers for primary axis
                        ez = (valid_pos[1] - valid_pos[0])
                        ez /= np.linalg.norm(ez)
                        # Use third marker for plane definition
                        temp_vec = valid_pos[2] - valid_pos[0]
                        ey = np.cross(temp_vec, ez)
                        ey /= np.linalg.norm(ey)
                        ex = np.cross(ey, ez)
                        t_rb_to_vc = centroid
                    else:
                        # Fallback to identity or previous frame
                        if i > 0:
                            translations.append(translations[-1])
                            rotations.append(rotations[-1])
                            continue
                        else:
                            ex = np.array([1, 0, 0])
                            ey = np.array([0, 1, 0])
                            ez = np.array([0, 0, 1])
                            t_rb_to_vc = centroid

            # Construct rotation matrix (RB to VC)
            R_rb_to_vc = np.column_stack((ex, ey, ez))
            
            # Ensure it's a proper rotation matrix
            U, _, Vt = np.linalg.svd(R_rb_to_vc)
            R_rb_to_vc = U @ Vt
            if np.linalg.det(R_rb_to_vc) < 0:
                U[:, -1] *= -1
                R_rb_to_vc = U @ Vt

            # Convert rotation matrix to rotation vector
            rotvec = Rotation.from_matrix(R_rb_to_vc).as_rotvec()

            translations.append(t_rb_to_vc)
            rotations.append(rotvec)

        # Create DataFrame with the transformations
        data = {
            "Translation_X": [t[0] for t in translations],
            "Translation_Y": [t[1] for t in translations],
            "Translation_Z": [t[2] for t in translations],
            "Rotation_X": [r[0] for r in rotations],
            "Rotation_Y": [r[1] for r in rotations],
            "Rotation_Z": [r[2] for r in rotations],
            "Frame": self.Frame,
            "Time_ms": self.Time_ms
            
        }
        return pd.DataFrame(data)
    def truncate_before_frame(self, frame_index):
        """
        Truncates the MocapData to only include data from the specified frame index onward.
        
        Args:
            frame_index (int): The frame index to start from.
        """
        if frame_index < 0 or frame_index >= len(self.Frame):
            raise ValueError("frame_index is out of bounds.")
        
        # Truncate Frame
        self.Frame = self.Frame[frame_index:]
        
        # Truncate Time_ms
        self.Time_ms = self.Time_ms[frame_index:]
        
        # Truncate Marker data
        for marker in self.Marker:
            marker.X = marker.X[frame_index:]
            marker.Y = marker.Y[frame_index:]
            marker.Z = marker.Z[frame_index:]

### ZED Data Loading and Processing ###
def get_df_ZED(ZED_csv_path):
    """
    Load ZED results CSV and return a DataFrame with an added 'Time_ms' column (milliseconds from first timestamp).
    """
    df = pd.read_csv(ZED_csv_path)
    required = [
        "Frame",
        "Translation_X", "Translation_Y", "Translation_Z",
        "Rotation_X",    "Rotation_Y",    "Rotation_Z",
        "Tracking_Fusion"
    ]
    for col in required:
        if col not in df.columns:
            print(f"Required column '{col}' is missing.")
            import sys
            sys.exit(1)

    if "Timestamp" in df.columns:
        df["Time_ms"] = (df["Timestamp"] - df["Timestamp"].iloc[0])
        df["Time_s"] = df["Time_ms"] / 1000.0

    absolute_rotation = np.linalg.norm(df[["Rotation_X", "Rotation_Y", "Rotation_Z"]].values, axis=1)
    df["Rotation_norm"] = absolute_rotation
    return df

def preprocess_zed_data(df_ZED, 
                        ZED_replace_stagnant_with_interpolation, 
                        ZED_remove_stagnant_poses, 
                        ZED_smooth_data, 
                        ZED_semantic_rebase, rebase_at_first_dict = None,
                        velocity_threshold=1e-6,
                        window_size=1):
    """
    Pre-process ZED data based on specified controls
    """
    if ZED_replace_stagnant_with_interpolation:
        df_ZED = replace_stagnant_poses_with_linear_interpolation(df_ZED, velocity_threshold)
    if ZED_remove_stagnant_poses:
        df_ZED = remove_stagnant_poses(df_ZED, velocity_threshold=velocity_threshold)
    if ZED_smooth_data:
        df_ZED = smooth_zed_data(df_ZED, window_size=window_size)
    # rebase the ZED data to set the first frame to (0,0,0) and zero rotation
    # IF "Spatial_Memory" exists, clear all frames prior to Spatial_Memory= "LOOP_CLOSED"
    if ZED_semantic_rebase and rebase_at_first_dict is not None:
        df_ZED = rebase_and_truncate_zed_data(df_ZED, rebase_at_first_dict) # only truncates
    # Extract and re-base continuous VISUAL_INERTIAL segments
    # segments = segment_visual_inertial_continuous(df_ZED)
    # rebase each segment to set the first frame to (0,0,0) and zero rotation. If there is only one segment, it will be rebased and returned as a single DataFrame.
    # [df_ZED, df_vi_original, df_vi_rebased] = rebase_and_combine_segments(segments,df_ZED)
    return df_ZED


def rebase_and_truncate_zed_data(df_ZED, rebase_at_first_dict):
    """ 
    Rebase and truncate ZED data based on semantic conditions described by rebase_at_first_dict
    e.g. for rebase_at_first_dict={"Spatial_Memory": ["KNOWN_MAP", "LOOP_CLOSED", "MAP_UPDATE", "cuVSLAM"]}, 
    the function will look for the first occurrence of Spatial_Memory being one of the specified values
    """
    print(f"Checking for semantic rebase conditions ({rebase_at_first_dict})")    
    loop_closed_index = []
    for col_name in list(rebase_at_first_dict.keys()):
        if col_name in df_ZED.columns:
            loop_closed_index = df_ZED[df_ZED[col_name].isin(rebase_at_first_dict[col_name])].index
    # If loop_closed_index is not empty, clear all frames prior to the first loop closed frame
    if not loop_closed_index.empty:
        first_loop_closed_index = loop_closed_index[0]
        df_ZED = rebase_df(df_ZED, first_loop_closed_index) #<<<<<<<<<< CURRENTLY ONLY TRUNCATE, DO NOT REBASE 
        # truncate anything before the first loop closed index 
        df_ZED = df_ZED.iloc[first_loop_closed_index:].reset_index(drop=True)
        print(f"Loop closed detected. Cleared all frames prior to index {first_loop_closed_index}.")
        # Print first two rows of rebased DataFrame for verification
        print("First two rows of rebased DataFrame:")
        print(df_ZED.head(2))
    else:
        print("No semantic rebase conditions met. No truncation applied.")
    return df_ZED

def smooth_zed_data(df, window_size=5):
    """
    Smooth the ZED data using a rolling mean with a specified window size.
    Applies to Translation_X, Translation_Y, Translation_Z, Rotation_X, Rotation_Y, Rotation_Z.
    Then recalculates Rotation_norm after smoothing.
    """
    columns_to_smooth = [
        "Translation_X", "Translation_Y", "Translation_Z",
        "Rotation_X", "Rotation_Y", "Rotation_Z",
    ]
    for col in columns_to_smooth:
        df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
    # recalculate Rotation_norm
    df["Rotation_norm"] = absolute_rotation_angle(df)
    # Report the smoothing operation
    print(f"Applied rolling mean smoothing with window size {window_size} to ZED data.")
    return df

def segment_visual_inertial_continuous(df):
    """
    Returns a list of DataFrames, each containing a continuous
    'VISUAL_INERTIAL' interval. Excludes other modes.
    A new segment starts if:
      - The mode changes from VISUAL_INERTIAL
      - The frame index jumps by more than 1
    """
    # Filter only VISUAL_INERTIAL
    df_vi = df[df["Tracking_Fusion"] == "VISUAL_INERTIAL"].copy()
    df_vi.sort_values(by="Frame", inplace=True)

    # Create an ID that increments when there's a gap > 1 frame
    df_vi["continuous_id"] = (df_vi["Frame"].diff() > 1).cumsum()

    segments = []
    for seg_id, seg_data in df_vi.groupby("continuous_id"):
        segments.append(seg_data.reset_index(drop=True))
    return segments

def rebase_and_combine_segments(segments, df_ZED):
    rebased_segments = []
    for i, seg in enumerate(segments):
        seg_rebased = rebase_segment(seg)
        rebased_segments.append(seg_rebased)
        print(f"Segment {i}, frames {seg_rebased['Frame'].min()} - "
            f"{seg_rebased['Frame'].max()}, rows={len(seg_rebased)}")
    # Combine all rebased segments into one DataFrame (disregard INERTIAL)
    if not rebased_segments:
        print("No continuous VISUAL_INERTIAL data found. Exiting.")
        import sys
        sys.exit(0)
    df_vi_rebased = pd.concat(rebased_segments, ignore_index=True)
    # Sort by Frame and reset index
    df_vi_original = df_ZED[df_ZED["Tracking_Fusion"] == "VISUAL_INERTIAL"] \
                    .sort_values("Frame") \
                    .reset_index(drop=True)
    # Rebase df_vi_original to set the first frame to (0,0,0) and zero rotation
    df_vi_original = rebase_segment(df_vi_original)
    # if only one segment is found, set df_ZED to the rebased segment
    if len(rebased_segments) == 1:
        df_ZED = rebased_segments[0]
    else:
        # If multiple segments are found, let the user decide which one to use
        print("************Multiple segments found. Select one to use as df_ZED:")
        for i, seg in enumerate(rebased_segments):
            print(f"Segment {i}: frames {seg['Frame'].min()} - "
                  f"{seg['Frame'].max()}, rows={len(seg)}")
        print("Enter the segment number to use as df_ZED (0 to {}):".format(len(rebased_segments) - 1))
        selected_segment = int(input())
        if selected_segment < 0 or selected_segment >= len(rebased_segments):
            print("Invalid segment number. Exiting.")
            import sys
            sys.exit(1)
        # Use the selected segment as df_ZED
        df_ZED = rebased_segments[selected_segment]

    return df_ZED, df_vi_original, df_vi_rebased

def rebase_segment(seg_df):
    """
    Re-center segment so the first frame is at (0,0,0) with zero rotation.
    """


    seg_T = df_to_T(seg_df)
    T0 = seg_T[0]  # First frame's transformation matrix
    seg_T  = [np.linalg.inv(T0) @ T for T in seg_T]  # Rebase all frames
    # seg_T  = [T @ np.linalg.inv(T0) for T in seg_T]  # Rebase all frames
    seg_df_rebased = T_to_df(seg_T)  # Convert back to DataFrame format
    seg_df_rebased["Frame"] = seg_df["Frame"].values  # Keep original frame indices
    seg_df_rebased["Time_ms"] = seg_df["Time_ms"].values  # Keep original timestamps

    # # First frame's pose
    # tx0 = seg_df["Translation_X"].iloc[0]
    # ty0 = seg_df["Translation_Y"].iloc[0]
    # tz0 = seg_df["Translation_Z"].iloc[0]
    # rx0 = seg_df["Rotation_X"].iloc[0]
    # ry0 = seg_df["Rotation_Y"].iloc[0]
    # rz0 = seg_df["Rotation_Z"].iloc[0]

    # R0 = Rotation.from_rotvec([rx0, ry0, rz0]).as_matrix()

    # Tprime_list = []
    # Rprime_list = []

    # for i in range(len(seg_df)):
    #     tx = seg_df["Translation_X"].iloc[i]
    #     ty = seg_df["Translation_Y"].iloc[i]
    #     tz = seg_df["Translation_Z"].iloc[i]
    #     rx = seg_df["Rotation_X"].iloc[i]
    #     ry = seg_df["Rotation_Y"].iloc[i]
    #     rz = seg_df["Rotation_Z"].iloc[i]

    #     T = np.array([tx, ty, tz])
    #     R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()

    #     # Rebase
    #     Tprime = R0.T @ (T - np.array([tx0, ty0, tz0]))
    #     Rprime = R0.T @ R

    #     rvec_prime = Rotation.from_matrix(Rprime).as_rotvec()

    #     Tprime_list.append(Tprime)
    #     Rprime_list.append(rvec_prime)

    # seg_df_rebased = seg_df.copy()
    # seg_df_rebased["Translation_X"] = [t[0] for t in Tprime_list]
    # seg_df_rebased["Translation_Y"] = [t[1] for t in Tprime_list]
    # seg_df_rebased["Translation_Z"] = [t[2] for t in Tprime_list]
    # seg_df_rebased["Rotation_X"]    = [r[0] for r in Rprime_list]
    # seg_df_rebased["Rotation_Y"]    = [r[1] for r in Rprime_list]
    # seg_df_rebased["Rotation_Z"]    = [r[2] for r in Rprime_list]

    return seg_df_rebased
############################################################

### VICON Mocap Data Loading and Processing ###
# load mocap data from a csv file and return a MocapData object
def load_mocap(csv_path):
    """
    Loads motion capture (mocap) marker data from a CSV file.

    The function reads marker names from the 5th row of the CSV file, extracts columns corresponding to markers
    whose names contain "ZED:", and loads their X, Y, Z coordinates for each frame. It returns a MocapData object
    containing the frame indices and a list of Marker objects.

    Args:
        csv_path (str): Path to the CSV file containing mocap data.

    Returns:
        MocapData: An object containing frame indices and a list of Marker objects with their coordinates.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If the CSV file format is invalid or required columns are missing.

    Note:
        - The function assumes a specific CSV format where marker names are in the 5th row and data starts after that.
        - Only markers with names containing "ZED:" are loaded.
    """
    # assume 120 HZ for VICON data
    frame_rate = 240.0
    # Read marker names from the 5th row of the CSV file and length_unit from the 7th line
    with open(csv_path, 'r') as file:
        lines = [file.readline() for _ in range(7)]
    marker_line = lines[4].strip().split(',')
    marker_names = [col for col in marker_line if col and "ZED:" in col]
    length_unit_line = [unit for unit in lines[6].strip().split(',') if unit.strip()]
    
    # Determine the length-unit of the mocap data and set the conversion factor
    # raise an error if the length units are inconsistent, otherwise use the first unit
    if len(set(length_unit_line)) != 1:
        raise ValueError("Inconsistent length units in the CSV file.")
    else:
        # Use the first unit as the length unit 
        length_unit_name = length_unit_line[0]
    # Set the conversion factor based on the length unit
    if length_unit_name.lower() == "m":
        convertion_factor = 1000.0  # Convert meters to millimeters
    elif length_unit_name.lower() == "mm":
        convertion_factor = 1.0  # Already in millimeters
        
    
    # Read the data, skipping the first 1 lines
    df_mocap = pd.read_csv(csv_path, skiprows=1, header=5)
    frames = df_mocap.iloc[:, 0].values
    # Build a list of Marker objects
    markers = []
    # The first two columns are Frame and Sub_Frame, so we start from index 2
    col_idx = 2
    for marker_name in marker_names: 
        x = df_mocap.iloc[:, col_idx].values * convertion_factor
        y = df_mocap.iloc[:, col_idx + 1].values * convertion_factor
        z = df_mocap.iloc[:, col_idx + 2].values * convertion_factor
        # Create a Marker object and append it to the list
        markers.append(Marker(marker_name, x, y, z))
        col_idx += 3
    return MocapData(frames,frame_rate, markers)







def calculate_vicon_rigid_body(df_mocap, min_markers=4, print_stats=True):
    """
    Calculates the rigid body rotation and translation in the global VICON coordinate system
    from the VICON markers in a MocapData object using the Kabsch algorithm (via SVD).
    Handles marker occlusion by using only valid (non-NaN) markers while maintaining 
    consistent translation estimates.
    
    Returns:
        df_vicon: A DataFrame containing Vicon rigid body translation and rotation data.
        residuals: np.ndarray of shape (n_frames,), RMS error per frame.
        
    Notes:
        - Uses only valid markers (non-NaN) for each frame
        - Requires minimum 3 markers for rigid body estimation
        - Returns absolute pose in global VICON coordinates
    """

    n_frames = len(df_mocap.Frame)
    n_markers = len(df_mocap.Marker) 

    # Reference marker positions from the first frame
    ref_ind = 0
    ref = np.stack([
        [df_mocap.Marker[j].X[ref_ind], df_mocap.Marker[j].Y[ref_ind], df_mocap.Marker[j].Z[ref_ind]]
        for j in range(n_markers)
    ])
    
    ref_centroid = ref.mean(axis=0)
    ref_centered = ref - ref_centroid

    translations = []
    rotations = []
    residuals = []
    abs_rot_angle = []

    for i in range(n_frames):
        # Get current marker positions and identify valid (non-NaN) markers
        pts_full = np.stack([
            [df_mocap.Marker[j].X[i], df_mocap.Marker[j].Y[i], df_mocap.Marker[j].Z[i]]
            for j in range(n_markers)
        ])
        
        # Find valid markers (non-NaN in all coordinates)
        valid_mask = ~np.isnan(pts_full).any(axis=1)
        
        if valid_mask.sum() < min_markers:  # minimum markers needed for rigid body estimation
            # Not enough valid markers - assign NaN values
            print(f"Warning: Only {valid_mask.sum()} valid markers in frame {i}, assigning NaN")
            translations.append(np.array([np.nan, np.nan, np.nan]))
            rotations.append(np.array([np.nan, np.nan, np.nan]))
            residuals.append(np.nan)
            abs_rot_angle.append(np.nan)
            continue
        
        # Extract valid markers only
        pts_valid = pts_full[valid_mask]
        ref_valid = ref[valid_mask]
        
        # Center the valid markers using their own centroids
        pts_centroid = pts_valid.mean(axis=0)
        ref_valid_centroid = ref_valid.mean(axis=0)
        
        pts_centered = pts_valid - pts_centroid
        ref_centered_valid = ref_valid - ref_valid_centroid

        # SVD/Kabsch on valid markers only
        H = pts_centered.T @ ref_centered_valid
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt



        # rotation vector and absolute rotation angle
        rotvec = Rotation.from_matrix(R).as_rotvec()
        rotations.append(rotvec)
        abs_rot_angle.append(np.linalg.norm(rotvec)) # absolute orientation in VICON coordinates

        
        # Translation: absolute position of rigid body center in VICON coordinates
        # We want the full rigid body center, corrected for missing markers
        ref_full_centroid = ref.mean(axis=0)  # Original centroid with all markers
        centroid_correction = ref_full_centroid - ref_valid_centroid  # Offset due to missing markers
        transformed_correction = R @ centroid_correction  # Apply current rotation to the correction
        translation = pts_centroid + transformed_correction  # Corrected position of full rigid body center
        translations.append(translation)

        # Compute residuals (RMS error) using only valid markers
        ref_trans = (R @ ref_centered_valid.T).T + pts_centroid
        rms = np.sqrt(np.mean(np.sum((pts_valid - ref_trans)**2, axis=1)))
        residuals.append(rms)
        
    # Create a DataFrame with the same structure as df_ZED
    data = {
        "Translation_X": np.array([t[0] for t in translations]),
        "Translation_Y": np.array([t[1] for t in translations]),
        "Translation_Z": np.array([t[2] for t in translations]),
        "Rotation_X":    np.array([r[0] for r in rotations]),
        "Rotation_Y":    np.array([r[1] for r in rotations]),
        "Rotation_Z":    np.array([r[2] for r in rotations]),
        "Rotation_norm": np.array(abs_rot_angle),
        "Frame": df_mocap.Frame,
        "Time_ms": df_mocap.Time_ms,
        "Residuals": np.array(residuals)
    }
    
    df_vicon = pd.DataFrame(data)
    

    if print_stats:
        print(f"Vicon rigid-body fitting residuals stats (mm): "
        f"min={df_vicon['Residuals'].min():.6f}, "
        f"max={df_vicon['Residuals'].max():.6f}, "
        f"mean={df_vicon['Residuals'].mean():.6f}, "
        f"len={len(df_vicon['Residuals'])}")

    return df_vicon, np.array(residuals)


def plot_vicon_rigid_body(df_vicon, title=None):
    """
    Plot the 3D trajectory of the rigid body translation and the orientation (as axes) over time from a MocapData object.
    """
    

    # Extract translation and rotation arrays
    tx = df_vicon.Translation_X.to_numpy()
    ty = df_vicon.Translation_Y.to_numpy()
    tz = df_vicon.Translation_Z.to_numpy()
    # Assuming Rotation_X, Y, Z are in radians and represent rotation vectors
    rx = df_vicon.Rotation_X.to_numpy()
    ry = df_vicon.Rotation_Y.to_numpy()
    rz = df_vicon.Rotation_Z.to_numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tx, ty, tz, label='Rigid Body Trajectory', color='blue')
    ax.scatter(tx[0], ty[0], tz[0], color='green', marker='o', s=60, label='Start')
    ax.scatter(tx[-1], ty[-1], tz[-1], color='red', marker='x', s=60, label='End')

    # Draw orientation axes at intervals
    n = len(tx)
    step = max(1, n // 30)  # Plot at most 30 axes for clarity
    axis_len = 20  # Length of orientation axes (adjust as needed)
    for i in range(0, n, step):
        R = Rotation.from_rotvec([rx[i], ry[i], rz[i]]).as_matrix()
        origin = np.array([tx[i], ty[i], tz[i]])
        for j, color in enumerate(['r', 'g', 'b']):
            vec = R[:, j] * axis_len
            ax.quiver(
                origin[0], origin[1], origin[2],
                vec[0], vec[1], vec[2],
                color=color, length=axis_len, normalize=True, arrow_length_ratio=0.2, linewidth=1.5
            )

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    if title:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True)
    try:
        plt.tight_layout()
    except (AttributeError, ValueError):
        pass  # tight_layout may fail with 3D plots or NumPy 2.0 compatibility
    plt.show(block=False)

    return plt

def plot_zed_vicon_orientation(df_zed, df_mocap, title=None):
    """
    Plot the orientation (Rotation_X/Y/Z) of ZED and VICON data over time for comparison.
    df_zed: DataFrame with ZED results (columns: Frame, Rotation_X, Rotation_Y, Rotation_Z)
    df_mocap: MocapData object with .Frame, .Rotation_X, .Rotation_Y, .Rotation_Z
    """
    # ZED data
    time_zed = df_zed["Time_ms"]
    rx_zed = df_zed["Rotation_X"]
    ry_zed = df_zed["Rotation_Y"]
    rz_zed = df_zed["Rotation_Z"]
    # VICON data
    time_vicon = df_mocap.Time_ms  # Assuming Time_ms is in milliseconds
    rx_vicon = df_mocap.Rotation_X
    ry_vicon = df_mocap.Rotation_Y
    rz_vicon = df_mocap.Rotation_Z

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(time_zed, rx_zed, color="r", label="ZED Rotation_X")
    axes[0].plot(time_vicon, rx_vicon, color="r", linestyle='--', label="VICON Rotation_X")
    axes[1].plot(time_zed, ry_zed, color="g", label="ZED Rotation_Y")
    axes[1].plot(time_vicon, ry_vicon, color="g", linestyle='--', label="VICON Rotation_Y")
    axes[2].plot(time_zed, rz_zed, color="b", label="ZED Rotation_Z")
    axes[2].plot(time_vicon, rz_vicon, color="b", linestyle='--', label="VICON Rotation_Z")

    axes[0].set_ylabel("Rot X (rad)")
    axes[1].set_ylabel("Rot Y (rad)")
    axes[2].set_ylabel("Rot Z (rad)")
    axes[2].set_xlabel("Time (ms)")

    for ax in axes:
        ax.legend()
        ax.grid(True)
    if title:
        plt.suptitle(title)
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)

    return plt

def df_to_T(df):
        T_list = []
        for i in range(len(df)):
            tx = df["Translation_X"].iloc[i]
            ty = df["Translation_Y"].iloc[i]
            tz = df["Translation_Z"].iloc[i]
            rx = df["Rotation_X"].iloc[i]
            ry = df["Rotation_Y"].iloc[i]
            rz = df["Rotation_Z"].iloc[i]

            R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            T_list.append(T)
        return T_list    

def T_to_df(T_list):
    data = {
        "Translation_X": [T[0, 3] if not np.isnan(T).any() else np.nan for T in T_list],
        "Translation_Y": [T[1, 3] if not np.isnan(T).any() else np.nan for T in T_list],
        "Translation_Z": [T[2, 3] if not np.isnan(T).any() else np.nan for T in T_list],
        "Rotation_X": [
            Rotation.from_matrix(T[:3, :3]).as_rotvec()[0] if not np.isnan(T).any() else np.nan
            for T in T_list
        ],
        "Rotation_Y": [
            Rotation.from_matrix(T[:3, :3]).as_rotvec()[1] if not np.isnan(T).any() else np.nan
            for T in T_list
        ],
        "Rotation_Z": [
            Rotation.from_matrix(T[:3, :3]).as_rotvec()[2] if not np.isnan(T).any() else np.nan
            for T in T_list
        ],
    }
    return pd.DataFrame(data)
def absolute_rotation_angle(df):
    """
    Calculate the and returns absolute rotation angle from the rotation vectors in a DataFrame.
    The rotation vectors are assumed to be in radians.
    """
    angles = np.linalg.norm(df[["Rotation_X", "Rotation_Y", "Rotation_Z"]].to_numpy(), axis=1)
    return np.array(angles)

def marker_centeroid(markers):
    """
    Calculate the centroid of a list of markers.
    Each marker is assumed to have X, Y, Z attributes.
    """
    x = np.mean([marker.X for marker in markers],axis=0)
    y = np.mean([marker.Y for marker in markers],axis=0)
    z = np.mean([marker.Z for marker in markers],axis=0)
    return (x, y, z)

def estimate_temporal_offset_zed_vicon(df_zed, df_mocap,discard_fraction):
    """
    Estimate the temporal offset (lag) between ZED and VICON data using cross-correlation of angular velocities.
    Positive lag means VICON is ahead of ZED, negative lag means ZED is ahead of VICON.
    discard_fraction: fraction of data to discard from start and end for lag estimation. If None or 0.0, use the whole data.
    If negative, use only the first abs(discard_fraction) of the data.
    Returns the estimated lag in milliseconds.
    """
    from scipy.signal import correlate
    n = len(df_zed)
    if discard_fraction is None or discard_fraction == 0.0:
        # Use the whole data
        start = 0
        end = n
    elif discard_fraction < 0.0:
        # use only first abs(discard_fraction) of the data
        start = 0
        end = int(n * abs(discard_fraction))
    else:
        # Calculate start and end indices based on discard_fraction
        start = int(n * discard_fraction)
        end = n - int(n * discard_fraction)
    df_zed = df_zed.iloc[start:end].reset_index(drop=True)
    print(f"Using segment of ZED data from t={df_zed['Time_ms'].iloc[0]} ms ({df_zed['Frame'].iloc[0]}) to t={df_zed['Time_ms'].iloc[-1]} ms ({df_zed['Frame'].iloc[-1]}) to estimate lag.")

    # Use canonical time axes in milliseconds
    t_zed = df_zed['Time_ms']
    t_vicon = df_mocap.Time_ms

    # Differentiated absolute rotation angle to get angula
    abs_angle_zed = np.sqrt(df_zed.Rotation_X**2 + df_zed.Rotation_Y**2 + df_zed.Rotation_Z**2)
    abs_angle_vicon = np.sqrt(df_mocap.Rotation_X**2 + df_mocap.Rotation_Y**2 + df_mocap.Rotation_Z**2)
    # angular_velocity_zed = np.gradient(abs_angle_zed, t_zed)  # Rate of change
    # angular_velocity_vicon_full = np.gradient(abs_angle_vicon, t_vicon)
    
    t_zed , angular_velocity_zed = angular_speed_magnitude_from_rotvec(df_zed)
    t_vicon , angular_velocity_vicon_full = angular_speed_magnitude_from_rotvec(df_mocap)
    

    # Interpolate VICON angular velocity to ZED time axis
    angular_velocity_vicon = np.interp(t_zed, t_vicon, angular_velocity_vicon_full)
    t_zed = df_zed['Time_ms']  # Ensure t_zed is from the original DataFrame for accurate lag conversion
    # Designate and normalize signals for cross-correlation
    zed_sig = angular_velocity_zed
    vicon_sig = angular_velocity_vicon
    def normalize_signal(sig):
        sig = sig - np.mean(sig)  # Remove DC offset
        std = np.std(sig)
        if std < 1e-8:  # Avoid division by near-zero
            return sig
        return sig / std
    zed_sig = normalize_signal(zed_sig)
    vicon_sig = normalize_signal(vicon_sig) 
    
    # Cross-correlation
    from scipy.optimize import fminbound
    from scipy.interpolate import interp1d
    corr = correlate(zed_sig, vicon_sig, mode='full')
    corr_normalized = corr / (len(zed_sig))  # Normalize by length
    # Find the lag at maximum correlation
    lags = np.arange(-len(zed_sig)+1, len(zed_sig))
    max_ind = np.argmax(corr_normalized)
    lag_samples_discrete = lags[max_ind]
    
    # Sub-sample refinement using scipy.optimize.fminbound with cubic interpolation
    corr_interp = interp1d(lags, corr_normalized, kind='cubic', fill_value='extrapolate')
    # fminbound minimizes, so negate the correlation function to find the maximum
    lag_samples_subsample = fminbound(lambda x: -corr_interp(x), 
                                      lag_samples_discrete - 1.0, 
                                      lag_samples_discrete + 1.0,
                                      xtol=1e-8)  # High precision
    
    # Convert lags to milliseconds using exact time shift
    # lag_samples_subsample is the number of samples to shift
    # Compute time difference by taking samples at different positions
    lag_idx_lower = int(np.floor(lag_samples_subsample))
    lag_idx_upper = int(np.ceil(lag_samples_subsample))
    frac = lag_samples_subsample - lag_idx_lower
    
    # Get time differences at two nearby sample offsets
    if lag_idx_lower >= 0:
        t_lower = t_zed.iloc[min(lag_idx_lower, len(t_zed)-1)] - t_zed.iloc[0]
    else:
        t_lower = -(t_zed.iloc[min(-lag_idx_lower, len(t_zed)-1)] - t_zed.iloc[0])
    
    if lag_idx_upper >= 0:
        t_upper = t_zed.iloc[min(lag_idx_upper, len(t_zed)-1)] - t_zed.iloc[0]
    else:
        t_upper = -(t_zed.iloc[min(-lag_idx_upper, len(t_zed)-1)] - t_zed.iloc[0])
    
    lag_ms = t_lower + frac * (t_upper - t_lower)
    print(f"Estimated temporal offset (lag): {lag_ms:.2f} ms (positive: VICON ahead, negative: ZED ahead)")
    print(f"Discrete lag: {lag_samples_discrete} samples, Sub-sampled lag: {lag_samples_subsample:.4f} samples")
    print(f"Max correlation value: {corr_normalized[max_ind]:.4f}")
    return lag_ms


def synchronize_zed_vicon(df_zed, df_rb, df_mocap, lag_ms):
    """
    Shift the VICON (mocap) Time_ms axis by the estimated lag (in ms) so that it is synchronized with the ZED Time_ms axis.
    Returns the updated df_zed and df_rb (with shifted Time_ms) and a new DataFrame with interpolated VICON data to match ZED timestamps.
    """
    # Adjust VICON and MoCap timestamps by the lag
    # df_rb.Time_ms = df_rb.Time_ms + lag_ms
    df_mocap.Time_ms = df_mocap.Time_ms + lag_ms

    # Interpolate VICON data to match ZED timestamps
    interpolated_data = {
    #     'Translation_X': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Translation_X),
    #     'Translation_Y': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Translation_Y),
    #     'Translation_Z': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Translation_Z),
    #     'Rotation_X': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Rotation_X),
    #     'Rotation_Y': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Rotation_Y),
    #     'Rotation_Z': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Rotation_Z),
        'Frame': np.interp(df_zed['Time_ms'], df_mocap.Time_ms, df_mocap.Frame)
    }

    # Create a new DataFrame for the interpolated VICON data
    # df_rb_interpolated = pd.DataFrame(interpolated_data)
    # df_rb_interpolated['Time_ms'] = df_zed['Time_ms']
    mocap_marker_interpolated = []
    for marker in df_mocap.Marker:
        # Interpolate each marker's position to match ZED timestamps
        marker_data = {
            'X': np.interp(df_zed['Time_ms'], df_mocap.Time_ms, marker.X),
            'Y': np.interp(df_zed['Time_ms'], df_mocap.Time_ms, marker.Y),
            'Z': np.interp(df_zed['Time_ms'], df_mocap.Time_ms, marker.Z)
        }
        mocap_marker_interpolated.append(Marker(marker.name, marker_data['X'],marker_data['Y'],marker_data['Z']))

    df_mocap_interpolated = MocapData(frames=interpolated_data["Frame"], frame_rate=df_mocap.Frame_Rate, markers=mocap_marker_interpolated, Time_ms=df_zed['Time_ms'].to_numpy())

    # return df_zed, df_rb, df_rb_interpolated, df_mocap, df_mocap_interpolated
    return df_mocap_interpolated

def synchronize_zed_vicon_trial(df_zed_og, df_rb_og, lag_ms, discard_fraction=None):
    """
    Shift the VICON (mocap) Time_ms axis by the estimated lag (in ms) so that it is synchronized with the ZED Time_ms axis.
    Returns the updated df_zed and df_rb (with shifted Time_ms) and a new DataFrame with interpolated VICON data to match ZED timestamps.
    discard_fraction: fraction of data to discard from start and end to avoid edge effects (e.g., 0.05 for 5%)
    0.0 or None means no discard.
    """
    df_zed = df_zed_og.copy()
    df_rb = df_rb_og.copy()
    # discard first and last discard_fraction of the data to avoid edge effects
    # If discard_fraction is None, use the whole data
    n = len(df_zed)
    if discard_fraction is None or discard_fraction == 0.0:
        # Use the whole data
        start = 0
        end = n
    elif discard_fraction < 0.0:
        # use only first abs(discard_fraction) of the data
        start = 0
        end = int(n * abs(discard_fraction))
    else:
        # Calculate start and end indices based on discard_fraction
        start = int(n // (1 / discard_fraction))
        end = n - int(n // (1 / discard_fraction))




    df_zed = df_zed.iloc[start:end].reset_index(drop=True)
    # Adjust VICON and MoCap timestamps by the lag
    df_rb.Time_ms = df_rb.Time_ms + lag_ms
    t_zed = df_zed['Time_ms']
    t_vicon = df_rb['Time_ms']
    

    # Interpolate VICON data to match ZED timestamps
    # abs_angle_zed = np.sqrt(df_zed.Rotation_X**2 + df_zed.Rotation_Y**2 + df_zed.Rotation_Z**2)
    # abs_angle_vicon = np.sqrt(df_rb.Rotation_X**2 + df_rb.Rotation_Y**2 + df_rb.Rotation_Z**2)
    # angular_velocity_zed = np.gradient(abs_angle_zed, t_zed)  # Rate of change
    # angular_velocity_vicon_full = np.gradient(abs_angle_vicon, t_vicon)
    t_mid_vicon, omega_vicon = angular_speed_magnitude_from_rotvec(df_rb)
    t_mid_zed, omega_zed = angular_speed_magnitude_from_rotvec(df_zed)




    # Interpolate VICON angular velocity to ZED time axis
    # omega_vicon_interp = np.interp(t_mid_zed, t_mid_vicon, omega_vicon)
    # error = omega_zed - omega_vicon_interp
    # rms = np.sqrt(np.mean(error**2))
    

    
 

    
    omega_vicon_interp = CubicSpline(t_mid_vicon, omega_vicon)(t_mid_zed)
    inside = (t_mid_zed >= t_mid_vicon.min()) & (t_mid_zed <= t_mid_vicon.max())
    error = (omega_zed - omega_vicon_interp)[inside]
    rms = np.sqrt(np.mean(error**2))

    
    # Interpolate VICON angular velocity to ZED time axis


    # interpolated_data = {
    #     'Translation_X': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Translation_X),
    #     'Translation_Y': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Translation_Y),
    #     'Translation_Z': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Translation_Z),
    #     'Rotation_X': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Rotation_X),
    #     'Rotation_Y': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Rotation_Y),
    #     'Rotation_Z': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Rotation_Z),
    #     'Frame': np.interp(df_zed['Time_ms'], df_rb.Time_ms, df_rb.Frame)
    # }

    # # Create a new DataFrame for the interpolated VICON data
    # df_rb_interpolated = pd.DataFrame(interpolated_data)
    # df_rb_interpolated['Time_ms'] = df_zed['Time_ms']

    # return df_zed, df_rb_interpolated
    return rms



def plot_orientation_signals_vs_time(df_zed, df_mocap, lag=0, zed_dt=None, vicon_rate=240):
    """
    Plot the orientation signals (Rotation_X/Y/Z) of ZED and VICON as a function of time (seconds),
    with optional lag applied to VICON. ZED time is computed from timestamps if available, otherwise uses frame index * zed_dt.
    VICON time is computed as Frame index / vicon_rate.
    """
    # ZED time axis
    if 'Timestamp' in df_zed.columns:
        t_zed = (df_zed['Timestamp'] - df_zed['Timestamp'].iloc[0]) / 1e9  # assuming nanoseconds
    elif zed_dt is not None:
        t_zed = (df_zed['Frame'] - df_zed['Frame'].iloc[0]) * zed_dt
    else:
        t_zed = (df_zed['Frame'] - df_zed['Frame'].iloc[0]) * (1/30)  # fallback: assume 30Hz

    # VICON time axis
    t_vicon = (df_mocap.Frame - df_mocap.Frame[0]) / vicon_rate
    # Apply lag (in frames) to VICON if needed
    if lag != 0:
        t_vicon = t_vicon + lag / vicon_rate

    # Plot
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 7))
    axes[0].plot(t_zed, df_zed['Rotation_X'], 'r', label='ZED Rotation_X')
    axes[0].plot(t_vicon, df_mocap.Rotation_X, 'r--', label='VICON Rotation_X')
    axes[1].plot(t_zed, df_zed['Rotation_Y'], 'g', label='ZED Rotation_Y')
    axes[1].plot(t_vicon, df_mocap.Rotation_Y, 'g--', label='VICON Rotation_Y')
    axes[2].plot(t_zed, df_zed['Rotation_Z'], 'b', label='ZED Rotation_Z')
    axes[2].plot(t_vicon, df_mocap.Rotation_Z, 'b--', label='VICON Rotation_Z')

    axes[0].set_ylabel('Rot X (rad)')
    axes[1].set_ylabel('Rot Y (rad)')
    axes[2].set_ylabel('Rot Z (rad)')
    axes[2].set_xlabel('Time (s)')
    for ax in axes:
        ax.legend()
        ax.grid(True)
    plt.suptitle('ZED vs VICON Orientation Signals vs Time (lag={} frames)'.format(lag))
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)

    return plt

# def estimate_temporal_offset(df_zed, df_rb, max_lag=200):
#     """
#     Estimate the temporal offset (in milliseconds) between ZED and VICON orientation signals
#     using cross-correlation of the absolute angle of rotation (norm of rotation vector) signals.
#     Returns the lag in milliseconds (positive: VICON ahead, negative: ZED ahead).
#     Uses df_zed['Time_ms'] as the canonical ZED time axis (already in ms).
#     """
#     from scipy.signal import correlate
#     # discard first and last 5% of the data to avoid edge effects
#     # n = len(df_zed)
#     # start = n // 20
#     # end = n - n // 20
#     # df_zed = df_zed.iloc[start:end].reset_index(drop=True)
#     print(f"Using segment of ZED data from t={df_zed['Time_ms'].iloc[0]} ms ({df_zed['Frame'].iloc[0]}) to t={df_zed['Time_ms'].iloc[-1]} ms ({df_zed['Frame'].iloc[-1]}) to estimate lag.")

#     # Use canonical time axes in milliseconds
#     t_zed = df_zed['Time_ms']
#     t_rb = df_rb.Time_ms

#     ## Compute absolute rotation angle (norm of rotation vector)
#     # find the first frame index in df_zed where Spatial_Memory is OK
#     # first_ok_frame = df_zed[df_zed['Spatial_Memory'] == 'OK'].index[0]
#     # first_ok_frame = df_zed[ df_zed['Spatial_Memory']== 'MAP_UPDATE'].index[0]
#     # Rebase ZED and VICON DataFrames to start with the first frame
#     df_zed_rebased = rebase_df(df_zed)
#     df_rb_rebased = rebase_df(df_rb)
    
#     # Compute absolute rotation angle (norm of rotation vector)
#     abs_angle_zed = np.sqrt(df_zed_rebased["Rotation_X"]**2 + df_zed_rebased["Rotation_Y"]**2 + df_zed_rebased["Rotation_Z"]**2)
#     abs_angle_rb = np.sqrt(df_rb_rebased["Rotation_X"]**2 + df_rb_rebased["Rotation_Y"]**2 + df_rb_rebased["Rotation_Z"]**2)
    
#     # Interpolate VICON absolute angle to ZED time axis
#     abs_angle_rb_interp = np.interp(t_zed, t_rb, abs_angle_rb)

#     # Normalize signals
#     def normsig(sig):
#         return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
#     zed_sig = normsig(abs_angle_zed)
#     rb_sig = normsig(abs_angle_rb_interp)
#     # Cross-correlation
#     corr = correlate(zed_sig, rb_sig, mode='full')
#     lags = np.arange(-len(zed_sig)+1, len(zed_sig))
#     # Convert lags to milliseconds
#     dt = np.mean(np.diff(df_zed['Time_ms'].values))  # ms
#     lags_ms = lags * dt
#     # Limit to max_lag (in samples)
#     mid = len(corr) // 2
#     search_range = slice(mid - max_lag, mid + max_lag + 1)
#     lag_ms = lags_ms[search_range][np.argmax(corr[search_range])]
#     print(f"Estimated temporal offset (lag, abs angle): {lag_ms:.2f} ms (positive: VICON ahead, negative: ZED ahead)")
#     #  print np.max(corr[search_range]) as a proxy for alignment confidence.
#     print(f"Max correlation value: {np.max(corr[search_range]):.4f}")
#     # Return the estimated lag in milliseconds
#     return lag_ms

def rebase_df(df,base_ind=0):
    # Convert DataFrame to transformation matrices
    T = df_to_T(df)
    # Rebase each transformation matrix to start with the base index
    T_rebased = [np.linalg.inv(T[base_ind]) @ T[i] for i in range(len(T))]
    # Convert back to DataFrame
    df_rebased = T_to_df(T_rebased)
    # Keep original frame indices and timestamps
    df_rebased["Frame"] = df["Frame"].values
    df_rebased["Time_ms"] = df["Time_ms"].values-df["Time_ms"].values[base_ind]  # Rebase time to start at zero
    df_rebased["Time_s"] = df_rebased["Time_ms"]/1000.0
    # recalculate Rotation_norm
    df_rebased["Rotation_norm"] = absolute_rotation_angle(df_rebased)
    # inherit all other columns from the original DataFrame
    for col in df.columns:
        if col not in df_rebased.columns:
            df_rebased[col] = df[col].values
    return df_rebased

def plot_absolute_rotation_angle(df_zed, df_rb, rb_fit=None, title="Absolute Rotation Angle and Rate (ZED & VICON)"):
    """
    Plot the absolute rotation angle (norm of rotation vector and incremental angle) for ZED and VICON on a single axis for direct comparison.
    Optionally includes interpolated VICON data if provided.
    """
    marker_size = 3  # Size of markers for ZED and VICON plots
    line_width = 1 # Width of lines for ZED and VICON plots

    time_zed = df_zed["Time_ms"]
    time_vicon = df_rb["Time_ms"]

    abs_angle_zed = absolute_rotation_angle(df_zed)
    abs_angle_rb = absolute_rotation_angle(df_rb)

    if rb_fit is not None:
        # rb_fit.angles() is already rebased relative to the first frame
        abs_angle_rb_fit = np.linalg.norm(rb_fit.angles(), axis=1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(title)
    ## Subplot 1: Absolute Rotation Angle
    ax1.plot(time_zed, abs_angle_zed, color="purple", label="ZED |rotvec|", linewidth=line_width, marker='x', markersize=marker_size)
    ax1.plot(time_vicon, abs_angle_rb, color="orange", label="VICON |rotvec|", linewidth=line_width, marker='+', markersize=marker_size)
    if rb_fit is not None:
        ax1.plot(time_vicon, abs_angle_rb_fit, color="blue", label="RB fit |rotvec|", linewidth=line_width, marker='.', markersize=marker_size)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Rotation Angle (rad)")
    ax1.grid(True)
    ax1.legend()
    ## Subplot 2
    ### OLD Absoulute Rotation Angle Rate
    # ax2.plot(time_zed, np.diff(abs_angle_zed, prepend=0)/np.diff(time_zed, prepend=1), color="purple", label="ZED |rotvec|",linewidth=line_width, marker='x', markersize=marker_size)
    # ax2.plot(time_vicon, np.diff(abs_angle_rb, prepend=0)/np.diff(time_vicon, prepend=1), color="orange", label="VICON |rotvec|", linewidth=line_width, marker='+', markersize=marker_size)
    # ax2.set_ylabel("Rotation Rate (rad/ms)")
    ### NEW: Rotational increments from rotvec differentiation
    t_mid_vicon, omega_vicon = angular_speed_magnitude_from_rotvec(df_rb)
    t_mid_zed, omega_zed = angular_speed_magnitude_from_rotvec(df_zed)
    ax2.plot(t_mid_vicon, omega_vicon*1e3, color="blue", label="VICON Angular Speed", linewidth=line_width, marker=None, markersize=marker_size, linestyle='-')
    ax2.plot(t_mid_zed, omega_zed*1e3, color="purple", label="ZED Angular Speed", linewidth=line_width, marker=None, markersize=marker_size, linestyle='-')
    ax2.set_ylabel("Rotation Rate (rad/s)")

    

    ax2.set_xlabel("Time (ms)")
    ax2.grid(True)
    ax2.legend()
    try:
        plt.tight_layout()
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)
    return plt

def rotational_increments_from_rotvec(df, cols=("Rotation_X", "Rotation_Y", "Rotation_Z")):
    """
    Compute relative rotation increments as rotation vectors.

    df: DataFrame with rotation vectors in radians (axis angle) in columns cols
    Returns: (N-1, 3) array of relative rotation vectors
    """
    rv = df.loc[:, cols].to_numpy(dtype=float)            # (N, 3)
    Rabs = Rotation.from_rotvec(rv)                       # length N Rotation object

    dR = Rabs[1:] * Rabs[:-1].inv()                       # relative rotations
    drv = dR.as_rotvec()                                  # (N-1, 3)
    return drv

def angular_speed_magnitude_from_rotvec(df, t_col="Time_ms", cols=("Rotation_X", "Rotation_Y", "Rotation_Z")):
    """
    Scalar angular speed magnitude series for correlation.
    Returns: times_mid (N-1,), omega (N-1,)
    """
    # remove rows with NaN in rotation columns
    df = df.dropna(subset=cols)
    t = df[t_col].to_numpy(dtype=float)
    dt = np.diff(t)
    drv = rotational_increments_from_rotvec(df, cols=cols)

    angle = np.linalg.norm(drv, axis=1)                   # radians
    omega = angle / dt                                    # rad per second

    t_mid = 0.5 * (t[1:] + t[:-1])
    return t_mid, omega

# def calculate_absolute_rotation_angle_error(df_zed, df_rb, rb_fit=None, title="Absolute Rotation Angle (ZED & VICON)"):
#     """
#     Calculate the absolute rotation angle (norm of rotation vector) for ZED and VICON on a single axis for direct comparison.
#     Optionally includes interpolated VICON data if provided.
#     """
   

#     ## Compute absolute rotation angle (norm of rotation vector)
#     # find the first frame index in df_zed where Spatial_Memory is OK
#     # first_ok_frame = df_zed[df_zed['Spatial_Memory']== 'OK'].index[0]
#     # first_ok_frame = df_zed[ df_zed['Spatial_Memory']== 'MAP_UPDATE'].index[0]
#     # Rebase ZED and VICON DataFrames to start with the first frame
#     df_zed_rebased = rebase_df(df_zed)
#     df_rb_rebased = rebase_df(df_rb)
    
#     # Compute absolute rotation angle (norm of rotation vector)
#     abs_angle_zed = np.diff(np.sqrt(df_zed_rebased["Rotation_X"]**2 + df_zed_rebased["Rotation_Y"]**2 + df_zed_rebased["Rotation_Z"]**2))
#     abs_angle_rb = np.diff(np.sqrt(df_rb_rebased["Rotation_X"]**2 + df_rb_rebased["Rotation_Y"]**2 + df_rb_rebased["Rotation_Z"]**2))
    
#     # Calculate error 
#     error = abs_angle_zed - abs_angle_rb
#     # Calculate root mean square error (RMS)
#     rms = np.sqrt(np.mean(error**2))
#     return rms

def estimate_relative_transform(T_As, T_Bs):
    """
    Estimate the fixed transform T_BA such that:
        T_B[n] ≈ T_BA @ T_A[n]
    Inputs:
        T_As: list of 4×4 matrices (poses of point A in its frame)
        T_Bs: list of 4×4 matrices (poses of point B in its frame)
    Returns:
        T_BA: 4×4 matrix (fixed transform from A to B)
    """
    rel_rotations = []
    rel_translations = []

    for T_A, T_B in zip(T_As, T_Bs):
        T_A_inv = np.linalg.inv(T_A)
        T_BA_i = T_B @ T_A_inv
        rel_rotations.append(T_BA_i[:3, :3])
        rel_translations.append(T_BA_i[:3, 3])

    # Average rotation
    R_avg = average_rotations(rel_rotations)
    t_avg = np.mean(rel_translations, axis=0)

    T_BA = np.eye(4)
    T_BA[:3, :3] = R_avg
    T_BA[:3, 3] = t_avg
    return T_BA

def homogenous_transform_from_vec(rotvec,transvec):
    """    Create a 4x4 homogenous transformation matrix from a rotation vector and translation vector.
    Args:
        rotvec (array-like): Rotation vector (3 elements).
        transvec (array-like): Translation vector (3 elements).                     
    Returns:
        np.ndarray: 4x4 homogenous transformation matrix.
    """
    # Create rotation matrix from rotation vector
    R_mat = Rotation.from_rotvec(rotvec).as_matrix()
    # Create homogenous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = transvec
    return T

def invert_transform(T):
    """Invert a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def plot_df_to_df_distance(df_A, df_B, loop_closed_indices=None, title="Distance between two DataFrames"):
    """
    Plot the distance between two DataFrames in the same frame of reference.
    """    
    tx_A = df_A["Translation_X"].to_numpy()
    ty_A = df_A["Translation_Y"].to_numpy()
    tz_A = df_A["Translation_Z"].to_numpy()

    tx_B = df_B["Translation_X"].to_numpy()
    ty_B = df_B["Translation_Y"].to_numpy()
    tz_B = df_B["Translation_Z"].to_numpy()

    distances = np.sqrt(
        (tx_A - tx_B) ** 2 +
        (ty_A - ty_B) ** 2 +
        (tz_A - tz_B) ** 2
    )

    plt.figure()
    # plt.plot(distances, label="Distance between A and B")
    plt.plot(distances, label="Distance between ${}^{V}\mathbf{T}_{W}\,{}^{W}\mathbf{T}_{C}(t)$ and ${}^{V}\mathbf{T}_{M}(t)\,{}^{M}\mathbf{T}_{C}$")


    # Mark loop closure data points based on positional_tracking state
    # if loop_closed_indices is not None:
    #     loop_closure_distances = distances[loop_closed_indices]
    #     plt.scatter(loop_closed_indices, loop_closure_distances, color="red", label="Loop Closure Points", zorder=5)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Distance (mm)")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    return plt, distances

def plot_t_distance_from_origin(t, title="Distance from Origin"):
    """
    Plot the distance from the origin for a given translation vector.
    """
    distances = np.linalg.norm(t,axis=1)  # Assuming t is a Nx3 array of translations


    plt.figure()
    plt.plot(distances, label="Distance from Origin")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Distance (mm)")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    return plt, distances

# def average_rotations(rotations):
#     """Average a list of 3x3 rotation matrices and project onto SO(3) via SVD."""
#     R_stack = np.stack(rotations, axis=0)
#     R_mean = np.mean(R_stack, axis=0)
#     U, _, Vt = np.linalg.svd(R_mean)
#     R_avg = U @ Vt
#     # Ensure a right-handed coordinate system (determinant = 1)
#     if np.linalg.det(R_avg) < 0:
#         U[:, -1] *= -1
#         R_avg = U @ Vt
#     return R_avg

def average_rotations(rotations):
    """
    Average a list of 3x3 rotation matrices.
    Returns the average rotation as a 3x3 matrix.
    """
    if not rotations:
        raise ValueError("Rotation list is empty.")

    # Convert all rotations to quaternions
    quats = np.array([Rotation.from_matrix(Ri).as_quat() for Ri in rotations])  # shape: (N, 4)

    # Ensure all quaternions are on the same hemisphere
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]

    # Average and normalize
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    # Convert back to rotation matrix
    return Rotation.from_quat(mean_quat).as_matrix()

def optimal_constant_transform(T_As, T_Bs):
    """
    T_As, T_Bs: lists of 4x4 numpy arrays representing poses of A and B at each time.
    Returns T_AB: 4x4 numpy array, the optimal constant transformation from A to B.
    """
    rel_rotations = []
    rel_translations = []
    for T_A, T_B in zip(T_As, T_Bs):
        T_A_inv = invert_transform(T_A)
        T_AB_t = T_B @ T_A_inv
        R = T_AB_t[:3, :3]
        t = T_AB_t[:3, 3]
        rel_rotations.append(R)
        rel_translations.append(t)
    R_AB = average_rotations(rel_rotations)
    t_AB = np.mean(np.stack(rel_translations, axis=0), axis=0)
    T_AB = np.eye(4)
    T_AB[:3, :3] = R_AB
    T_AB[:3, 3] = t_AB
    return T_AB

###############
def estimate_T_BA(T_As, T_Bs):
    T_As = np.asarray(T_As, dtype=float)
    T_Bs = np.asarray(T_Bs, dtype=float)

    if T_As.shape != T_Bs.shape or T_As.shape[-2:] != (4, 4):
        raise ValueError("T_As and T_Bs must be Nx4x4 arrays of the same length.")

    # Split into rotation (R) and translation (t)
    R_As = T_As[:, :3, :3]
    R_Bs = T_Bs[:, :3, :3]
    t_As = T_As[:, :3, 3]
    t_Bs = T_Bs[:, :3, 3]

    # --------- 1. Optimal rotation (^B R_A) via Orthogonal Procrustes ---------
    M = np.zeros((3, 3))
    for R_A, R_B in zip(R_As, R_Bs):
        M += R_A.T @ R_B                       # sum_i (R_Aᵀ R_B)

    U, _, Vt = np.linalg.svd(M)
    R_BA = U @ Vt
    # Enforce a proper rotation (det = +1)
    if np.linalg.det(R_BA) < 0:
        U[:, -1] *= -1
        R_BA = U @ Vt

    # --------- 2. Optimal translation (^B t_A)  -------------------------------
    # Solve  R_A · t_BA  =  t_B - t_A   for all samples, least-squares
    A = np.concatenate(R_As, axis=0)           # (3N × 3)
    b = (t_Bs - t_As).reshape(-1, 1)           # (3N × 1)
    t_BA, *_ = np.linalg.lstsq(A, b, rcond=None)
    t_BA = t_BA.flatten()

    # --------- 3. Assemble homogeneous matrix ---------------------------------
    T_BA = np.eye(4)
    T_BA[:3, :3] = R_BA
    T_BA[:3, 3]  = t_BA
    

    return T_BA





def plot_zed_path_in_vicon_frame_of_reference(df_zed_in_vicon, df_mocap_interpolated,T_zed_as_marker, df_rb_interpolated,df_zed_as_marker, title="ZED Path in VICON Frame of Reference"):
    

    # Call this in your slider update function:
    mocap_marker_interpolated = df_mocap_interpolated.Marker

    # Prepare zed pose data (already in mm)
    zed_x = df_zed_in_vicon["Translation_X"].values
    zed_y = df_zed_in_vicon["Translation_Y"].values
    zed_z = df_zed_in_vicon["Translation_Z"].values

    # Collect all marker coords
    marker_x_all, marker_y_all, marker_z_all = [], [], []
    if mocap_marker_interpolated is not None:
        for marker in mocap_marker_interpolated:
            marker_x_all.extend(marker.X)
            marker_y_all.extend(marker.Y)
            marker_z_all.extend(marker.Z)

    # Combine everything to find global min/max
    all_x = np.concatenate([zed_x, marker_x_all]) if marker_x_all else zed_x
    all_y = np.concatenate([zed_y, marker_y_all]) if marker_y_all else zed_y
    all_z = np.concatenate([zed_z, marker_z_all]) if marker_z_all else zed_z

    min_x, max_x = np.nanmin(all_x), np.nanmax(all_x)
    min_y, max_y = np.nanmin(all_y), np.nanmax(all_y)
    min_z, max_z = np.nanmin(all_z), np.nanmax(all_z)

    # Prepare figure/axes
    fig = plt.figure(figsize=(10, 6))
    # create plot with proper spacing for slider
    ax = fig.add_subplot(111, projection='3d', position=[0.1, 0.1, 0.8, 0.85])
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Fix the axis limits so that all points are in sight
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    frames_count = len(df_zed_in_vicon)

    # Plot storage
    line_zed, = ax.plot(zed_x, zed_y, zed_z, color='blue', label='ZED Trajectory', visible=True, alpha=0.2, linestyle='dashed')
    line_zed_marker, = ax.plot([], [], [], color='orange', label='ZED-Marker Trajectory', visible=True, alpha=0.2, linestyle='dashed')
    scatter_start = ax.scatter([], [], [], color='green', marker='o', label='_nolegend_', visible=False)
    scatter_end = ax.scatter([], [], [], color='black', marker='o', label='_nolegend_', visible=False)

    scatter_zed_marker = ax.scatter([],[],[], color='orange', marker='^', label='ZED-Marker', alpha=1,s=50, visible=False)

    scatter_markers = []
    if mocap_marker_interpolated is not None:
        for marker in mocap_marker_interpolated:
            # sc = ax.scatter([], [], [], label=marker.name, s=20, alpha=0.6)
            sc = ax.scatter([], [], [], label='_nolegend_', s=20, alpha=1, color='black')
            scatter_markers.append(sc)
        scatter_markers[0]._label = 'Mocap Markers'

    line_color = 'gray'  # Default color for lines
    line_width = 1.0  # Default line width
    line_style = ':'  # Default line style

    # Create lines for the various connections
    line1, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    line2, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    line3, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    line4, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    line5, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    line6, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    # line7, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    # line8, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    # line9, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    # line10, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    # line11, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)
    # line12, = ax.plot([], [], [], color=line_color, linewidth=line_width, label='_nolegend_', linestyle=line_style)


    # Create three lines to represent the ZED local axes
    zed_axis_x, = ax.plot([], [], [], color='red', linewidth=2, label='ZED X-axis')
    zed_axis_y, = ax.plot([], [], [], color='green', linewidth=2, label='ZED Y-axis')
    zed_axis_z, = ax.plot([], [], [], color='blue', linewidth=2, label='ZED Z-axis')
    # Create three lines to represent the Rigid-Body local axes
    rb_axis_x, = ax.plot([], [], [], color='orange', linewidth=2, label='RB X-axis')
    rb_axis_y, = ax.plot([], [], [], color='brown', linewidth=2, label='RB Y-axis')
    rb_axis_z, = ax.plot([], [], [], color='purple', linewidth=2, label='RB Z-axis')
    # Create three lines to represent the ZED as marker local axes
    zed_marker_axis_x, = ax.plot([], [], [], color='cyan', linewidth=2, label='ZED Marker X-axis')
    zed_marker_axis_y, = ax.plot([], [], [], color='magenta', linewidth=2, label='ZED Marker Y-axis')
    zed_marker_axis_z, = ax.plot([], [], [], color='yellow', linewidth=2, label='ZED Marker Z-axis')

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    def update_local_axes(i):
        # ZED local axes (existing code)
        length = 50.0
        r_vec = [
            df_zed_in_vicon["Rotation_X"].iloc[i],
            df_zed_in_vicon["Rotation_Y"].iloc[i],
            df_zed_in_vicon["Rotation_Z"].iloc[i],
        ]
        t_vec = np.array([
            df_zed_in_vicon["Translation_X"].iloc[i],
            df_zed_in_vicon["Translation_Y"].iloc[i],
            df_zed_in_vicon["Translation_Z"].iloc[i],
        ])
        R_mat = Rotation.from_rotvec(r_vec).as_matrix()
        x_end = t_vec + R_mat[:, 0] * length
        y_end = t_vec + R_mat[:, 1] * length
        z_end = t_vec + R_mat[:, 2] * length
        zed_axis_x.set_data_3d([t_vec[0], x_end[0]], [t_vec[1], x_end[1]], [t_vec[2], x_end[2]])
        zed_axis_y.set_data_3d([t_vec[0], y_end[0]], [t_vec[1], y_end[1]], [t_vec[2], y_end[2]])
        zed_axis_z.set_data_3d([t_vec[0], z_end[0]], [t_vec[1], z_end[1]], [t_vec[2], z_end[2]])

        # length = length / 2
        # Rigid-Body local axes (new code)
        rb_r_vec = [
            df_rb_interpolated["Rotation_X"].iloc[i],
            df_rb_interpolated["Rotation_Y"].iloc[i],
            df_rb_interpolated["Rotation_Z"].iloc[i],
        ]
        rb_t_vec = np.array([
            df_rb_interpolated["Translation_X"].iloc[i],
            df_rb_interpolated["Translation_Y"].iloc[i],
            df_rb_interpolated["Translation_Z"].iloc[i],
        ])
        rb_R_mat = Rotation.from_rotvec(rb_r_vec).as_matrix()
        rb_x_end = rb_t_vec + rb_R_mat[:, 0] * length
        rb_y_end = rb_t_vec + rb_R_mat[:, 1] * length
        rb_z_end = rb_t_vec + rb_R_mat[:, 2] * length
        rb_axis_x.set_data_3d([rb_t_vec[0], rb_x_end[0]], [rb_t_vec[1], rb_x_end[1]], [rb_t_vec[2], rb_x_end[2]])
        rb_axis_y.set_data_3d([rb_t_vec[0], rb_y_end[0]], [rb_t_vec[1], rb_y_end[1]], [rb_t_vec[2], rb_y_end[2]])
        rb_axis_z.set_data_3d([rb_t_vec[0], rb_z_end[0]], [rb_t_vec[1], rb_z_end[1]], [rb_t_vec[2], rb_z_end[2]])

        # ZED as marker local axes (new code)
        # length = 50
        zed_marker_r_vec = [
            df_zed_as_marker["Rotation_X"].iloc[i],
            df_zed_as_marker["Rotation_Y"].iloc[i],
            df_zed_as_marker["Rotation_Z"].iloc[i],
        ]
        zed_marker_t_vec = np.array([
            T_zed_as_marker[i][0, 3],  
            T_zed_as_marker[i][1, 3],  
            T_zed_as_marker[i][2, 3],  
        ])
        zed_marker_R_mat = Rotation.from_rotvec(zed_marker_r_vec).as_matrix()
        zed_marker_x_end = zed_marker_t_vec + zed_marker_R_mat[:, 0] * length
        zed_marker_y_end = zed_marker_t_vec + zed_marker_R_mat[:, 1] * length
        zed_marker_z_end = zed_marker_t_vec + zed_marker_R_mat[:, 2] * length
        zed_marker_axis_x.set_data_3d([zed_marker_t_vec[0], zed_marker_x_end[0]], [zed_marker_t_vec[1], zed_marker_x_end[1]], [zed_marker_t_vec[2], zed_marker_x_end[2]])
        zed_marker_axis_y.set_data_3d([zed_marker_t_vec[0], zed_marker_y_end[0]], [zed_marker_t_vec[1], zed_marker_y_end[1]], [zed_marker_t_vec[2], zed_marker_y_end[2]])
        zed_marker_axis_z.set_data_3d([zed_marker_t_vec[0], zed_marker_z_end[0]], [zed_marker_t_vec[1], zed_marker_z_end[1]], [zed_marker_t_vec[2], zed_marker_z_end[2]])

    def update(index):
        i = int(index)
        # Clear lines/scatters
        line_zed.set_data_3d([], [], [])
        scatter_start._offsets3d = ([], [], [])
        scatter_end._offsets3d = ([], [], [])
        scatter_zed_marker._offsets3d = ([], [], [])

        # get all positions of zed marker
        zed_marker_x = [T_zed_as_marker[j][0,3] for j in range(len(T_zed_as_marker))]
        zed_marker_y = [T_zed_as_marker[j][1,3] for j in range(len(T_zed_as_marker))]
        zed_marker_z = [T_zed_as_marker[j][2,3] for j in range(len(T_zed_as_marker))]

        
        # zed_marker_x = T_zed_as_marker[i][0,3] # X component in mm
        # zed_marker_y = T_zed_as_marker[i][1,3] # Y component in mm
        # zed_marker_z = T_zed_as_marker[i][2,3] # Z component in mm
        scatter_zed_marker._offsets3d = ([zed_marker_x[i]], [zed_marker_y[i]], [zed_marker_z[i]])

        # # Plot only the single frame i
        # x = zed_x[i]
        # y = zed_y[i]
        # z = zed_z[i]
        # Plot all frames
        x = zed_x
        y = zed_y
        z = zed_z
        line_zed.set_data_3d(x, y, z)
        line_zed_marker.set_data_3d(zed_marker_x, zed_marker_y, zed_marker_z)
        # scatter_start._offsets3d = ([x], [y], [z])
        # scatter_end._offsets3d = ([x], [y], [z])

        # print distance from zed to zed_marker
        dist_to_marker = np.sqrt((x[i] - zed_marker_x[i]) ** 2 + (y[i] - zed_marker_y[i]) ** 2 + (z[i] - zed_marker_z[i]) ** 2)
        # print(f"\r\033[FDistance from ZED to ZED marker (projected) at frame {i}: {dist_to_marker:.5f} mm")
        print(f"\rDistance from ZED to ZED marker (projected) at frame {i}: {dist_to_marker:.5f} mm", end="")

        
        update_local_axes(i)

        # Markers (scatter)
        if mocap_marker_interpolated is not None:
            sc_centroid = [0, 0, 0]
            sc_centroid[0] = np.mean([marker.X[i] for marker in mocap_marker_interpolated]) 
            sc_centroid[1] = np.mean([marker.Y[i] for marker in mocap_marker_interpolated])
            sc_centroid[2] = np.mean([marker.Z[i] for marker in mocap_marker_interpolated])

            # distance in mm
            sc_dist_to_zed_marker = np.sqrt(
                (sc_centroid[0] - zed_marker_x) ** 2 +
                (sc_centroid[1] - zed_marker_y) ** 2 + 
                (sc_centroid[2] - zed_marker_z) ** 2
            ) # distance in mm
            
            # print(f"\rDistance from current markers to ZED marker (projected) at frame {i}: {sc_dist_to_zed_marker:.5f} mm", end="") # used as control

            for sc, marker in zip(scatter_markers, mocap_marker_interpolated):
                mx = marker.X[i] 
                my = marker.Y[i] 
                mz = marker.Z[i] 
                sc._offsets3d = ([mx], [my], [mz])
           
            # Ensure we have enough markers for lines 1, 2, 3
            if len(mocap_marker_interpolated) >=4: 
                ### OLD lines 1, 2, 3 logic commented out
                # # 1) Line1: last two markers
                # mA = mocap_marker_interpolated[-2]
                # mB = mocap_marker_interpolated[-1]
                # A = np.array([mA.X[i], mA.Y[i], mA.Z[i]])
                # B = np.array([mB.X[i], mB.Y[i], mB.Z[i]])
                # line1.set_data_3d([A[0], B[0]], [A[1], B[1]], [A[2], B[2]])

                # # 2) Line2: midpoint of last two markers -> marker #1
                # midpoint = (A + B) / 2.0
                # m1 = mocap_marker_interpolated[1]
                # Cam1 = np.array([m1.X[i], m1.Y[i], m1.Z[i]])
                # line2.set_data_3d([midpoint[0], Cam1[0]],
                #                   [midpoint[1], Cam1[1]],
                #                   [midpoint[2], Cam1[2]])

                # # 3) Line3: from marker #4 a perpendicular to line2
                # m4 = mocap_marker_interpolated[4]
                # Cam5 = np.array([m4.X[i], m4.Y[i], m4.Z[i]])

                # # Parametric line2: L(t) = midpoint + t*(Cam1 - midpoint)
                # v = Cam1 - midpoint
                # w = midpoint - Cam5
                # # Solve t for perpendicular ( (Cam5->P) dot v = 0 )
                # # P = midpoint + t * v, with (P - Cam5) dot v = 0
                # # => (midpoint - Cam5 + t*v) dot v = 0 => t = -(w·v)/(v·v)
                # denom = np.dot(v, v)
                # if abs(denom) > 1e-12:    
                #     t_val = -np.dot(w, v) / denom
                #     P = midpoint + t_val * v
                #     line3.set_data_3d([Cam5[0], P[0]],
                #                       [Cam5[1], P[1]],
                #                       [Cam5[2], P[2]])
                # else:
                #     line3.set_data_3d([], [], [])
                # NEW lines 1, 2, 3 logic (ninja turtle)
                m_top, m_bl, m_br, m_fr, m_fl  = mocap_marker_interpolated[0:5] # assuming order: top, bottom-left, bottom-right, front-right, front-left 
                # line1: #fl -> #br
                line1.set_data_3d([m_fl.X[i], m_br.X[i]], [m_fl.Y[i], m_br.Y[i]], [m_fl.Z[i], m_br.Z[i]])
                # line2: #bl -> #fr
                line2.set_data_3d([m_bl.X[i], m_fr.X[i]], [m_bl.Y[i], m_fr.Y[i]], [m_bl.Z[i], m_fr.Z[i]])
                # line3: from top marker to every other marker
                line3.set_data_3d([m_top.X[i], m_bl.X[i]], [m_top.Y[i], m_bl.Y[i]], [m_top.Z[i], m_bl.Z[i]])
                line4.set_data_3d([m_top.X[i], m_br.X[i]], [m_top.Y[i], m_br.Y[i]], [m_top.Z[i], m_br.Z[i]])
                line5.set_data_3d([m_top.X[i], m_fr.X[i]], [m_top.Y[i], m_fr.Y[i]], [m_top.Z[i], m_fr.Z[i]])
                line6.set_data_3d([m_top.X[i], m_fl.X[i]], [m_top.Y[i], m_fl.Y[i]], [m_top.Z[i], m_fl.Z[i]])

            else:
                line1.set_data_3d([], [], [])
                line2.set_data_3d([], [], [])
                line3.set_data_3d([], [], [])
        
        # OLD logic for lines 4 through 10 commented out
        # Add lines 4 through 10
        # if mocap_marker_interpolated is not None and len(mocap_marker_interpolated) >= 5:
        #     m0, m1, m2, m3, m4 = mocap_marker_interpolated[0:5]
        #     A0 = np.array([m0.X[i], m0.Y[i], m0.Z[i]])
        #     A1 = np.array([m1.X[i], m1.Y[i], m1.Z[i]])
        #     A2 = np.array([m2.X[i], m2.Y[i], m2.Z[i]])
        #     A3 = np.array([m3.X[i], m3.Y[i], m3.Z[i]])
        #     A4 = np.array([m4.X[i], m4.Y[i], m4.Z[i]])

        #     # line4: #4->#0
        #     line4.set_data_3d([A4[0], A0[0]], [A4[1], A0[1]], [A4[2], A0[2]])
        #     # line5: #2->#3
        #     line5.set_data_3d([A2[0], A3[0]], [A2[1], A3[1]], [A2[2], A3[2]])
        #     # line6: #4->#2
        #     line6.set_data_3d([A4[0], A2[0]], [A4[1], A2[1]], [A4[2], A2[2]])
        #     # line7: #1->#0
        #     line7.set_data_3d([A1[0], A0[0]], [A1[1], A0[1]], [A1[2], A0[2]])
        #     # line8: #1->#2
        #     line8.set_data_3d([A1[0], A2[0]], [A1[1], A2[1]], [A1[2], A2[2]])
        #     # line9: #1->#3
        #     line9.set_data_3d([A1[0], A3[0]], [A1[1], A3[1]], [A1[2], A3[2]])
            
        #     # line10: #1->#4
        #     line10.set_data_3d([A1[0], A4[0]], [A1[1], A4[1]], [A1[2], A4[2]])  
        #     # line11: #3->#4
        #     line11.set_data_3d([A3[0], A4[0]], [A3[1], A4[1]], [A3[2], A4[2]])  
        #     # line12: #0->#3
        #     line12.set_data_3d([A0[0], A3[0]], [A0[1], A3[1]], [A0[2], A3[2]])
        # else:
        #     # Clear all if not enough markers
        #     line4.set_data_3d([], [], [])
        #     line5.set_data_3d([], [], [])
        #     line6.set_data_3d([], [], [])
        #     line7.set_data_3d([], [], [])
        #     line8.set_data_3d([], [], [])
        #     line9.set_data_3d([], [], [])
        #     line10.set_data_3d([], [], [])  # ADDED

        # Maintain equal aspect ratio through updates
        ax.set_box_aspect((1, 1, 1))
        fig.canvas.draw_idle()

    # Slider axis
    slider_ax = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(slider_ax, 'Frame', 0, frames_count - 1, valinit=0, valstep=1)
    frame_slider.on_changed(update)

    # Initial update
    update(0)

    # Set equal aspect ratio AFTER slider creation to preserve layout
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    
    plt.show(block=True)
    return fig


def plot_zed_orientation(csv_path):
    """
    Load a <xxx>_results.csv file and plot the Rotation_X/Y/Z components over frames.
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    for col in ("Frame","Timestamp", "Rotation_X", "Rotation_Y", "Rotation_Z"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {csv_path}")

    frames = df["Frame"]
    rx = df["Rotation_X"]
    ry = df["Rotation_Y"]
    rz = df["Rotation_Z"]
    time_stamps = df["Timestamp"] # assuming timestamps are in ms
    # convert time stamps to ms from t0, uwhere t0 is the first time stamp (reference point)
    time = time_stamps-time_stamps.iloc[0]



    # Plot
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(time, rx, color="r", label="Rotation_X")
    axes[1].plot(time, ry, color="g", label="Rotation_Y")
    axes[2].plot(time, rz, color="b", label="Rotation_Z")

    axes[0].set_ylabel("Rot X (rad)")
    axes[1].set_ylabel("Rot Y (rad)")
    axes[2].set_ylabel("Rot Z (rad)")
    axes[2].set_xlabel("Time (ms)")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    return plt



def plot_vicon_orientations(df_mocap):
    """
    Plot the Rotation_X/Y/Z components over time from a MocapData object (VICON or similar).
    """
    time = df_mocap.Time_ms  # Assuming Time_ms is in milliseconds
    rx = df_mocap.Rotation_X
    ry = df_mocap.Rotation_Y
    rz = df_mocap.Rotation_Z

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(time, rx, color="r", label="Rotation_X")
    axes[1].plot(time, ry, color="g", label="Rotation_Y")
    axes[2].plot(time, rz, color="b", label="Rotation_Z")

    axes[0].set_ylabel("Rot X (rad)")
    axes[1].set_ylabel("Rot Y (rad)")
    axes[2].set_ylabel("Rot Z (rad)")
    axes[2].set_xlabel("Time (ms)")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    try:
        plt.tight_layout()
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)

    return plt

# import numpy as np
# from scipy.spatial.transform import Rotation

def solve_hand_eye(A_list, B_list):
    """
    Solves AX = XB using rotation and translation separation (Tsai-Lenz).
    Inputs:
        A_list, B_list: lists of 4x4 relative transforms (must be ≥ 2)
    Returns:
        X: 4x4 transformation matrix
    """
    assert len(A_list) == len(B_list)

    n = len(A_list)
    R_As = [A[:3, :3] for A in A_list]
    R_Bs = [B[:3, :3] for B in B_list]

    log_As = [Rotation.from_matrix(R).as_rotvec() for R in R_As]
    log_Bs = [Rotation.from_matrix(R).as_rotvec() for R in R_Bs]

    M = np.zeros((3, 3))
    for a, b in zip(log_As, log_Bs):
        M += np.outer(b, a)

    U, _, Vt = np.linalg.svd(M)
    R_X = U @ Vt
    if np.linalg.det(R_X) < 0:
        Vt[-1, :] *= -1
        R_X = U @ Vt

    # Solve translation part
    # Solve t_X from overdetermined system: (I - R_A) t_X = t_A - R_X @ t_B
    A_blocks = []
    b_blocks = []
    t_As = [A[:3, 3] for A in A_list]
    t_Bs = [B[:3, 3] for B in B_list]
    for R_A, t_A, R_B, t_B in zip(R_As, t_As, R_Bs, t_Bs):
        A_i = np.eye(3) - R_A
        b_i = t_A - R_X @ t_B
        A_blocks.append(A_i)
        b_blocks.append(b_i)

    A_stack = np.vstack(A_blocks)
    b_stack = np.hstack(b_blocks)
    t_X, _, _, _ = np.linalg.lstsq(A_stack, b_stack, rcond=None)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


def estimate_transform_from_vicon_to_zed(T_rb_list, T_zed_list):
    # Compute relative transforms
    A_list = []
    B_list = []
    for i in range(1, len(T_rb_list)):
        A = np.linalg.inv(T_rb_list[i-1]) @ T_rb_list[i]
        B = np.linalg.inv(T_zed_list[i-1]) @ T_zed_list[i]
        A_list.append(A)
        B_list.append(B)

    # Step 1: Estimate T_rb_to_zed_global using AX = XB
    T_rb_to_zed = solve_hand_eye(A_list, B_list)

    # Step 2: Estimate T_cam_marker using transformed rb poses
    T_markers_in_zed = [T_rb_to_zed @ T_v for T_v in T_rb_list]

    T_cam_marker_list = [
        np.linalg.inv(T_m) @ T_c for T_m, T_c in zip(T_markers_in_zed, T_zed_list)
    ]

    # Average rotation and translation
    R_list = [T[:3, :3] for T in T_cam_marker_list]
    t_list = [T[:3, 3] for T in T_cam_marker_list]

    def average_rotations(rotations):
        quats = [Rotation.from_matrix(R).as_quat() for R in rotations]
        q_avg = np.mean(quats, axis=0)
        q_avg /= np.linalg.norm(q_avg)
        return Rotation.from_quat(q_avg).as_matrix()

    R_avg = average_rotations(R_list)
    t_avg = np.mean(t_list, axis=0)

    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = R_avg
    T_cam_marker[:3, 3] = t_avg

    return T_rb_to_zed, T_cam_marker

def plot_velocity(df_zed, df_zed_as_marker, loop_closed_indices=None, title="Camera Frame Velocity in VICON Frame of Reference"):
    """
    Plot the velocity of ZED in the VICON frame of reference.
    Assumes df_zed and df_rb have 'Translation_X', 'Translation_Y', 'Translation_Z' columns.
    """
    from scipy.spatial.transform import Rotation
    print("title: ", title)
    # Compute velocity as the difference between consecutive positions
    dt = np.mean(np.diff(df_zed['Time_ms'].values)) / 1000.0  # Convert ms to seconds
    vel_zed = np.sqrt(
        np.diff(df_zed['Translation_X'], prepend=df_zed['Translation_X'].iloc[0])**2 +
        np.diff(df_zed['Translation_Y'], prepend=df_zed['Translation_Y'].iloc[0])**2 +
        np.diff(df_zed['Translation_Z'], prepend=df_zed['Translation_Z'].iloc[0])**2
    ) / dt  # mm/s

    vel_rb = np.sqrt(
        np.diff(df_zed_as_marker['Translation_X'], prepend=df_zed_as_marker['Translation_X'].iloc[0])**2 +
        np.diff(df_zed_as_marker['Translation_Y'], prepend=df_zed_as_marker['Translation_Y'].iloc[0])**2 +
        np.diff(df_zed_as_marker['Translation_Z'], prepend=df_zed_as_marker['Translation_Z'].iloc[0])**2
    ) / dt  # mm/s

    time_zed = df_zed['Time_ms'] / 1000.0  # Convert ms to seconds
    time_zed_as_marker = df_zed_as_marker['Time_ms'] / 1000.0  # Convert ms to seconds

    plt.figure(figsize=(10, 5))
    plt.plot(time_zed, vel_zed, label='${}^{V}\mathbf{T}_{W}\,{}^{W}\mathbf{T}_{C}(t)$', color='blue')
    plt.plot(time_zed_as_marker, vel_rb, label='${}^{V}\mathbf{T}_{M}(t)\,{}^{M}\mathbf{T}_{C}$', color='orange', linestyle='--')
    # Mark loop closure events if provided
    if loop_closed_indices is not None:
        for idx in loop_closed_indices:
            if 0 <= idx < len(time_zed):
                plt.axvline(x=time_zed.iloc[idx], color='red', linestyle=':', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')
    plt.legend()
    plt.grid(True)
    try:
        plt.tight_layout()
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)


def replace_stagnant_poses_with_linear_interpolation(df, velocity_threshold=1e-3):
    """
    Identify segments where pose remains nearly static and then abruptly jumps,
    and replace those stagnant frames with linearly-interpolated values.
    Then recalculate the Rotation_norm column.
    """
    # 1. Compute frame-to-frame velocities (magnitude of translation deltas).
    translations = df[["Translation_X", "Translation_Y", "Translation_Z"]].to_numpy()
    # Compute velocities as the norm of the difference between consecutive translations
    velocities = np.linalg.norm(np.diff(translations, axis=0), axis=1)

    
    # Determine thresholds for spike and static detection
    std_temp = np.std(velocities)
    # calculate std of non-zero velocities
    std_velocities = np.std(velocities[velocities > std_temp* 0.1])
    spike_threshold = 3 * std_velocities  # Define a threshold for spikes
    static_threshold = 0.00001 * std_velocities  # Define a threshold for static segments



    # 2. Detect start (low-velocity) and end (high-velocity) of each stagnant segment.
    stagnant_segments = []
    loop_closure_ind = []
    stag_start_indx = []
    
    # loop in reverse to find segments from the end
    for i in range(len(velocities) - 1, 1, -1):
        if velocities[i] > static_threshold and velocities[i-1] < static_threshold:
            # We found a spike after a static segment
            loop_closure_ind.append(i+1)  # Store the index of the spike
            while i > 0 and velocities[i-1] < static_threshold:
                # Move backwards to find the start of the stagnant segment
                i -= 1
            # Now i is the last index of the stagnant segment
            stag_start_indx.append(i)  # Store the start index of the stagnant segment

    # 3. Apply the linear interpolation
    for i_start, i_end in zip(reversed(stag_start_indx), reversed(loop_closure_ind)):
        if i_end-i_start < 2:  # Ensure valid segment
            # print error
            print(f"Invalid stagnant segment from {i_start} to {i_end}")
            continue
        
        # boundary times
        t0 = df.at[i_start, "Time_ms"]
        t1 = df.at[i_end, "Time_ms"]

        cols = [
            "Translation_X", "Translation_Y", "Translation_Z",
            "Rotation_X",    "Rotation_Y",    "Rotation_Z"
        ]
        # values at the boundaries
        v0 = df.loc[i_start, cols].to_numpy()
        v2 = df.loc[i_end,  cols].to_numpy()

        # indices to correct: from i_start+1 up to i_end
        idx = list(range(i_start + 1, i_end))
        t  = df.loc[idx, "Time_ms"].to_numpy()
        # normalized time between boundaries
        alpha = (t - t0) / (t1 - t0)

        # linear interpolation per column
        df.loc[idx, cols] = v0 + alpha[:, None] * (v2 - v0)
        stagnant_segments.append(f"[{i_start}-{i_end}]")

    # Recalculate Rotation_norm column
    df["Rotation_norm"] = absolute_rotation_angle(df)   
    print(f"Stagnant segments replaced with linear interpolation at indices: [%s]" % ", ".join(stagnant_segments))
    return df

def remove_stagnant_poses(df, velocity_threshold=1e-3):
    """
    Identify segments where pose remains nearly static and then abruptly jumps,
    and remove those stagnant frames from the DataFrame.
    """
    translations = df[["Translation_X", "Translation_Y", "Translation_Z"]].to_numpy()
    velocities = np.linalg.norm(np.diff(translations, axis=0), axis=1)

    std_temp = np.std(velocities)
    std_velocities = np.std(velocities[velocities > std_temp * 0.1])
    static_threshold = 0.00001 * std_velocities
    stag_segments_index = []
    loop_closure_ind = []
    stag_start_indx = []
    for i in range(len(velocities) - 1, 1, -1):
        if velocities[i] > static_threshold and velocities[i-1] < static_threshold:
            loop_closure_ind.append(i+1)
            while i > 0 and velocities[i-1] < static_threshold:
                i -= 1
            stag_start_indx.append(i)

    for i_start, i_end in zip(reversed(stag_start_indx), reversed(loop_closure_ind)):
        if i_end - i_start < 2:
            print(f"Invalid stagnant segment from {i_start} to {i_end}")
            continue
        df.drop(index=range(i_start + 1, i_end), inplace=True)
        stag_segments_index.append(f"[{i_start}-{i_end}]")
        
    print(f"Stagnant segments removed at indices: [%s]" % ", ".join(stag_segments_index))
    df.reset_index(drop=True, inplace=True)
    return df

# helper to extract rotations & translations from a list of 4×4 mats
def split_rt(mats):
    t_list, R_list = [], []
    for T in mats:
        t_list.append(T[:3, 3])
        R_list.append(T[:3, :3])
    return t_list, R_list

def average_pose(transforms):
        """
        Averages a list of 4x4 transformation matrices."""
        quats = []
        trans = []
        for T in transforms:
            r = Rotation.from_matrix(T[:3, :3]).as_quat()
            quats.append(r)
            trans.append(T[:3, 3])
        quats = np.array(quats)
        trans = np.array(trans)
        # average translation
        t_mean = np.mean(trans, axis=0)
        # average rotation
        q_mean = np.mean(quats, axis=0)
        q_mean /= np.linalg.norm(q_mean)
        R_mean = Rotation.from_quat(q_mean).as_matrix()
        T_mean = np.eye(4)
        T_mean[:3, :3] = R_mean
        T_mean[:3, 3] = t_mean
        return T_mean

def plot_xyz_translation(t):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot([t[0] for t in t], label='X')
    ax.plot([t[1] for t in t], label='Y')
    ax.plot([t[2] for t in t], label='Z')
    ax.plot([np.linalg.norm(t) for t in t], label='Norm', linestyle='-', color='gray', alpha=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Translation (mm)')
    ax.set_title('ZED in ZED-marker Frame of Reference - X, Y, Z Translation')
    ax.legend()
    try:
        plt.tight_layout()
    except (AttributeError, ValueError):
        pass
    plt.show()



def plot_df_to_df_angles(df1, df2, title="Angle between two sets of orientations over time"):
    """
    Compute and plot the angle between two sets of orientations (given as DataFrames with Rotation_X/Y/Z columns) over time.
    """
    time = df1['Time_ms'] / 1000.0  # Convert ms to seconds
    angles = []
    for i in range(len(df1)):
        R1 = Rotation.from_rotvec([df1.at[i, 'Rotation_X'], df1.at[i, 'Rotation_Y'], df1.at[i, 'Rotation_Z']]).as_matrix()
        R2 = Rotation.from_rotvec([df2.at[i, 'Rotation_X'], df2.at[i, 'Rotation_Y'], df2.at[i, 'Rotation_Z']]).as_matrix()
        R_diff = R1.T @ R2
        angle = Rotation.from_matrix(R_diff).magnitude()  # Angle in radians
        angles.append(np.degrees(angle))  # Convert to degrees

    plt.figure(figsize=(10, 5))
    plt.plot(time, angles, label='Angle between orientations (degrees)', color='purple')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    try:
        plt.tight_layout()
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)
    return angles


def plot_markers_and_zed_in_marker_frame(mocap_markers, T_zed2rb, marker_coordinate_opt=2, verbose=True):
    """
    Visualize markers and ZED camera frame in marker-intrinsic coordinate system.
    
    Args:
        mocap_markers: List of Marker objects with X, Y, Z position arrays
        T_zed2rb: 4x4 transformation matrix from ZED to rigid body frame
        marker_coordinate_opt: Option for coordinate system definition (0, 1, or 2)
        verbose: If True, print debug information
    
    Returns:
        dict: Dictionary containing computed transformation matrices and coordinate system info
            - 'T_zed2marker_intrinsic': Transformation from ZED to marker-intrinsic frame
            - 'marker_centroid': Centroid of markers in VICON frame
            - 'x_axis': X-axis direction in VICON frame
            - 'y_axis': Y-axis direction in VICON frame
            - 'z_axis': Z-axis direction in VICON frame
    """
    
    # Extract marker positions at first frame
    m_top_pos = np.array([mocap_markers[0].X[0], mocap_markers[0].Y[0], mocap_markers[0].Z[0]])
    m_bl_pos = np.array([mocap_markers[1].X[0], mocap_markers[1].Y[0], mocap_markers[1].Z[0]])
    m_br_pos = np.array([mocap_markers[2].X[0], mocap_markers[2].Y[0], mocap_markers[2].Z[0]])
    m_fr_pos = np.array([mocap_markers[3].X[0], mocap_markers[3].Y[0], mocap_markers[3].Z[0]])
    m_fl_pos = np.array([mocap_markers[4].X[0], mocap_markers[4].Y[0], mocap_markers[4].Z[0]])
    
    # Origin: marker centroid
    marker_centroid = np.mean([m_top_pos, m_bl_pos, m_br_pos, m_fr_pos, m_fl_pos], axis=0)

    # Compute axes based on selected option
    match marker_coordinate_opt:
        case 0:  # uses 4 markers directly (2 diagonals)
            # X-axis: diagonal line from back-left (m_bl) to front-right (m_fr)
            x_vec = m_fr_pos - m_bl_pos
            x_axis = x_vec / np.linalg.norm(x_vec)
            
            # Z-axis: X cross with diagonal line from back-right (m_br) to front-left (m_fl)
            diag_line = m_fl_pos - m_br_pos
            z_vec = np.cross(x_axis, diag_line)
            z_axis = z_vec / np.linalg.norm(z_vec)
            
            # Y-axis: cross(Z, X)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
        case 1:  # uses 3 markers directly (top, front-left, front-right)
            # Z-axis: from marker centroid to top marker (m_top)
            z_vec = m_top_pos - marker_centroid
            z_axis = z_vec / np.linalg.norm(z_vec)
            # front line: from fl to fr
            front_line = m_fr_pos - m_fl_pos    
            # Y-axis: cross(front line, Z)
            y_vec = np.cross(front_line, z_axis)
            y_axis = y_vec / np.linalg.norm(y_vec)
            # X-axis: cross(Y, Z)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
        case 2:  # uses 4 markers (back line midpoint to front line midpoint)
            # Y-axis: from middle of back line (m_bl, m_br) to middle of front line (m_fl, m_fr) - faces forward
            back_midpoint = (m_bl_pos + m_br_pos) / 2
            front_midpoint = (m_fl_pos + m_fr_pos) / 2
            y_vec = front_midpoint - back_midpoint
            y_axis = y_vec / np.linalg.norm(y_vec)
            
            # X-axis: from back-left to back-right (right direction)
            x_vec = m_br_pos - m_bl_pos
            x_axis = x_vec / np.linalg.norm(x_vec)
            
            # Z-axis: cross(X, Y)
            z_vec = np.cross(x_axis, y_axis)
            z_axis = z_vec / np.linalg.norm(z_vec)
    
    # Construct rotation from RB/VICON to marker-intrinsic frame (pure rotation, no translation)
    # Both RB and marker-intrinsic frames are centered at marker centroid
    R_vicon2marker_intrinsic = np.column_stack([x_axis, y_axis, z_axis]).T
    R_rb2marker_intrinsic = R_vicon2marker_intrinsic  # RB frame is aligned with VICON at first frame
    T_rb2marker_intrinsic = np.vstack((np.hstack((R_rb2marker_intrinsic, np.zeros((3, 1)))), [0, 0, 0, 1]))
    
    # Transform calibration result to marker-intrinsic frame
    T_zed2marker_intrinsic = T_rb2marker_intrinsic @ T_zed2rb
    
    if verbose:
        print("Marker-intrinsic coordinate system computed:")
        print(f"Marker centroid (VICON frame): {marker_centroid}")
        print(f"X-axis direction (VICON frame): {x_axis}")
        print(f"Y-axis direction (VICON frame): {y_axis}")
        print(f"Z-axis direction (VICON frame): {z_axis}")
        print(f"T_zed2marker_intrinsic:\n{T_zed2marker_intrinsic}")
    
    # Create visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Transform markers to marker-intrinsic frame
    T_vicon2marker_intrinsic_full = np.vstack((np.hstack((R_vicon2marker_intrinsic, -R_vicon2marker_intrinsic @ marker_centroid.reshape(3, 1))), [0, 0, 0, 1]))
    
    # Helper function to create sphere mesh
    def create_sphere(center, radius, u_res=20, v_res=20):
        u = np.linspace(0, 2 * np.pi, u_res)
        v = np.linspace(0, np.pi, v_res)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        return x, y, z
    
    # Helper function to create cylinder mesh
    def create_cylinder(p1, p2, radius, u_res=20):
        # Vector from p1 to p2
        v = p2 - p1
        length = np.linalg.norm(v)
        if length < 1e-6:
            return None, None, None
        
        # Create orthonormal basis
        v_norm = v / length
        # Find perpendicular vector
        if abs(v_norm[0]) < 0.9:
            perp1 = np.array([1, 0, 0])
        else:
            perp1 = np.array([0, 1, 0])
        perp1 = perp1 - np.dot(perp1, v_norm) * v_norm
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(v_norm, perp1)
        
        # Create cylinder surface
        u = np.linspace(0, 2 * np.pi, u_res)
        h = np.linspace(0, length, 10)
        u_mesh, h_mesh = np.meshgrid(u, h)
        
        x = p1[0] + h_mesh * v_norm[0] + radius * np.cos(u_mesh) * perp1[0] + radius * np.sin(u_mesh) * perp2[0]
        y = p1[1] + h_mesh * v_norm[1] + radius * np.cos(u_mesh) * perp1[1] + radius * np.sin(u_mesh) * perp2[1]
        z = p1[2] + h_mesh * v_norm[2] + radius * np.cos(u_mesh) * perp1[2] + radius * np.sin(u_mesh) * perp2[2]
        
        return x, y, z
    
    marker_positions_intrinsic = []
    marker_names = ['top', 'back-left', 'back-right', 'front-right', 'front-left']
    
    # Collect marker positions in intrinsic frame first
    positions_to_add = [m_top_pos, m_bl_pos, m_br_pos, m_fr_pos, m_fl_pos]
    for pos in positions_to_add:
        pos_homo = np.append(pos, 1)
        pos_intrinsic = (T_vicon2marker_intrinsic_full @ pos_homo)[:3]
        marker_positions_intrinsic.append(pos_intrinsic)
    marker_positions_intrinsic = np.array(marker_positions_intrinsic)
    
    # Plot diagonal cylinders first (so they appear behind markers)
    # m_bl (index 1) to m_fr (index 3)
    x_cyl1, y_cyl1, z_cyl1 = create_cylinder(marker_positions_intrinsic[1], marker_positions_intrinsic[3], 5, u_res=20)
    if x_cyl1 is not None:
        ax.plot_surface(x_cyl1, y_cyl1, z_cyl1, color='gray', alpha=0.7, shade=False)
    
    # m_br (index 2) to m_fl (index 4)
    x_cyl2, y_cyl2, z_cyl2 = create_cylinder(marker_positions_intrinsic[2], marker_positions_intrinsic[4], 5, u_res=20)
    if x_cyl2 is not None:
        ax.plot_surface(x_cyl2, y_cyl2, z_cyl2, color='gray', alpha=0.7, shade=False)
    
    # Find nearest point between the two diagonals
    # Diagonal 1: m_bl to m_fr (indices 1 to 3)
    # Diagonal 2: m_br to m_fl (indices 2 to 4)
    p1 = marker_positions_intrinsic[1]  # m_bl
    d1 = marker_positions_intrinsic[3] - marker_positions_intrinsic[1]  # m_fr - m_bl
    
    p2 = marker_positions_intrinsic[2]  # m_br
    d2 = marker_positions_intrinsic[4] - marker_positions_intrinsic[2]  # m_fl - m_br
    
    # Find closest points between two lines: p1 + t*d1 and p2 + s*d2
    # Using formula for closest point between two lines in 3D
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    denom = a * c - b * b
    if abs(denom) > 1e-10:
        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom
        closest_p1 = p1 + t * d1
        closest_p2 = p2 + s * d2
        nearest_point = (closest_p1 + closest_p2) / 2  # Midpoint between closest points
    else:
        nearest_point = (p1 + p2) / 2  # Fallback to midpoint of line origins
    
    # Plot cylinder from top marker to nearest point
    m_top_intrinsic = marker_positions_intrinsic[0]
    x_cyl_rod, y_cyl_rod, z_cyl_rod = create_cylinder(m_top_intrinsic, nearest_point, 5, u_res=16)
    if x_cyl_rod is not None:
        ax.plot_surface(x_cyl_rod, y_cyl_rod, z_cyl_rod, color='gray', alpha=0.7, shade=False)
    
    # Plot markers as gray spheres
    for i, pos in enumerate(positions_to_add):
        pos_homo = np.append(pos, 1)
        pos_intrinsic = (T_vicon2marker_intrinsic_full @ pos_homo)[:3]
        
        # Create and plot sphere (diameter 25mm = radius 12.5mm)
        x_sphere, y_sphere, z_sphere = create_sphere(pos_intrinsic, 12.5, u_res=15, v_res=15)
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.9, shade=True)
        
        # Add text label
        ax.text(pos_intrinsic[0], pos_intrinsic[1], pos_intrinsic[2], f'  {marker_names[i]}', fontsize=9)
    
    # Get ZED camera origin and axes in marker-intrinsic frame
    zed_origin_intrinsic = (T_zed2marker_intrinsic @ np.array([0, 0, 0, 1]))[:3]
    zed_axes_length = 50  # mm
    zed_x_axis = (T_zed2marker_intrinsic @ np.array([zed_axes_length, 0, 0, 1]))[:3]
    zed_y_axis = (T_zed2marker_intrinsic @ np.array([0, zed_axes_length, 0, 1]))[:3]
    zed_z_axis = (T_zed2marker_intrinsic @ np.array([0, 0, zed_axes_length, 1]))[:3]
    
    # Plot ZED camera origin
    ax.scatter(*zed_origin_intrinsic, s=200, c='blue', marker='s', label='ZED origin')
    
    # Plot ZED axes
    ax.plot([zed_origin_intrinsic[0], zed_x_axis[0]], [zed_origin_intrinsic[1], zed_x_axis[1]], [zed_origin_intrinsic[2], zed_x_axis[2]], 'r-', linewidth=2, label='ZED X')
    ax.plot([zed_origin_intrinsic[0], zed_y_axis[0]], [zed_origin_intrinsic[1], zed_y_axis[1]], [zed_origin_intrinsic[2], zed_y_axis[2]], 'g-', linewidth=2, label='ZED Y')
    ax.plot([zed_origin_intrinsic[0], zed_z_axis[0]], [zed_origin_intrinsic[1], zed_z_axis[1]], [zed_origin_intrinsic[2], zed_z_axis[2]], 'b-', linewidth=2, label='ZED Z')
    
    # Plot marker-intrinsic frame origin and axes
    ax.scatter(0, 0, 0, s=200, c='black', marker='*', label='Marker origin')
    ax.plot([0, 50], [0, 0], [0, 0], 'r--', linewidth=1, alpha=0.5)  # X
    ax.plot([0, 0], [0, 50], [0, 0], 'g--', linewidth=1, alpha=0.5)  # Y
    ax.plot([0, 0], [0, 0], [0, 50], 'b--', linewidth=1, alpha=0.5)  # Z
    
    # Plot circles in ZED focal plane transformed to marker-intrinsic frame
    circle_angles = np.linspace(0, 2*np.pi, 100)
    circle_radius = 11.02  # mm
    
    # Circle 1: center at (0, 0, -20) in ZED frame
    circle1_x = circle_radius * np.cos(circle_angles)
    circle1_y = circle_radius * np.sin(circle_angles)
    circle1_z = -20 * np.ones_like(circle_angles)
    
    circle1_intrinsic = []
    for cx, cy, cz in zip(circle1_x, circle1_y, circle1_z):
        pt_homo = np.array([cx, cy, cz, 1])
        pt_intrinsic = (T_zed2marker_intrinsic @ pt_homo)[:3]
        circle1_intrinsic.append(pt_intrinsic)
    circle1_intrinsic = np.array(circle1_intrinsic)
    ax.plot(circle1_intrinsic[:, 0], circle1_intrinsic[:, 1], circle1_intrinsic[:, 2], 'c-', linewidth=2)
    
    # Circle 2: center at (50, 0, -20) in ZED frame
    circle2_x = 50 + circle_radius * np.cos(circle_angles)
    circle2_y = circle_radius * np.sin(circle_angles)
    circle2_z = -20 * np.ones_like(circle_angles)
    
    circle2_intrinsic = []
    for cx, cy, cz in zip(circle2_x, circle2_y, circle2_z):
        pt_homo = np.array([cx, cy, cz, 1])
        pt_intrinsic = (T_zed2marker_intrinsic @ pt_homo)[:3]
        circle2_intrinsic.append(pt_intrinsic)
    circle2_intrinsic = np.array(circle2_intrinsic)
    ax.plot(circle2_intrinsic[:, 0], circle2_intrinsic[:, 1], circle2_intrinsic[:, 2], 'c-', linewidth=2)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Markers and ZED Camera in Marker-Intrinsic Frame')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show(block=False)
    
    if verbose:
        print(f"Verification plot displayed. ZED origin distance from marker centroid: {np.linalg.norm(zed_origin_intrinsic):.2f} mm")
    
    # Return results
    return {
        'T_zed2marker_intrinsic': T_zed2marker_intrinsic,
        'marker_centroid': marker_centroid,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis
    }


