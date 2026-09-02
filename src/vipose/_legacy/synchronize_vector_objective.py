"""
Drop-in replacement objective for stage-2 temporal synchronisation.

WHY
---
The current objective, synchronize_zed_vicon_trial (SynchronizationNew.py:1003),
minimises the mismatch between angular SPEED MAGNITUDES:

    error = |omega_zed| - |omega_vicon_interp|          # scalars

That discards the rotation axis. It localises the cuVSLAM pipelines' lag to
~3 ms but ZED-SDK's to only ~10 ms, which is what stage 3 was invented to
patch up.

This version matches the angular velocity VECTORS, solving for the unknown
constant rotation between the two frames:

    min_R  sum_t || w_zed(t) - R w_vicon(t + lag) ||^2 ,   R in SO(3)

The inner problem is closed-form (Kabsch), so cost per lag evaluation is one
interpolation plus one 3x3 SVD -- comparable to the current objective.

It remains usable BEFORE frame registration, which is the constraint that
rules out using the rotational residual here: the unknown relative rotation is
SOLVED FOR, not assumed. It just uses three components per sample instead of
one, so it is far better conditioned.

Frame convention: this uses the same SPATIAL (world-frame) increments as
rotational_increments_from_rotvec, i.e. dR = R[1:] * R[:-1].inv(). With that
convention w_zed lives in the SLAM world frame W and w_vicon in the Vicon
frame V, so the two are related by the constant rotation V_R_W -- exactly the
kind of constant R that Kabsch recovers. (Body-frame increments would work
equally well and would recover C_R_M instead.)

HOW TO USE
----------
1. Paste `synchronize_zed_vicon_trial_vector` into SynchronizationNew.py.
2. In plot_path_lib.py, inside process_zed_vicon_calibration, change the
   objective (currently line ~194):

       rms = synchronize_zed_vicon_trial(df_ZED, df_rb, lag,
                                         discard_fraction=config.discard_fraction)
   to
       rms = synchronize_zed_vicon_trial_vector(df_ZED, df_rb, lag,
                                         discard_fraction=config.discard_fraction)

3. Re-run `plot_path_unified.py --tests 2,3,5 --pipelines 1,2,3 --unified`.

WHAT SUCCESS LOOKS LIKE
-----------------------
The test is whether the lag it converges to needs no stage-3 correction.
Re-run stage3_refine.py afterwards: if every cell's stage-3 shift comes back
under ~2 ms (versus up to 11 ms now), the vector objective has removed the
need for stage 3 and the synchronisation section needs no caveat.

`compare_objectives` below prints both objectives' landscapes side by side for
one trial, so you can see the sharpening directly before committing to a
full re-run. Call it from inside your own pipeline where df_ZED and df_rb are
already built correctly.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _omega_vectors(df, t_col="Time_ms", cols=("Rotation_X", "Rotation_Y", "Rotation_Z")):
    """Angular velocity VECTORS (rad/ms), spatial convention, matching
    rotational_increments_from_rotvec. Returns (t_mid, w) with w shape (N-1,3)."""
    df = df.dropna(subset=list(cols))
    t = df[t_col].to_numpy(dtype=float)
    rv = df.loc[:, cols].to_numpy(dtype=float)
    R = Rotation.from_rotvec(rv)
    dR = R[1:] * R[:-1].inv()                      # spatial increments
    drv = dR.as_rotvec()                           # (N-1, 3) radians
    dt = np.diff(t)
    good = dt > 0
    return 0.5 * (t[1:] + t[:-1])[good], drv[good] / dt[good, None]


def synchronize_zed_vicon_trial_vector(df_zed_og, df_rb_og, lag_ms,
                                       discard_fraction=None,
                                       min_speed_deg_s=5.0):
    """Vector-matching temporal-alignment objective. Lower is better.

    Same signature and same lag sign convention as synchronize_zed_vicon_trial:
    the VICON time axis is shifted by +lag_ms to align with the ZED axis.

    min_speed_deg_s gates out near-stationary samples, whose axis direction is
    noise. 5 deg/s is well below the 19-24 deg/s median of these recordings.
    """
    df_zed = df_zed_og.copy()
    df_rb = df_rb_og.copy()

    n = len(df_zed)
    if discard_fraction is None or discard_fraction == 0.0:
        start, end = 0, n
    elif discard_fraction < 0.0:
        start, end = 0, int(n * abs(discard_fraction))
    else:
        start = int(n // (1 / discard_fraction))
        end = n - start
    df_zed = df_zed.iloc[start:end].reset_index(drop=True)

    df_rb.Time_ms = df_rb.Time_ms + lag_ms

    t_z, w_z = _omega_vectors(df_zed)
    t_v, w_v = _omega_vectors(df_rb)
    if len(t_z) < 10 or len(t_v) < 10:
        return np.inf

    w_vi = np.column_stack([np.interp(t_z, t_v, w_v[:, k]) for k in range(3)])

    thr = np.deg2rad(min_speed_deg_s) / 1000.0     # rad/ms
    keep = ((np.linalg.norm(w_z, axis=1) > thr) &
            (np.linalg.norm(w_vi, axis=1) > thr))
    if keep.sum() < 50:
        return np.inf

    A, B = w_z[keep], w_vi[keep]
    U, _, Vt = np.linalg.svd(B.T @ A)              # want A ~ R B
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    resid = A - B @ R.T
    return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def compare_objectives(df_ZED, df_rb, lag_ms, span=40.0, step=2.0,
                       discard_fraction=0.0):
    """Print both objectives vs lag offset, each normalised to its own minimum.

    Call with the df_ZED / df_rb that process_zed_vicon_calibration has already
    built, e.g. just after the stage-2 block. A usable objective shows a clear
    dip near 0 and rises away from it; a flat row means that objective cannot
    localise the lag.
    """
    from SynchronizationNew import synchronize_zed_vicon_trial

    offs = np.arange(-span, span + step / 2, step)
    mag = np.array([synchronize_zed_vicon_trial(df_ZED, df_rb, float(lag_ms + o),
                                                discard_fraction=discard_fraction)
                    for o in offs])
    vec = np.array([synchronize_zed_vicon_trial_vector(df_ZED, df_rb, float(lag_ms + o),
                                                       discard_fraction=discard_fraction)
                    for o in offs])
    print(f"  {'offset[ms]':>11s} {'|w| magnitude':>15s} {'w vector':>12s}")
    for o, m, v in zip(offs, mag, vec):
        print(f"  {o:+11.1f} {m / mag.min():15.4f} {v / vec.min():12.4f}"
              + ("   <- stage-2" if abs(o) < 1e-9 else ""))
    print(f"\n  argmin: magnitude {offs[np.argmin(mag)]:+.1f} ms, "
          f"vector {offs[np.argmin(vec)]:+.1f} ms")
    print(f"  sharpness (value at +/-20 ms / value at min): "
          f"magnitude {mag[np.argmin(np.abs(offs - 20))] / mag.min():.3f}, "
          f"vector {vec[np.argmin(np.abs(offs - 20))] / vec.min():.3f}")
    return offs, mag, vec
