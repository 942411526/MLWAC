"""
WAC.py
======================
Fused waypoint-velocity controller for robot navigation.

"""

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants  (override these to match your robot's limits)
# ---------------------------------------------------------------------------

DT    = 1 / 3       # Time-step duration [s]
EPS   = 1e-6        # Displacement threshold below which the robot is "at target"
MAX_V = 1.0         # Maximum linear  velocity [m/s]
MAX_W = 1.0         # Maximum angular velocity [rad/s]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

def combined_controller(
    waypoints: np.ndarray,
    velocities: np.ndarray,
    dt: float = 1 / 3,
) -> np.ndarray:
    """Compute control commands by fusing waypoint geometry and velocity predictions.

    For each step, the angular velocity ``w`` is selected or blended from two
    sources:

    * **Waypoint-derived** : ``arctan(dy / dx) / DT`` — geometric heading from
      the predicted waypoint offset.
    * **Velocity-predicted**: ``w`` from the model's action output.

    Selection logic
    ---------------
    1. If the waypoint displacement is negligible (both ``|dx|`` and ``|dy|`` < ``EPS``),
       output ``[0, w_predicted]`` (robot is already at the target).
    2. If **both** angular signals are small (< 0.05 rad/s), choose the one with
       the smaller absolute magnitude to avoid over-correction.
    3. Otherwise, if the predicted angular velocity is smaller than the
       waypoint-derived one, use the waypoint-derived value.
    4. If the predicted value is larger and both signals have the **same sign**,
       blend them with equal weight (0.8 / 0.2). If they have **opposite signs**,
       trust the predicted value and leave ``w`` unchanged.

    Args:
        waypoints:  Array of shape ``(N, 2)`` containing ``(dx, dy)`` waypoint
                    offsets in the robot frame.
        velocities: Array of shape ``(N, 2)`` containing ``(w, v)`` predicted
                    angular and linear velocities.
        dt:         Time-step duration in seconds (default: ``1/3`` s).

    Returns:
        control_commands: Array of shape ``(N, 2)`` where each row is
        ``[v_clipped, w_clipped]``.
    """
    N = waypoints.shape[0]
    control_commands = np.zeros((N, 2))

    for i in range(N):
        dx, dy = waypoints[i]
        w = velocities[i][0]

        # ------------------------------------------------------------------ #
        # Case 1: negligible displacement — stay in place, keep predicted w
        # ------------------------------------------------------------------ #
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            control_commands[i] = [0.0, w]
            continue

        # ------------------------------------------------------------------ #
        # Waypoint-derived angular velocity
        # ------------------------------------------------------------------ #
        w_wp = np.arctan(dy / dx) / DT

        # ------------------------------------------------------------------ #
        # Case 2 & 3 & 4: angular velocity selection / blending
        # ------------------------------------------------------------------ #
        if np.abs(w) < 0.05 and np.abs(w_wp) < 0.05:
            # Both signals are small — use the one closer to zero
            if np.abs(w) > np.abs(w_wp):
                w = w_wp

        else:
            if np.abs(w) < np.abs(w_wp):
                # Waypoint signal is dominant
                w = w_wp
            else:
                # Predicted signal is dominant — blend if same sign
                if (w > 0) == (w_wp > 0):          # same sign
                    w = 0.8 * w + 0.2 * w_wp
                # else: opposite signs → keep predicted w unchanged

        # ------------------------------------------------------------------ #
        # Linear velocity from waypoint geometry
        # ------------------------------------------------------------------ #
        v = dx / DT

        control_commands[i] = [
            np.clip(v, 0.0, MAX_V),
            np.clip(w, -MAX_W, MAX_W),
        ]

    return control_commands