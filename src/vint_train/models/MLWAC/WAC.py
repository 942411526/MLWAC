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

def combined_controller(waypoints: np.ndarray, velocities: np.ndarray, dt: float = 1.0/3.0) -> np.ndarray:
    """
    WAC
    
    
        waypoints: (N, 2) 
        velocities: (N, 1) 
        dt 
    
    return:
        control_commands: (N, 2) 
    """
    N = waypoints.shape[0]
    control_commands = np.zeros((N, 2))
    
    for i in range(N):
        dx, dy = waypoints[i]
        w = velocities[i][0]
        
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            control_commands[i] = [0, w]
        else:
            w_waypoint = np.arctan2(dy, dx) / DT
            
            if np.abs(w) < 0.05 and np.abs(w_waypoint) < 0.05:
                w = w_waypoint if np.abs(w) > np.abs(w_waypoint) else w
            else:
                if np.abs(w) < np.abs(w_waypoint):
                    w = w_waypoint
                else:
                    if (w > 0 and w_waypoint > 0) or (w < 0 and w_waypoint < 0):
                        w = w * 0.8 + w_waypoint * 0.2
            
            v = dx / DT
            control_commands[i] = [np.clip(v, 0, MAX_V), np.clip(w, -MAX_W, MAX_W)]
    
    return control_commands
