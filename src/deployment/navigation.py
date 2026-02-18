import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PointStamped
import message_filters
from tf.transformations import euler_from_quaternion

# Utils
from utils import (
    msg_to_pil, 
    to_numpy, 
    transform_images, 
    load_model,
    _generate_3d_waypoint_state,
    generate_waypoints_torch
)
from vint_train.training.train_utils import get_action
from topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC
)

# CONSTANTS
TOPOMAP_IMAGES_DIR = "/home/yx/topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
DT = 1.0 / robot_config["frame_rate"]
EPS = 1e-8
WAYPOINT_TIMEOUT = 1  # seconds

# GLOBALS
context_queue = []
context_size = 5  
state_queue = []
state_size = 5


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


      




def approx_sync_callback(rgb: Image, state: Odometry) -> None:
    """Callback for synchronized image and state messages"""
    global context_queue, state_queue
    
    obs_img = msg_to_pil(rgb)
    
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

    if state_size is not None:
        orientation_q = state.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        linear = state.twist.twist.linear.x
        angular = state.twist.twist.angular.z
        temp = [roll, pitch, yaw, linear, angular]

        if len(state_queue) < state_size + 1:
            state_queue.append(temp)
        else:
            state_queue.pop(0)
            state_queue.append(temp)


def main(args: argparse.Namespace) -> None:
    global context_size, state_size

    # Load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # Load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if not os.path.exists(ckpth_path):
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    
    print(f"Loading model from {ckpth_path}...")
    model = load_model(ckpth_path, model_params, device)
    model = model.to(device)
    model.eval()

    # Load topomap
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    topomap_filenames = sorted(
        os.listdir(topomap_dir), 
        key=lambda x: int(x.split(".")[0])
    )
    num_nodes = len(topomap_filenames)
    topomap = [PILImage.open(os.path.join(topomap_dir, fname)) for fname in topomap_filenames]

    # Validate goal node
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
    
    closest_node = 0
    reached_goal = False

    # ROS initialization
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    
    rgb_sub = message_filters.Subscriber('/rgbd_camera/color/image', Image)
    state_sub = message_filters.Subscriber('/state_estimation', Odometry)
    
    waypoint_pub = rospy.Publisher("/way_point", PointStamped, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    vel_pub = rospy.Publisher("/cmd_vel", TwistStamped, queue_size=1)
    
    ts = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, state_sub], 
        queue_size=100, 
        slop=0.15, 
        allow_headerless=True
    )
    ts.registerCallback(approx_sync_callback)

    print("Registered with master node. Waiting for image observations...")

    # Navigation loop
    while not rospy.is_shutdown():
        if len(context_queue) > model_params["context_size"] and len(state_queue) > state_size:
            
            if model_params["model_type"] == "nomad_pre_train_disntance_with_distance":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                current_obs = obs_images[-1].to(device) 
                obs_images = torch.cat(obs_images, dim=1).to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                
                goal_image = [
                    transform_images(g_img, model_params["image_size"], center_crop=False).to(device) 
                    for g_img in topomap[start:end + 1]
                ]
                goal_image = torch.concat(goal_image, dim=0)
                
                state_seq = torch.tensor(state_queue).to(device)
                
                # Predict distances
                dists = model("distance_net", obs_img=current_obs, goal_img=goal_image)
                dists = to_numpy(dists.flatten())
                
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(goal_image) - 1)
                img = goal_image[sg_idx].unsqueeze(0)
                
                obs_cond = model("vision_encoder", obs_img=obs_images, goal_img=img, input_goal_mask=mask)
                
                # Infer waypoints
                with torch.no_grad():
                    waypoint_pred, wp_enc = model("waypoint_pred_net", feature=obs_cond)
                    waypoint_pred = waypoint_pred[0]

                # Infer actions
                pred_action = model("action_pred_net", feature=torch.cat((obs_cond, wp_enc), 1))
                
                if model_params["normalize"]:
                    waypoint_pred[:2] *= (MAX_V / RATE) 
                
                # Get control commands
                controllers = combined_controller(
                    waypoint_pred.detach().cpu().numpy(), 
                    pred_action[0].detach().cpu().numpy()
                                       

                )[1]
                v, w = controllers
                
                # Publish velocity command
                vel_msg = TwistStamped()
                vel_msg.twist.linear.x = v
                vel_msg.twist.angular.z = w
                vel_pub.publish(vel_msg)
            

        
        reached_goal = closest_node == goal_node
        
        if reached_goal:
            print("Reached goal! Stopping...")
            vel_msg = TwistStamped()
            vel_msg.twist.linear.x = 0
            vel_msg.twist.angular.z = 0
            vel_pub.publish(vel_msg)
            break
            
        rate.sleep()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot"
    )
    parser.add_argument(
        "--model", "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml)"
    )

    parser.add_argument(
        "--dir", "-d",
        default="",
        type=str,
        help="path to topomap images"
    )
    parser.add_argument(
        "--goal-node", "-g",
        default=-1,
        type=int,
        help="goal node index in the topomap (-1 for last node)"
    )
    parser.add_argument(
        "--close-threshold", "-t",
        default=3,
        type=int,
        help="temporal distance within the next node before localizing to it"
    )
    parser.add_argument(
        "--radius", "-r",
        default=2,
        type=int,
        help="temporal number of local nodes to look at for localization"
    )
    parser.add_argument(
        "--num-samples", "-n",
        default=4,
        type=int,
        help="Number of actions sampled from the exploration model"
    )
    
    args = parser.parse_args()

    main(args)





