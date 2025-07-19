#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
import sqlite3
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import tf_transformations
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import cv2
from PIL import Image
import os
from datetime import datetime
import warnings
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import json
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

class ImprovedUKFLocalizer:
    def __init__(self, map_file, initial_pose=None, velocity_based_tuning=True):
        """
        Initialize improved UKF Localizer with proper initialization and parameter tuning
        """
        self.load_map(map_file)
        self.velocity_based_tuning = velocity_based_tuning
        self.current_velocity = 0.0
        self.velocity_history = []
        
        # UKF parameters - properly tuned
        self.dim_x = 3  # [x, y, theta]
        self.dim_z = 2  # [range, bearing] - simplified measurement model
        
        # Create sigma points with optimized parameters
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
        
        # Initialize UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=0.1,
            hx=self.measurement_function,
            fx=self.state_transition,
            points=points
        )
        
        # Better initial state handling
        if initial_pose is None:
            self.ukf.x = np.array([0.0, 0.0, 0.0])
        else:
            self.ukf.x = np.array(initial_pose)
        
        # Properly tuned covariance matrices
        self.setup_noise_models()
        
        # Store trajectory and metadata
        self.trajectory = []
        self.timestamps = []
        self.velocities = []
        self.covariances = []
        self.innovations = []  # For debugging
        
        # Laser scan parameters - simplified
        self.max_range = 8.0
        self.min_range = 0.1
        
        # Map processing
        self.preprocess_map()
        
        # Motion model parameters
        self.motion_noise_std = {
            'linear': 0.02,   # 2cm std for linear motion
            'angular': 0.05,  # 3° std for angular motion
            'cross': 0.01     # Cross-correlation noise
        }
        
    def setup_noise_models(self):
        """Setup properly tuned noise models"""
        # Process noise - motion uncertainty
        self.ukf.Q = np.array([
            [0.01, 0.0, 0.0],   # x position variance
            [0.0, 0.01, 0.0],   # y position variance  
            [0.0, 0.0, 0.05]    # orientation variance
        ])
        
        # Measurement noise - sensor uncertainty
        self.ukf.R = np.array([
            [0.1, 0.0],   # range measurement variance
            [0.0, 0.2]    # bearing measurement variance
        ])
        
        # Initial covariance - conservative but not too large
        self.ukf.P = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5]
        ])
    
    def load_map(self, map_file):
        """Load map from YAML file with better error handling"""
        try:
            with open(map_file, 'r') as f:
                map_config = yaml.safe_load(f)
            
            # Handle relative and absolute paths
            map_image_path = map_config['image']
            if not os.path.isabs(map_image_path):
                map_image_path = os.path.join(os.path.dirname(map_file), map_image_path)
            
            if not os.path.exists(map_image_path):
                # Try other common locations
                possible_paths = [
                    map_config['image'],
                    os.path.join(os.getcwd(), map_config['image']),
                    os.path.join(os.path.dirname(map_file), os.path.basename(map_config['image']))
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        map_image_path = path
                        break
                else:
                    raise FileNotFoundError(f"Could not find map image. Tried: {possible_paths}")
            
            self.map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
            
            if self.map_image is None:
                raise ValueError(f"Could not load map image from: {map_image_path}")
            
            # Map parameters
            self.resolution = map_config['resolution']
            self.origin = np.array(map_config['origin'][:2])
            self.occupied_thresh = map_config.get('occupied_thresh', 0.65)
            self.free_thresh = map_config.get('free_thresh', 0.196)
            
            print(f"Map loaded: {self.map_image.shape}, resolution: {self.resolution}m/pixel")
            print(f"Map origin: {self.origin}")
            
        except Exception as e:
            print(f"Error loading map: {e}")
            raise
    
    def preprocess_map(self):
        """Preprocess map for efficient distance calculations"""
        # Create binary obstacle map (occupied = 1, free = 0)
        # Typical occupancy grid: 0 = free, 100 = occupied, -1 = unknown
        self.obstacle_map = (self.map_image < 50).astype(np.uint8)  # Black pixels are obstacles
        
        # Get obstacle coordinates in map frame
        obstacle_coords = np.where(self.obstacle_map == 1)
        if len(obstacle_coords[0]) == 0:
            print("Warning: No obstacles found in map!")
            self.obstacle_points_world = np.array([[0, 0]])
            self.obstacle_tree = None
            return
        
        self.obstacle_points_map = np.column_stack((obstacle_coords[1], obstacle_coords[0]))  # (x, y)
        
        # Convert to world coordinates
        self.obstacle_points_world = self.map_to_world(self.obstacle_points_map)
        
        # Build KD-tree for fast nearest neighbor queries
        self.obstacle_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        self.obstacle_tree.fit(self.obstacle_points_world)
        
        print(f"Preprocessed {len(self.obstacle_points_world)} obstacle points")
    
    def world_to_map(self, world_coords):
        """Convert world coordinates to map pixel coordinates"""
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)
        
        map_coords = np.zeros_like(world_coords)
        map_coords[:, 0] = (world_coords[:, 0] - self.origin[0]) / self.resolution
        map_coords[:, 1] = (world_coords[:, 1] - self.origin[1]) / self.resolution
        
        return map_coords.astype(int)
    
    def map_to_world(self, map_coords):
        """Convert map pixel coordinates to world coordinates"""
        if map_coords.ndim == 1:
            map_coords = map_coords.reshape(1, -1)
        
        world_coords = np.zeros_like(map_coords, dtype=float)
        world_coords[:, 0] = map_coords[:, 0] * self.resolution + self.origin[0]
        world_coords[:, 1] = map_coords[:, 1] * self.resolution + self.origin[1]
        
        return world_coords
    
    def state_transition(self, x, dt, u=None):
        """
        State transition function with proper odometry-based motion model
        """
        if u is None or len(u) != 2:
            return x.copy()  # No motion if no control input
        
        x_new = x.copy()
        linear_vel, angular_vel = u[0], u[1]
        
        # Differential drive motion model
        if abs(angular_vel) < 1e-6:  # Straight line motion
            x_new[0] = x[0] + linear_vel * np.cos(x[2]) * dt
            x_new[1] = x[1] + linear_vel * np.sin(x[2]) * dt
            x_new[2] = x[2]  # No rotation
        else:
            # Curved motion
            radius = linear_vel / angular_vel
            dtheta = angular_vel * dt
            
            x_new[0] = x[0] + radius * (np.sin(x[2] + dtheta) - np.sin(x[2]))
            x_new[1] = x[1] - radius * (np.cos(x[2] + dtheta) - np.cos(x[2]))
            x_new[2] = x[2] + dtheta
        
        # Normalize angle to [-pi, pi]
        x_new[2] = self.normalize_angle(x_new[2])
        
        return x_new
    
    def measurement_function(self, x):
        """
        Measurement function: returns expected sensor readings given robot state
        Uses simplified range-bearing model to closest obstacle
        """
        if self.obstacle_tree is None or len(self.obstacle_points_world) == 0:
            return np.array([self.max_range, 0.0])  # No obstacles visible
        
        robot_pos = np.array([x[0], x[1]]).reshape(1, -1)
        robot_angle = x[2]
        
        try:
            # Find closest obstacle
            distances, indices = self.obstacle_tree.kneighbors(robot_pos)
            closest_distance = distances[0, 0]
            closest_point = self.obstacle_points_world[indices[0, 0]]
            
            # Calculate bearing to closest obstacle
            dx = closest_point[0] - x[0]
            dy = closest_point[1] - x[1]
            absolute_bearing = np.arctan2(dy, dx)
            relative_bearing = self.normalize_angle(absolute_bearing - robot_angle)
            
            # Clamp range to sensor limits
            range_measurement = np.clip(closest_distance, self.min_range, self.max_range)
            
            return np.array([range_measurement, relative_bearing])
            
        except Exception as e:
            print(f"Warning in measurement function: {e}")
            return np.array([self.max_range, 0.0])
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def update(self, odom_data, scan_data, timestamp, dt=None):
        """
        Update UKF with sensor data
        """
        # Calculate time step
        if dt is None:
            if len(self.timestamps) > 0:
                dt = max(0.01, min((timestamp - self.timestamps[-1]) / 1e9, 0.5))
            else:
                dt = 0.1
        
        self.ukf.dt = dt
        
        # Extract control input from odometry
        if odom_data is not None:
            linear_vel = odom_data.twist.twist.linear.x
            angular_vel = odom_data.twist.twist.angular.z
            u = np.array([linear_vel, angular_vel])
            
            # Track velocity for analysis
            velocity = np.sqrt(linear_vel**2 + angular_vel**2)
            self.current_velocity = velocity
            self.velocities.append(velocity)
            
            # Adaptive noise based on velocity
            if self.velocity_based_tuning:
                velocity_factor = 1.0 + 2.0 * velocity  # Increase noise with speed
                self.ukf.Q = self.ukf.Q * velocity_factor
        else:
            u = np.array([0.0, 0.0])
            self.velocities.append(0.0)
        
        # Predict step
        self.ukf.predict(u=u)
        
        # Update step with laser scan
        if scan_data is not None:
            measurement = self.process_laser_scan(scan_data)
            if measurement is not None:
                # Store innovation for debugging
                predicted_z = self.measurement_function(self.ukf.x)
                innovation = measurement - predicted_z
                self.innovations.append(np.linalg.norm(innovation))
                
                self.ukf.update(measurement)
        
        # Store trajectory
        self.trajectory.append(self.ukf.x.copy())
        self.timestamps.append(timestamp)
        self.covariances.append(self.ukf.P.copy())
    
    def process_laser_scan(self, scan_data):
        """
        Process laser scan to extract meaningful measurement
        """
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
        
        # Filter valid ranges
        valid_mask = np.isfinite(ranges) & (ranges > scan_data.range_min) & (ranges < scan_data.range_max)
        
        if not np.any(valid_mask):
            return None
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Find closest obstacle in front sector (-45° to +45°)
        front_mask = np.abs(valid_angles) < np.pi/4
        
        if np.any(front_mask):
            front_ranges = valid_ranges[front_mask]
            front_angles = valid_angles[front_mask]
            
            # Get closest reading
            min_idx = np.argmin(front_ranges)
            closest_range = front_ranges[min_idx]
            closest_angle = front_angles[min_idx]
            
            return np.array([closest_range, closest_angle])
        
        return None

class RosbagReader:
    """Robust rosbag reader"""
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.required_topics = ['/odom', '/scan']
        self.optional_topics = ['/imu', '/tf', '/tf_static']
    
    def read_messages(self):
        """Read messages with proper error handling"""
        messages = {topic: [] for topic in self.required_topics + self.optional_topics}
        
        try:
            storage_options = rosbag2_py.StorageOptions(uri=self.bag_file, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            
            topic_types = reader.get_all_topics_and_types()
            type_map = {topic.name: topic.type for topic in topic_types}
            
            available_topics = set(type_map.keys())
            print(f"Available topics: {list(available_topics)}")
            
            # Check if required topics are available
            for topic in self.required_topics:
                if topic not in available_topics:
                    print(f"Warning: Required topic {topic} not found in bag!")
            
            message_count = 0
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                if topic in messages:
                    try:
                        msg_type = get_message(type_map[topic])
                        msg = deserialize_message(data, msg_type)
                        messages[topic].append((timestamp, msg))
                        message_count += 1
                        
                        if message_count % 1000 == 0:
                            print(f"Processed {message_count} messages...")
                            
                    except Exception as e:
                        print(f"Warning: Failed to deserialize {topic} message: {e}")
            
            print(f"Successfully read {message_count} total messages")
            for topic, msgs in messages.items():
                if msgs:
                    print(f"  {topic}: {len(msgs)} messages")
            
        except Exception as e:
            print(f"Error reading rosbag: {e}")
            raise
        
        return messages

def load_ground_truth(csv_file):
    """Load and validate ground truth data"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Ground truth columns: {list(df.columns)}")
        
        # Handle different column naming conventions
        column_mappings = {
            'timestamp': ['time', 'timestamp', 't'],
            'x': ['pos_x', 'x', 'position_x'],
            'y': ['pos_y', 'y', 'position_y'], 
            'z': ['pos_z', 'z', 'position_z'],
            'qx': ['orient_x', 'qx', 'orientation_x'],
            'qy': ['orient_y', 'qy', 'orientation_y'],
            'qz': ['orient_z', 'qz', 'orientation_z'],
            'qw': ['orient_w', 'qw', 'orientation_w']
        }
        
        # Map columns
        mapped_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    mapped_columns[standard_name] = possible_name
                    break
        
        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in mapped_columns.items()})
        
        # Validate required columns
        required_cols = ['time', 'x', 'y', 'qx', 'qy', 'qz', 'qw']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert quaternion to yaw angle
        def quat_to_yaw(qx, qy, qz, qw):
            return tf_transformations.euler_from_quaternion([qx, qy, qz, qw])[2]
        
        df['yaw'] = df.apply(lambda row: quat_to_yaw(row['qx'], row['qy'], row['qz'], row['qw']), axis=1)
        
        # Sort by timestamp
        df = df.sort_values('time').reset_index(drop=True)
        
        # Basic outlier removal
        for col in ['x', 'y']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outliers.any():
                print(f"Removing {outliers.sum()} outliers in {col}")
                df = df[~outliers].reset_index(drop=True)
        
        print(f"Loaded {len(df)} ground truth poses")
        print(f"Time range: {df['time'].iloc[0]:.2f} - {df['time'].iloc[-1]:.2f}")
        print(f"Position range: X[{df['x'].min():.2f}, {df['x'].max():.2f}], Y[{df['y'].min():.2f}, {df['y'].max():.2f}]")
        
        return df
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        raise

def synchronize_trajectories(ukf_trajectory, ukf_timestamps, ground_truth_df, max_time_diff=0.2):
    """
    Improved trajectory synchronization with interpolation
    """
    # Convert timestamps to seconds
    ukf_times = np.array(ukf_timestamps) / 1e9
    gt_times = ground_truth_df['time'].values
    
    # Handle different time formats
    if gt_times[0] > 1e10:  # Nanoseconds
        gt_times = gt_times / 1e9
    elif gt_times[0] > 1e6:  # Microseconds  
        gt_times = gt_times / 1e6
    
    print(f"UKF time range: {ukf_times[0]:.2f} - {ukf_times[-1]:.2f}")
    print(f"GT time range: {gt_times[0]:.2f} - {gt_times[-1]:.2f}")
    
    # Find overlapping time range
    start_time = max(ukf_times[0], gt_times[0])
    end_time = min(ukf_times[-1], gt_times[-1])
    
    if start_time >= end_time:
        raise ValueError("No temporal overlap between UKF and ground truth data!")
    
    print(f"Overlapping time range: {start_time:.2f} - {end_time:.2f}")
    
    # Interpolate ground truth to UKF timestamps
    synchronized_pairs = []
    
    # Create interpolation functions for ground truth
    gt_x_interp = interp1d(gt_times, ground_truth_df['x'], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    gt_y_interp = interp1d(gt_times, ground_truth_df['y'], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    gt_yaw_interp = interp1d(gt_times, ground_truth_df['yaw'], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    
    sync_count = 0
    for i, (ukf_time, ukf_pose) in enumerate(zip(ukf_times, ukf_trajectory)):
        # Check if within valid time range
        if start_time <= ukf_time <= end_time:
            # Interpolate ground truth at UKF timestamp
            gt_x = float(gt_x_interp(ukf_time))
            gt_y = float(gt_y_interp(ukf_time))
            gt_yaw = float(gt_yaw_interp(ukf_time))
            
            synchronized_pairs.append({
                'time': ukf_time,
                'ukf_x': ukf_pose[0],
                'ukf_y': ukf_pose[1],
                'ukf_yaw': ukf_pose[2],
                'gt_x': gt_x,
                'gt_y': gt_y,
                'gt_yaw': gt_yaw
            })
            sync_count += 1
    
    print(f"Synchronized {sync_count} poses using interpolation")
    return pd.DataFrame(synchronized_pairs)

def calculate_rmse_metrics(sync_data):
    """
    Calculate comprehensive RMSE and accuracy metrics
    """
    if len(sync_data) == 0:
        return {}
    
    # Position errors (Euclidean distance)
    position_errors = np.sqrt((sync_data['ukf_x'] - sync_data['gt_x'])**2 + 
                             (sync_data['ukf_y'] - sync_data['gt_y'])**2)
    
    # Individual axis errors
    x_errors = sync_data['ukf_x'] - sync_data['gt_x']
    y_errors = sync_data['ukf_y'] - sync_data['gt_y']
    
    # Angular errors (handle wrapping)
    yaw_errors = sync_data['ukf_yaw'] - sync_data['gt_yaw']
    yaw_errors = np.array([np.arctan2(np.sin(err), np.cos(err)) for err in yaw_errors])
    
    # RMSE calculations (Root Mean Square Error)
    position_rmse = np.sqrt(np.mean(position_errors**2))
    x_rmse = np.sqrt(np.mean(x_errors**2))
    y_rmse = np.sqrt(np.mean(y_errors**2))
    yaw_rmse = np.sqrt(np.mean(yaw_errors**2))
    
    # MAE calculations (Mean Absolute Error)
    position_mae = np.mean(position_errors)
    x_mae = np.mean(np.abs(x_errors))
    y_mae = np.mean(np.abs(y_errors))
    yaw_mae = np.mean(np.abs(yaw_errors))
    
    # Percentile errors
    position_95th = np.percentile(position_errors, 95)
    position_75th = np.percentile(position_errors, 75)
    yaw_95th = np.percentile(np.abs(yaw_errors), 95)
    
    # Standard deviations
    position_std = np.std(position_errors)
    yaw_std = np.std(yaw_errors)
    
    # Maximum errors
    max_position_error = np.max(position_errors)
    max_yaw_error = np.max(np.abs(yaw_errors))
    
    # Calculate trajectory statistics
    gt_distances = np.sqrt(np.diff(sync_data['gt_x'])**2 + np.diff(sync_data['gt_y'])**2)
    total_distance = np.sum(gt_distances)
    avg_speed = total_distance / (sync_data['time'].iloc[-1] - sync_data['time'].iloc[0]) if len(sync_data) > 1 else 0
    
    # Accuracy percentage (within certain thresholds)
    accuracy_thresholds = [0.1, 0.2, 0.5, 1.0]  # meters
    accuracies = {}
    for threshold in accuracy_thresholds:
        within_threshold = np.sum(position_errors <= threshold)
        accuracies[f'accuracy_{threshold}m'] = (within_threshold / len(position_errors)) * 100
    
    # Drift analysis (linear trend in errors)
    time_normalized = np.arange(len(position_errors))
    drift_slope = np.polyfit(time_normalized, position_errors, 1)[0] if len(position_errors) > 1 else 0
    
    return {
        'position_rmse': position_rmse,
        'x_rmse': x_rmse,
        'y_rmse': y_rmse,
        'yaw_rmse': yaw_rmse,
        'position_mae': position_mae,
        'x_mae': x_mae,
        'y_mae': y_mae,
        'yaw_mae': yaw_mae,
        'position_95th': position_95th,
        'position_75th': position_75th,
        'yaw_95th': yaw_95th,
        'position_std': position_std,
        'yaw_std': yaw_std,
        'max_position_error': max_position_error,
        'max_yaw_error': max_yaw_error,
        'avg_speed': avg_speed,
        'total_distance': total_distance,
        'drift_slope': drift_slope,
        'num_samples': len(sync_data),
        'time_span': sync_data['time'].iloc[-1] - sync_data['time'].iloc[0] if len(sync_data) > 1 else 0,
        **accuracies
    }

def create_analysis_plots(sync_data, metrics, output_dir, speed_label):
    """
    Create comprehensive analysis plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate errors for plotting
    position_errors = np.sqrt((sync_data['ukf_x'] - sync_data['gt_x'])**2 + 
                             (sync_data['ukf_y'] - sync_data['gt_y'])**2)
    
    yaw_errors = sync_data['ukf_yaw'] - sync_data['gt_yaw']
    yaw_errors = np.array([np.arctan2(np.sin(err), np.cos(err)) for err in yaw_errors])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Trajectory comparison (large subplot)
    ax1 = plt.subplot(3, 3, (1, 4))
    ax1.plot(sync_data['gt_x'], sync_data['gt_y'], 'g-', linewidth=3, 
             label='Ground Truth', alpha=0.8)
    ax1.plot(sync_data['ukf_x'], sync_data['ukf_y'], 'r--', linewidth=2, 
             label='UKF Estimate', alpha=0.8)
    ax1.scatter(sync_data['gt_x'].iloc[0], sync_data['gt_y'].iloc[0], 
                c='green', s=150, marker='o', label='Start', edgecolor='black', linewidth=2)
    ax1.scatter(sync_data['gt_x'].iloc[-1], sync_data['gt_y'].iloc[-1], 
                c='red', s=150, marker='s', label='End', edgecolor='black', linewidth=2)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(f'Trajectory Comparison - {speed_label}\nRMSE: {metrics["position_rmse"]:.3f}m', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
# 2. Position error over time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(sync_data['time'] - sync_data['time'].iloc[0], position_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Position Error (m)', fontsize=10)
    ax2.set_title(f'Position Error vs Time\nMean: {metrics["position_mae"]:.3f}m', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Position error histogram
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(position_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(metrics["position_mae"], color='red', linestyle='--', linewidth=2, label=f'MAE: {metrics["position_mae"]:.3f}m')
    ax3.axvline(metrics["position_rmse"], color='orange', linestyle='--', linewidth=2, label=f'RMSE: {metrics["position_rmse"]:.3f}m')
    ax3.set_xlabel('Position Error (m)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Error Distribution', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. X and Y errors over time
    ax4 = plt.subplot(3, 3, 5)
    ax4.plot(sync_data['time'] - sync_data['time'].iloc[0], sync_data['ukf_x'] - sync_data['gt_x'], 'r-', linewidth=1.5, label='X Error')
    ax4.plot(sync_data['time'] - sync_data['time'].iloc[0], sync_data['ukf_y'] - sync_data['gt_y'], 'b-', linewidth=1.5, label='Y Error')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Position Error (m)', fontsize=10)
    ax4.set_title(f'X/Y Errors\nX RMSE: {metrics["x_rmse"]:.3f}m, Y RMSE: {metrics["y_rmse"]:.3f}m', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Yaw error over time
    ax5 = plt.subplot(3, 3, 6)
    ax5.plot(sync_data['time'] - sync_data['time'].iloc[0], np.degrees(yaw_errors), 'purple', linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Yaw Error (degrees)', fontsize=10)
    ax5.set_title(f'Yaw Error vs Time\nRMSE: {np.degrees(metrics["yaw_rmse"]):.1f}°', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # 6. Error scatter plot
    ax6 = plt.subplot(3, 3, 7)
    scatter = ax6.scatter(sync_data['ukf_x'] - sync_data['gt_x'], 
                         sync_data['ukf_y'] - sync_data['gt_y'], 
                         c=position_errors, cmap='viridis', alpha=0.6)
    ax6.set_xlabel('X Error (m)', fontsize=10)
    ax6.set_ylabel('Y Error (m)', fontsize=10)
    ax6.set_title('X vs Y Error Distribution', fontsize=11)
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Total Error (m)')
    
    # 7. Cumulative error
    ax7 = plt.subplot(3, 3, 8)
    cumulative_error = np.cumsum(position_errors) / np.arange(1, len(position_errors) + 1)
    ax7.plot(sync_data['time'] - sync_data['time'].iloc[0], cumulative_error, 'green', linewidth=2)
    ax7.set_xlabel('Time (s)', fontsize=10)
    ax7.set_ylabel('Cumulative Mean Error (m)', fontsize=10)
    ax7.set_title('Cumulative Average Error', fontsize=11)
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance summary text
    ax8 = plt.subplot(3, 3, 9)
    ax8.axis('off')
    
    # Performance assessment
    if metrics["position_rmse"] < 0.5:
        performance = "EXCELLENT"
        color = 'green'
    elif metrics["position_rmse"] < 1.0:
        performance = "GOOD"  
        color = 'orange'
    elif metrics["position_rmse"] < 2.0:
        performance = "FAIR"
        color = 'red'
    else:
        performance = "POOR"
        color = 'darkred'
    
    summary_text = f"""
PERFORMANCE SUMMARY
Speed: {speed_label}
Samples: {metrics['num_samples']}
Time: {metrics['time_span']:.1f}s
Distance: {metrics['total_distance']:.1f}m

POSITION ACCURACY:
RMSE: {metrics['position_rmse']:.3f}m
MAE: {metrics['position_mae']:.3f}m
95th %ile: {metrics['position_95th']:.3f}m

ORIENTATION ACCURACY:
Yaw RMSE: {np.degrees(metrics['yaw_rmse']):.1f}°
Yaw MAE: {np.degrees(metrics['yaw_mae']):.1f}°

ACCURACY RATES:
<0.5m: {metrics.get('accuracy_0.5m', 0):.1f}%
<1.0m: {metrics.get('accuracy_1.0m', 0):.1f}%

OVERALL: {performance}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'ukf_localization_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'detailed_results.json')
    detailed_results = {
        'speed_label': speed_label,
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items()},
        'sync_data_summary': {
            'num_points': len(sync_data),
            'time_range': [float(sync_data['time'].min()), float(sync_data['time'].max())],
            'position_range_x': [float(sync_data['gt_x'].min()), float(sync_data['gt_x'].max())],
            'position_range_y': [float(sync_data['gt_y'].min()), float(sync_data['gt_y'].max())]
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    plt.show()
    return output_file

def print_performance_report(metrics, speed_label):
    """
    Print comprehensive performance report
    """
    print("=" * 60)
    print("UKF LOCALIZATION PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Speed Configuration: {speed_label}")
    print(f"Number of synchronized samples: {metrics['num_samples']}")
    print(f"Total trajectory distance: {metrics['total_distance']:.2f} m")
    print(f"Average speed: {metrics['avg_speed']:.3f} m/s")
    print(f"Time span: {metrics['time_span']:.1f} seconds")
    print()
    
    print("RMSE METRICS:")
    print(f"  Position RMSE: {metrics['position_rmse']:.6f} m")
    print(f"  X-axis RMSE: {metrics['x_rmse']:.6f} m") 
    print(f"  Y-axis RMSE: {metrics['y_rmse']:.6f} m")
    print(f"  Yaw RMSE: {metrics['yaw_rmse']:.6f} rad ({np.degrees(metrics['yaw_rmse']):.3f}°)")
    print()
    
    print("MAE METRICS:")
    print(f"  Position MAE: {metrics['position_mae']:.6f} m")
    print(f"  X-axis MAE: {metrics['x_mae']:.6f} m")
    print(f"  Y-axis MAE: {metrics['y_mae']:.6f} m") 
    print(f"  Yaw MAE: {metrics['yaw_mae']:.6f} rad ({np.degrees(metrics['yaw_mae']):.3f}°)")
    print()
    
    print("PERCENTILE ERRORS:")
    print(f"  95th percentile position error: {metrics['position_95th']:.6f} m")
    print(f"  75th percentile position error: {metrics['position_75th']:.6f} m")
    print(f"  95th percentile yaw error: {np.degrees(metrics['yaw_95th']):.3f}°")
    print()
    
    print("MAXIMUM ERRORS:")
    print(f"  Max Position Error: {metrics['max_position_error']:.6f} m")
    print(f"  Max Yaw Error: {np.degrees(metrics['max_yaw_error']):.3f}°")
    print()
    
    print("STANDARD DEVIATION:")
    print(f"  Position Std: {metrics['position_std']:.6f} m")
    print(f"  Yaw Std: {np.degrees(metrics['yaw_std']):.3f}°")
    print()
    
    print("ACCURACY RATES:")
    for threshold in [0.1, 0.2, 0.5, 1.0]:
        key = f'accuracy_{threshold}m'
        if key in metrics:
            print(f"  Within {threshold}m: {metrics[key]:.1f}%")
    print()
    
    print("DRIFT ANALYSIS:")
    print(f"  Error drift slope: {metrics['drift_slope']:.6f} m/sample")
    if abs(metrics['drift_slope']) > 0.01:
        print("  WARNING: Significant drift detected!")
    print()
    
    # Performance assessment
    if metrics["position_rmse"] < 0.2:
        performance = "EXCELLENT"
        accuracy_rating = "95%+"
    elif metrics["position_rmse"] < 0.5:
        performance = "VERY GOOD" 
        accuracy_rating = "85-95%"
    elif metrics["position_rmse"] < 1.0:
        performance = "GOOD"
        accuracy_rating = "70-85%"
    elif metrics["position_rmse"] < 2.0:
        performance = "FAIR"
        accuracy_rating = "50-70%"
    else:
        performance = "NEEDS IMPROVEMENT"
        accuracy_rating = "<50%"
    
    print("PERFORMANCE ASSESSMENT:")
    print(f"  Overall Performance: {performance}")
    print(f"  Localization Accuracy: {accuracy_rating}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='UKF Localization Analysis')
    parser.add_argument('--bag', required=True, help='Path to ROS bag file (.db3)')
    parser.add_argument('--map', required=True, help='Path to map YAML file')
    parser.add_argument('--groundtruth', required=True, help='Path to ground truth CSV file')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Starting UKF Localization Analysis...")
    print(f"Bag file: {args.bag}")
    print(f"Map file: {args.map}")
    print(f"Ground truth: {args.groundtruth}")
    print()
    
    # Extract speed from filename for labeling
    bag_filename = os.path.basename(args.bag)
    if 'speed' in bag_filename.lower():
        speed_match = bag_filename.split('speed_')[1].split('_')[0] if 'speed_' in bag_filename else bag_filename.split('speed')[1].split('_')[0]
        speed_label = f"{speed_match} m/s"
    else:
        speed_label = "Unknown Speed"
    
    # Load ground truth data
    print("Loading ground truth data...")
    try:
        ground_truth_df = load_ground_truth(args.groundtruth)
    except Exception as e:
        print(f"Failed to load ground truth: {e}")
        return
    
    # Read rosbag data
    print("Reading rosbag data...")
    try:
        reader = RosbagReader(args.bag)
        messages = reader.read_messages()
    except Exception as e:
        print(f"Failed to read rosbag: {e}")
        return
    
    if not messages['/odom'] or not messages['/scan']:
        print("Error: Missing required topics (/odom or /scan)")
        return
    
    # Estimate average velocity for initial tuning
    velocities = []
    for _, odom_msg in messages['/odom'][:100]:  # Sample first 100 messages
        linear_vel = odom_msg.twist.twist.linear.x
        angular_vel = odom_msg.twist.twist.angular.z
        velocity = np.sqrt(linear_vel**2 + angular_vel**2)
        velocities.append(velocity)
    
    avg_velocity = np.mean(velocities) if velocities else 0.1
    print(f"Estimated average velocity: {avg_velocity:.4f} m/s")
    
    # Initialize UKF localizer with better initial pose estimation
    print("Initializing UKF localizer...")
    try:
        # Use first ground truth pose as initial estimate if available
        initial_pose = [ground_truth_df['x'].iloc[0], 
                       ground_truth_df['y'].iloc[0], 
                       ground_truth_df['yaw'].iloc[0]]
        
        ukf_localizer = ImprovedUKFLocalizer(
            map_file=args.map,
            initial_pose=initial_pose,
            velocity_based_tuning=True
        )
    except Exception as e:
        print(f"Failed to initialize UKF: {e}")
        return
    
    # Process sensor data
    print("Processing sensor data...")
    odom_dict = {timestamp: msg for timestamp, msg in messages['/odom']}
    scan_dict = {timestamp: msg for timestamp, msg in messages['/scan']}
    
    # Get all timestamps and sort
    all_timestamps = sorted(set(odom_dict.keys()) | set(scan_dict.keys()))
    
    processed_count = 0
    for timestamp in all_timestamps:
        odom_data = odom_dict.get(timestamp)
        scan_data = scan_dict.get(timestamp)
        
        # Skip if neither sensor has data
        if odom_data is None and scan_data is None:
            continue
        
        try:
            ukf_localizer.update(odom_data, scan_data, timestamp)
            processed_count += 1
            
            if processed_count % 200 == 0:
                print(f"  Processed {processed_count} poses...")
                
        except Exception as e:
            print(f"Warning: Failed to process timestamp {timestamp}: {e}")
    
    print(f"Processed {processed_count} poses")
    
    if len(ukf_localizer.trajectory) == 0:
        print("Error: No poses were processed successfully!")
        return
    
    # Synchronize with ground truth
    print("Synchronizing with ground truth...")
    try:
        sync_data = synchronize_trajectories(
            ukf_localizer.trajectory,
            ukf_localizer.timestamps,
            ground_truth_df
        )
    except Exception as e:
        print(f"Failed to synchronize trajectories: {e}")
        return
    
    if len(sync_data) == 0:
        print("Error: No synchronized data points found!")
        return
        
    print(f"Synchronized {len(sync_data)} poses")
    
    # Calculate metrics
    print("Calculating RMSE metrics...")
    metrics = calculate_rmse_metrics(sync_data)
    
    # Print performance report
    print_performance_report(metrics, speed_label)
    
    # Create output directory
    output_dir = os.path.join(args.output, f"results-{speed_label.replace(' ', '').replace('/', '')}")
    print(f"\nGenerating plots and saving to {output_dir}...")
    
    # Generate analysis plots
    try:
        plot_file = create_analysis_plots(sync_data, metrics, output_dir, speed_label)
        print(f"Analysis complete! Results saved to: {output_dir}")
        
        # Save synchronized data for further analysis
        sync_file = os.path.join(output_dir, 'synchronized_data.csv')
        sync_data.to_csv(sync_file, index=False)
        print(f"Synchronized data saved to: {sync_file}")
        
    except Exception as e:
        print(f"Failed to generate plots: {e}")

if __name__ == "__main__":
    main()