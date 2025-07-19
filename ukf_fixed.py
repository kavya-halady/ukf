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
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import cv2
from PIL import Image
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UKFLocalizer:
    def __init__(self, map_file, initial_pose=None):
        """
        Initialize UKF Localizer

        Args:
            map_file: Path to map YAML file
            initial_pose: Initial pose [x, y, theta] (optional)
        """
        self.load_map(map_file)

        # UKF parameters
        self.dim_x = 3  # [x, y, theta]
        self.dim_z = 2  # [range, bearing] measurements from laser scan

        # Create sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=0)

        # Initialize UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=0.1,
            hx=self.measurement_function,
            fx=self.state_transition,
            points=points
        )

        # Initial state
        if initial_pose is None:
            self.ukf.x = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        else:
            self.ukf.x = np.array(initial_pose)

        # Initial covariance
        self.ukf.P = np.eye(3) * 0.1

        # Process noise
        self.ukf.Q = np.diag([0.1, 0.1, 0.05])  # Process noise for [x, y, theta]

        # Measurement noise
        self.ukf.R = np.diag([0.1, 0.05])  # Measurement noise for [range, bearing]

        # Store trajectory
        self.trajectory = []
        self.timestamps = []

    def load_map(self, map_file):
        """Load map from YAML file"""
        with open(map_file, 'r') as f:
            map_config = yaml.safe_load(f)

        # Load map image
        map_image_path = os.path.join(os.path.dirname(map_file), map_config['image'])
        self.map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)

        # Map parameters
        self.resolution = map_config['resolution']
        self.origin = map_config['origin']
        self.occupied_thresh = map_config.get('occupied_thresh', 0.65)
        self.free_thresh = map_config.get('free_thresh', 0.196)

        print(f"Map loaded: {self.map_image.shape}, resolution: {self.resolution}")

    def state_transition(self, x, dt, u=None):
        """
        State transition function for UKF

        Args:
            x: Current state [x, y, theta]
            dt: Time step
            u: Control input [v, w] (linear and angular velocity)

        Returns:
            Next state
        """
        if u is None:
            u = np.array([0.0, 0.0])  # No motion model

        x_new = np.zeros_like(x)

        # Simple motion model: x = x + v*cos(theta)*dt, y = y + v*sin(theta)*dt, theta = theta + w*dt
        x_new[0] = x[0] + u[0] * np.cos(x[2]) * dt
        x_new[1] = x[1] + u[0] * np.sin(x[2]) * dt
        x_new[2] = x[2] + u[1] * dt

        # Normalize theta
        x_new[2] = self.normalize_angle(x_new[2])

        return x_new

    def measurement_function(self, x):
        """
        Measurement function for UKF (simplified laser scan model)

        Args:
            x: Current state [x, y, theta]

        Returns:
            Expected measurements
        """
        # Simplified measurement model - returns expected range and bearing
        # In real implementation, this would simulate laser scan hits on map

        # For demonstration, return distance to nearest obstacle
        map_x = int((x[0] - self.origin[0]) / self.resolution)
        map_y = int((x[1] - self.origin[1]) / self.resolution)

        if (0 <= map_x < self.map_image.shape[1] and
            0 <= map_y < self.map_image.shape[0]):
            # Simple distance to obstacle
            range_measurement = self.get_distance_to_obstacle(map_x, map_y)
            bearing_measurement = 0.0  # Simplified bearing
        else:
            range_measurement = 10.0  # Max range
            bearing_measurement = 0.0

        return np.array([range_measurement, bearing_measurement])

    def get_distance_to_obstacle(self, x, y):
        """Calculate distance to nearest obstacle"""
        # Simple implementation - check surrounding area
        search_radius = 50
        min_distance = float('inf')

        for dx in range(-search_radius, search_radius):
            for dy in range(-search_radius, search_radius):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.map_image.shape[1] and
                    0 <= ny < self.map_image.shape[0]):
                    if self.map_image[ny, nx] < 100:  # Obstacle
                        distance = np.sqrt(dx*dx + dy*dy) * self.resolution
                        min_distance = min(min_distance, distance)

        return min(min_distance, 10.0)  # Cap at 10 meters

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def update(self, odom_data, scan_data, imu_data, timestamp):
        """
        Update UKF with sensor data

        Args:
            odom_data: Odometry data
            scan_data: Laser scan data
            imu_data: IMU data
            timestamp: Current timestamp
        """
        # Extract control input from odometry
        if odom_data is not None:
            linear_vel = odom_data.twist.twist.linear.x
            angular_vel = odom_data.twist.twist.angular.z
            u = np.array([linear_vel, angular_vel])
        else:
            u = np.array([0.0, 0.0])

        # Predict step
        self.ukf.predict(u=u)

        # Update step with laser scan
        if scan_data is not None:
            # Process laser scan to get measurements
            measurements = self.process_laser_scan(scan_data)
            if measurements is not None:
                self.ukf.update(measurements)

        # Store trajectory
        self.trajectory.append(self.ukf.x.copy())
        self.timestamps.append(timestamp)

    def process_laser_scan(self, scan_data):
        """Process laser scan data to extract meaningful measurements"""
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        # Filter valid ranges
        valid_indices = np.isfinite(ranges) & (ranges > scan_data.range_min) & (ranges < scan_data.range_max)

        if not np.any(valid_indices):
            return None

        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        # Take closest measurement as example
        min_idx = np.argmin(valid_ranges)
        closest_range = valid_ranges[min_idx]
        closest_bearing = valid_angles[min_idx]

        return np.array([closest_range, closest_bearing])

class RosbagReader:
    def __init__(self, bag_file):
        """Initialize rosbag reader"""
        self.bag_file = bag_file
        self.topics = ['/odom', '/scan', '/imu', '/tf', '/tf_static']

    def read_messages(self):
        """Read messages from rosbag"""
        messages = {topic: [] for topic in self.topics}

        storage_options = rosbag2_py.StorageOptions(uri=self.bag_file, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        try:
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()

                if topic in self.topics:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    messages[topic].append((timestamp, msg))
        finally:
            # Handle different versions of rosbag2_py
            # Some versions have close(), others don't
            if hasattr(reader, 'close'):
                reader.close()
            else:
                # For versions without close(), the reader is automatically cleaned up
                pass
        
        return messages

def load_ground_truth(csv_file):
    """Load ground truth data from CSV"""
    df = pd.read_csv(csv_file)

    # Expected columns: time, x, y, z, qx, qy, qz, qw
    required_cols = ['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in ground truth CSV")

    # Convert quaternion to euler angle (yaw)
    df['yaw'] = df.apply(lambda row: tf_transformations.euler_from_quaternion([
        row['qx'], row['qy'], row['qz'], row['qw']
    ])[2], axis=1)

    return df

def synchronize_data(ukf_trajectory, ukf_timestamps, ground_truth_df):
    """Synchronize UKF trajectory with ground truth data"""
    synchronized_pairs = []

    # Convert timestamps to seconds
    ukf_times = np.array(ukf_timestamps) / 1e9  # Convert nanoseconds to seconds
    gt_times = ground_truth_df['time'].values

    for i, ukf_time in enumerate(ukf_times):
        # Find closest ground truth timestamp
        time_diffs = np.abs(gt_times - ukf_time)
        closest_idx = np.argmin(time_diffs)

        # Only include if time difference is less than 0.5 seconds
        if time_diffs[closest_idx] < 0.5:
            ukf_pose = ukf_trajectory[i]
            gt_pose = ground_truth_df.iloc[closest_idx]

            synchronized_pairs.append({
                'time': ukf_time,
                'ukf_x': ukf_pose[0],
                'ukf_y': ukf_pose[1],
                'ukf_yaw': ukf_pose[2],
                'gt_x': gt_pose['x'],
                'gt_y': gt_pose['y'],
                'gt_yaw': gt_pose['yaw']
            })

    return pd.DataFrame(synchronized_pairs)

def calculate_rmse_metrics(sync_data):
    """Calculate RMSE and other metrics"""
    if len(sync_data) == 0:
        return {}

    # Position RMSE
    position_errors = np.sqrt((sync_data['ukf_x'] - sync_data['gt_x'])**2 +
                             (sync_data['ukf_y'] - sync_data['gt_y'])**2)

    # Angular RMSE (handle angle wrapping)
    angle_errors = np.abs(sync_data['ukf_yaw'] - sync_data['gt_yaw'])
    angle_errors = np.minimum(angle_errors, 2*np.pi - angle_errors)

    # Individual axis RMSE
    x_rmse = np.sqrt(np.mean((sync_data['ukf_x'] - sync_data['gt_x'])**2))
    y_rmse = np.sqrt(np.mean((sync_data['ukf_y'] - sync_data['gt_y'])**2))
    yaw_rmse = np.sqrt(np.mean(angle_errors**2))

    # Overall position RMSE
    position_rmse = np.sqrt(np.mean(position_errors**2))

    # Mean Absolute Error (MAE)
    position_mae = np.mean(position_errors)
    yaw_mae = np.mean(angle_errors)

    # Maximum errors
    max_position_error = np.max(position_errors)
    max_yaw_error = np.max(angle_errors)

    # Standard deviation of errors
    position_std = np.std(position_errors)
    yaw_std = np.std(angle_errors)

    return {
      'position_rmse': position_rmse,
      'x_rmse': x_rmse,
      'y_rmse': y_rmse,
      'yaw_rmse': yaw_rmse,
      'position_mae': position_mae,
      'yaw_mae': yaw_mae,
      'max_position_error': max_position_error,
      'max_yaw_error': max_yaw_error,
      'position_std': position_std,
      'yaw_std': yaw_std,
      'num_samples': len(sync_data)
    }

def estimate_velocity_from_bag(messages):
    """Estimate average velocity from bag data"""
    odom_messages = messages.get('/odom', [])

    if not odom_messages:
        return 0.0

    velocities = []
    for timestamp, msg in odom_messages:
        linear_vel = msg.twist.twist.linear.x
        velocities.append(abs(linear_vel))

    return np.mean(velocities) if velocities else 0.0

def plot_results(sync_data, metrics, velocity, output_dir):
    """Create comprehensive plots"""
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'UKF Localization Results (Avg Velocity: {velocity:.3f} m/s)', fontsize=16)

    # 1. Trajectory comparison
    axes[0, 0].plot(sync_data['gt_x'], sync_data['gt_y'], 'g-', linewidth=2, label='Ground Truth')
    axes[0, 0].plot(sync_data['ukf_x'], sync_data['ukf_y'], 'r--', linewidth=2, label='UKF Estimate')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')

    # 2. Position errors over time
    position_errors = np.sqrt((sync_data['ukf_x'] - sync_data['gt_x'])**2 +
                             (sync_data['ukf_y'] - sync_data['gt_y'])**2)
    axes[0, 1].plot(sync_data['time'], position_errors, 'b-', linewidth=1.5)
    axes[0, 1].axhline(y=metrics['position_rmse'], color='r', linestyle='--',
                           label=f'RMSE: {metrics["position_rmse"]:.4f} m')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position Error (m)')
    axes[0, 1].set_title('Position Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. X position comparison
    axes[0, 2].plot(sync_data['time'], sync_data['gt_x'], 'g-', linewidth=2, label='Ground Truth')
    axes[0, 2].plot(sync_data['time'], sync_data['ukf_x'], 'r--', linewidth=2, label='UKF Estimate')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('X Position (m)')
    axes[0, 2].set_title(f'X Position (RMSE: {metrics["x_rmse"]:.4f} m)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Y position comparison
    axes[1, 0].plot(sync_data['time'], sync_data['gt_y'], 'g-', linewidth=2, label='Ground Truth')
    axes[1, 0].plot(sync_data['time'], sync_data['ukf_y'], 'r--', linewidth=2, label='UKF Estimate')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Position (m)')
    axes[1, 0].set_title(f'Y Position (RMSE: {metrics["y_rmse"]:.4f} m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 5. Yaw angle comparison
    axes[1, 1].plot(sync_data['time'], sync_data['gt_yaw'], 'g-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(sync_data['time'], sync_data['ukf_yaw'], 'r--', linewidth=2, label='UKF Estimate')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Yaw Angle (rad)')
    axes[1, 1].set_title(f'Yaw Angle (RMSE: {metrics["yaw_rmse"]:.4f} rad)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 6. Error distribution histogram
    axes[1, 2].hist(position_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 2].axvline(x=metrics['position_rmse'], color='r', linestyle='--',
                           label=f'RMSE: {metrics["position_rmse"]:.4f} m')
    axes[1, 2].axvline(x=metrics['position_mae'], color='orange', linestyle='--',
                           label=f'MAE: {metrics["position_mae"]:.4f} m')
    axes[1, 2].set_xlabel('Position Error (m)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Position Error Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ukf_localization_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create RMSE vs Velocity plot if multiple runs
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(['Position RMSE', 'X RMSE', 'Y RMSE', 'Yaw RMSE (×10)'],
               [metrics['position_rmse'], metrics['x_rmse'], metrics['y_rmse'], metrics['yaw_rmse']*10],
               color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax.set_ylabel('RMSE Value')
    ax.set_title(f'RMSE Metrics Summary (Velocity: {velocity:.3f} m/s)')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate([metrics['position_rmse'], metrics['x_rmse'], metrics['y_rmse'], metrics['yaw_rmse']*10]):
            ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_metrics(metrics, velocity):
    """Print detailed metrics report"""
    print("\n" + "="*60)
    print(f"UKF LOCALIZATION PERFORMANCE REPORT")
    print("="*60)
    print(f"Average Velocity: {velocity:.4f} m/s")
    print(f"Number of synchronized samples: {metrics['num_samples']}")
    print("\nRMSE METRICS:")
    print(f" Position RMSE: {metrics['position_rmse']:.6f} m")
    print(f" X-axis RMSE: {metrics['x_rmse']:.6f} m")
    print(f" Y-axis RMSE: {metrics['y_rmse']:.6f} m")
    print(f" Yaw RMSE: {metrics['yaw_rmse']:.6f} rad ({np.degrees(metrics['yaw_rmse']):.3f}°)")

    print("\nMAE METRICS:")
    print(f"  Position MAE:      {metrics['position_mae']:.6f} m")
    print(f"  Yaw MAE:           {metrics['yaw_mae']:.6f} rad ({np.degrees(metrics['yaw_mae']):.3f}°)")

    print("\nMAXIMUM ERRORS:")
    print(f"  Max Position Error: {metrics['max_position_error']:.6f} m")
    print(f"  Max Yaw Error:      {metrics['max_yaw_error']:.6f} rad ({np.degrees(metrics['max_yaw_error']):.3f}°)")

    print("\nSTANDARD DEVIATION:")
    print(f"  Position Std:       {metrics['position_std']:.6f} m")
    print(f"  Yaw Std:            {metrics['yaw_std']:.6f} rad ({np.degrees(metrics['yaw_std']):.3f}°)")

    # Performance assessment
    print("\nPERFORMANCE ASSESSMENT:")
    if metrics['position_rmse'] < 0.1:
        performance = "EXCELLENT"
    elif metrics['position_rmse'] < 0.3:
        performance = "GOOD"
    elif metrics['position_rmse'] < 0.5:
        performance = "ACCEPTABLE"
    else:
        performance = "NEEDS IMPROVEMENT"

    print(f"  Overall Performance: {performance}")
    print(f"  Localization Accuracy: {(1 - min(metrics['position_rmse'], 1.0))*100:.1f}%")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='UKF Localization with RMSE Evaluation')
    parser.add_argument('--bag', required=True, help='Path to ROS2 bag file (.db3)')
    parser.add_argument('--map', required=True, help='Path to map YAML file')
    parser.add_argument('--groundtruth', required=True, help='Path to ground truth CSV file')
    parser.add_argument('--output', default='results', help='Output directory for results')

    args = parser.parse_args()

    print("Starting UKF Localization Analysis...")
    print(f"Bag file: {args.bag}")
    print(f"Map file: {args.map}")
    print(f"Ground truth: {args.groundtruth}")

    try:
        # Load ground truth data
        print("\nLoading ground truth data...")
        ground_truth_df = load_ground_truth(args.groundtruth)
        print(f"Loaded {len(ground_truth_df)} ground truth poses")

        # Read rosbag messages
        print("\nReading rosbag data...")
        bag_reader = RosbagReader(args.bag)
        messages = bag_reader.read_messages()

        # Estimate velocity
        avg_velocity = estimate_velocity_from_bag(messages)
        print(f"Estimated average velocity: {avg_velocity:.4f} m/s")

        # Initialize UKF localizer
        print("\nInitializing UKF localizer...")
        ukf_localizer = UKFLocalizer(args.map)

        # Process messages chronologically
        print("Processing sensor data...")
        all_messages = []

        for topic, msg_list in messages.items():
            for timestamp, msg in msg_list:
                all_messages.append((timestamp, topic, msg))

        # Sort by timestamp
        all_messages.sort(key=lambda x: x[0])

        # Process messages
        current_odom = None
        current_scan = None
        current_imu = None

        for timestamp, topic, msg in all_messages:
            if topic == '/odom':
                current_odom = msg
                # Update UKF when we have odometry
                ukf_localizer.update(current_odom, current_scan, current_imu, timestamp)
            elif topic == '/scan':
                current_scan = msg
            elif topic == '/imu':
                current_imu = msg

        print(f"Processed {len(ukf_localizer.trajectory)} poses")

        # Synchronize with ground truth
        print("\nSynchronizing with ground truth...")
        sync_data = synchronize_data(ukf_localizer.trajectory,
                                   ukf_localizer.timestamps,
                                   ground_truth_df)

        if len(sync_data) == 0:
            print("ERROR: No synchronized data points found!")
            return

        print(f"Synchronized {len(sync_data)} poses")

        # Calculate RMSE metrics
        print("\nCalculating RMSE metrics...")
        metrics = calculate_rmse_metrics(sync_data)

        # Print detailed results
        print_detailed_metrics(metrics, avg_velocity)

        # Generate plots
        print(f"\nGenerating plots and saving to {args.output}...")
        plot_results(sync_data, metrics, avg_velocity, args.output)

        # Save results to file
        results_file = os.path.join(args.output, 'metrics_report.txt')
        with open(results_file, 'w') as f:
            f.write(f"UKF Localization Performance Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bag file: {args.bag}\n")
            f.write(f"Map file: {args.map}\n")
            f.write(f"Ground truth: {args.groundtruth}\n\n")
            f.write(f"Average Velocity: {avg_velocity:.6f} m/s\n")
            f.write(f"Synchronized samples: {metrics['num_samples']}\n\n")
            f.write("RMSE METRICS:\n")
            for key, value in metrics.items():
                if 'rmse' in key.lower():
                    f.write(f"  {key}: {value:.6f}\n")
            f.write(f"\nPosition RMSE: {metrics['position_rmse']:.6f} m\n")
            f.write(f"Angular RMSE: {metrics['yaw_rmse']:.6f} rad\n")

        # Save synchronized data
        sync_data.to_csv(os.path.join(args.output, 'synchronized_data.csv'), index=False)

        print(f"\nResults saved to {args.output}/")
        print("Analysis complete!")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()