#!/usr/bin/env python3
"""
UKF Localization for TurtleBot 4 with ROS2 Bag Data
Based on the manifold-based UKF approach from the reference paper
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cholesky
import cv2
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional
import time

class SO2:
    """SO(2) Lie Group operations"""
    
    @staticmethod
    def exp(phi: np.ndarray) -> np.ndarray:
        """Exponential map: R -> SO(2)"""
        if phi.ndim == 0 or len(phi) == 1:
            phi = phi.item() if hasattr(phi, 'item') else phi
            return np.array([[np.cos(phi), -np.sin(phi)],
                           [np.sin(phi), np.cos(phi)]])
        else:
            # Handle batch
            c = np.cos(phi)
            s = np.sin(phi)
            R = np.zeros((len(phi), 2, 2))
            R[:, 0, 0] = c
            R[:, 0, 1] = -s
            R[:, 1, 0] = s
            R[:, 1, 1] = c
            return R
    
    @staticmethod
    def log(R_mat: np.ndarray) -> np.ndarray:
        """Logarithm map: SO(2) -> R"""
        if R_mat.ndim == 2:
            return np.arctan2(R_mat[1, 0], R_mat[0, 0])
        else:
            return np.arctan2(R_mat[:, 1, 0], R_mat[:, 0, 0])

class SE2:
    """SE(2) Lie Group operations"""
    
    @staticmethod
    def exp(xi: np.ndarray) -> np.ndarray:
        """Exponential map: R^3 -> SE(2)"""
        if xi.ndim == 1:
            rho = xi[:2]
            phi = xi[2]
            
            V = SE2._V_matrix(phi)
            T = np.eye(3)
            T[:2, :2] = SO2.exp(phi)
            T[:2, 2] = V @ rho
            return T
        else:
            # Batch processing
            batch_size = xi.shape[0]
            T = np.zeros((batch_size, 3, 3))
            T[:, 2, 2] = 1
            
            rho = xi[:, :2]
            phi = xi[:, 2]
            
            V = SE2._V_matrix_batch(phi)
            R_batch = SO2.exp(phi)
            T[:, :2, :2] = R_batch
            T[:, :2, 2] = np.einsum('bij,bj->bi', V, rho)
            return T
    
    @staticmethod
    def log(T: np.ndarray) -> np.ndarray:
        """Logarithm map: SE(2) -> R^3"""
        if T.ndim == 2:
            R_mat = T[:2, :2]
            t = T[:2, 2]
            phi = SO2.log(R_mat)
            
            V_inv = SE2._V_inv_matrix(phi)
            rho = V_inv @ t
            
            return np.array([rho[0], rho[1], phi])
        else:
            # Batch processing
            batch_size = T.shape[0]
            xi = np.zeros((batch_size, 3))
            
            R_batch = T[:, :2, :2]
            t_batch = T[:, :2, 2]
            phi = SO2.log(R_batch)
            
            V_inv = SE2._V_inv_matrix_batch(phi)
            rho = np.einsum('bij,bj->bi', V_inv, t_batch)
            
            xi[:, :2] = rho
            xi[:, 2] = phi
            return xi
    
    @staticmethod
    def _V_matrix(phi: float) -> np.ndarray:
        """Left Jacobian matrix for SE(2)"""
        if abs(phi) < 1e-8:
            return np.eye(2)
        
        s = np.sin(phi)
        c = np.cos(phi)
        return (s/phi) * np.eye(2) + ((1-c)/phi) * np.array([[0, -1], [1, 0]])
    
    @staticmethod
    def _V_inv_matrix(phi: float) -> np.ndarray:
        """Inverse of left Jacobian matrix for SE(2)"""
        if abs(phi) < 1e-8:
            return np.eye(2)
        
        half_phi = 0.5 * phi
        cot_half = 1. / np.tan(half_phi)
        return half_phi * cot_half * np.eye(2) + half_phi * np.array([[0, 1], [-1, 0]])
    
    @staticmethod
    def _V_matrix_batch(phi: np.ndarray) -> np.ndarray:
        """Batch version of V matrix"""
        batch_size = len(phi)
        V = np.zeros((batch_size, 2, 2))
        
        small_angle = np.abs(phi) < 1e-8
        large_angle = ~small_angle
        
        # Small angle approximation
        V[small_angle] = np.eye(2)[None, :, :]
        
        # Large angle
        if np.any(large_angle):
            phi_large = phi[large_angle]
            s = np.sin(phi_large)
            c = np.cos(phi_large)
            
            sin_over_phi = s / phi_large
            one_minus_cos_over_phi = (1 - c) / phi_large
            
            V[large_angle, 0, 0] = sin_over_phi
            V[large_angle, 1, 1] = sin_over_phi
            V[large_angle, 0, 1] = -one_minus_cos_over_phi
            V[large_angle, 1, 0] = one_minus_cos_over_phi
        
        return V
    
    @staticmethod
    def _V_inv_matrix_batch(phi: np.ndarray) -> np.ndarray:
        """Batch version of V inverse matrix"""
        batch_size = len(phi)
        V_inv = np.zeros((batch_size, 2, 2))
        
        small_angle = np.abs(phi) < 1e-8
        large_angle = ~small_angle
        
        # Small angle approximation
        V_inv[small_angle] = np.eye(2)[None, :, :]
        
        # Large angle
        if np.any(large_angle):
            half_phi = 0.5 * phi[large_angle]
            cot_half = 1. / np.tan(half_phi)
            
            V_inv[large_angle, 0, 0] = half_phi * cot_half
            V_inv[large_angle, 1, 1] = half_phi * cot_half
            V_inv[large_angle, 0, 1] = half_phi
            V_inv[large_angle, 1, 0] = -half_phi
        
        return V_inv

class RobotState:
    """Robot state representation"""
    
    def __init__(self, x: float = 0., y: float = 0., theta: float = 0.):
        self.Rot = SO2.exp(theta)
        self.p = np.array([x, y])
    
    @property
    def theta(self):
        return SO2.log(self.Rot)
    
    def to_array(self) -> np.ndarray:
        """Convert to [x, y, theta] array"""
        return np.array([self.p[0], self.p[1], self.theta])
    
    def to_SE2(self) -> np.ndarray:
        """Convert to SE(2) matrix"""
        T = np.eye(3)
        T[:2, :2] = self.Rot
        T[:2, 2] = self.p
        return T

class UKFManifold:
    """Unscented Kalman Filter on SE(2) manifold"""
    
    def __init__(self, initial_state: RobotState, P0: np.ndarray, Q: np.ndarray, 
                 R: np.ndarray, alpha: float = 1e-3, beta: float = 2., kappa: float = 0.,
                 manifold_type: str = 'left_SE2'):
        """
        Initialize UKF
        
        Args:
            initial_state: Initial robot state
            P0: Initial covariance matrix (3x3)
            Q: Process noise covariance (3x3)
            R: Measurement noise covariance (2x2 for GPS-like measurements)
            alpha: Sigma point spread parameter
            beta: Prior knowledge parameter (2 for Gaussian)
            kappa: Secondary scaling parameter
            manifold_type: 'left_SE2', 'right_SE2', or 'SO2xR2'
        """
        self.state = initial_state
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        
        # UKF parameters
        self.n = 3  # State dimension
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Compute sigma point parameters
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.n_sigma = 2 * self.n + 1
        
        # Sigma point weights
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = self.Wc[i] = 1 / (2 * (self.n + self.lambda_))
        
        # Manifold operations
        self.manifold_type = manifold_type
        self._setup_manifold_operations()
    
    def _setup_manifold_operations(self):
        """Setup retraction and inverse retraction operations"""
        if self.manifold_type == 'left_SE2':
            self.phi = self._left_phi
            self.phi_inv = self._left_phi_inv
        elif self.manifold_type == 'right_SE2':
            self.phi = self._right_phi
            self.phi_inv = self._right_phi_inv
        else:  # SO2xR2
            self.phi = self._so2r2_phi
            self.phi_inv = self._so2r2_phi_inv
    
    def _left_phi(self, state: RobotState, xi: np.ndarray) -> RobotState:
        """Left SE(2) retraction"""
        T_state = state.to_SE2()
        T_xi = SE2.exp(xi)
        T_new = T_xi @ T_state
        
        new_state = RobotState()
        new_state.Rot = T_new[:2, :2]
        new_state.p = T_new[:2, 2]
        return new_state
    
    def _left_phi_inv(self, state1: RobotState, state2: RobotState) -> np.ndarray:
        """Left SE(2) inverse retraction"""
        T1 = state1.to_SE2()
        T2 = state2.to_SE2()
        T_diff = np.linalg.inv(T1) @ T2
        return SE2.log(T_diff)
    
    def _right_phi(self, state: RobotState, xi: np.ndarray) -> RobotState:
        """Right SE(2) retraction"""
        T_state = state.to_SE2()
        T_xi = SE2.exp(xi)
        T_new = T_state @ T_xi
        
        new_state = RobotState()
        new_state.Rot = T_new[:2, :2]
        new_state.p = T_new[:2, 2]
        return new_state
    
    def _right_phi_inv(self, state1: RobotState, state2: RobotState) -> np.ndarray:
        """Right SE(2) inverse retraction"""
        T1 = state1.to_SE2()
        T2 = state2.to_SE2()
        T_diff = T1 @ np.linalg.inv(T2)
        return SE2.log(T_diff)
    
    def _so2r2_phi(self, state: RobotState, xi: np.ndarray) -> RobotState:
        """SO(2) x R^2 retraction"""
        new_state = RobotState()
        new_state.Rot = state.Rot @ SO2.exp(xi[2])
        new_state.p = state.p + xi[:2]
        return new_state
    
    def _so2r2_phi_inv(self, state1: RobotState, state2: RobotState) -> np.ndarray:
        """SO(2) x R^2 inverse retraction"""
        xi = np.zeros(3)
        xi[:2] = state2.p - state1.p
        xi[2] = SO2.log(state1.Rot.T @ state2.Rot)
        return xi
    
    def _generate_sigma_points(self) -> List[RobotState]:
        """Generate sigma points"""
        try:
            sqrt_matrix = cholesky((self.n + self.lambda_) * self.P, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh((self.n + self.lambda_) * self.P)
            eigenvals = np.maximum(eigenvals, 1e-12)
            sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        sigma_points = []
        
        # Central point
        sigma_points.append(self.state)
        
        # Positive perturbations
        for i in range(self.n):
            xi = sqrt_matrix[:, i]
            sigma_points.append(self.phi(self.state, xi))
        
        # Negative perturbations
        for i in range(self.n):
            xi = -sqrt_matrix[:, i]
            sigma_points.append(self.phi(self.state, xi))
        
        return sigma_points
    
    def _compute_mean_and_covariance(self, sigma_points: List[RobotState], 
                                   weights_m: np.ndarray, weights_c: np.ndarray) -> Tuple[RobotState, np.ndarray]:
        """Compute mean and covariance from sigma points"""
        # Initialize mean as first sigma point
        mean_state = sigma_points[0]
        
        # Iterative mean computation
        for _ in range(10):  # Max iterations
            xi_sum = np.zeros(3)
            for i, sp in enumerate(sigma_points):
                xi_i = self.phi_inv(mean_state, sp)
                xi_sum += weights_m[i] * xi_i
            
            if np.linalg.norm(xi_sum) < 1e-6:
                break
                
            mean_state = self.phi(mean_state, xi_sum)
        
        # Compute covariance
        P = np.zeros((3, 3))
        for i, sp in enumerate(sigma_points):
            xi_i = self.phi_inv(mean_state, sp)
            P += weights_c[i] * np.outer(xi_i, xi_i)
        
        return mean_state, P
    
    def predict(self, u: np.ndarray, dt: float):
        """Prediction step"""
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Propagate sigma points through motion model
        predicted_points = []
        for sp in sigma_points:
            pred_sp = self._motion_model(sp, u, dt)
            predicted_points.append(pred_sp)
        
        # Compute predicted mean and covariance
        self.state, P_pred = self._compute_mean_and_covariance(
            predicted_points, self.Wm, self.Wc)
        
        # Add process noise
        self.P = P_pred + self.Q
    
    def update(self, z: np.ndarray):
        """Update step with measurement"""
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Predict measurements
        predicted_measurements = []
        for sp in sigma_points:
            z_pred = self._measurement_model(sp)
            predicted_measurements.append(z_pred)
        
        # Compute predicted measurement mean and covariance
        z_pred_mean = np.zeros_like(z)
        for i, z_pred in enumerate(predicted_measurements):
            z_pred_mean += self.Wm[i] * z_pred
        
        S = np.zeros((len(z), len(z)))
        Pxz = np.zeros((3, len(z)))
        
        for i, (sp, z_pred) in enumerate(zip(sigma_points, predicted_measurements)):
            z_diff = z_pred - z_pred_mean
            S += self.Wc[i] * np.outer(z_diff, z_diff)
            
            xi_i = self.phi_inv(self.state, sp)
            Pxz += self.Wc[i] * np.outer(xi_i, z_diff)
        
        S += self.R
        
        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            K = Pxz @ np.linalg.pinv(S)
        
        # Innovation
        innovation = z - z_pred_mean
        
        # Update state and covariance
        xi_update = K @ innovation
        self.state = self.phi(self.state, xi_update)
        self.P = self.P - K @ S @ K.T
    
    def _motion_model(self, state: RobotState, u: np.ndarray, dt: float) -> RobotState:
        """Motion model: [v, omega] control input"""
        v, omega = u
        
        new_state = RobotState()
        
        if abs(omega) < 1e-6:
            # Straight line motion
            displacement = v * dt * np.array([np.cos(state.theta), np.sin(state.theta)])
            new_state.p = state.p + displacement
            new_state.Rot = state.Rot
        else:
            # Circular motion
            dtheta = omega * dt
            new_state.Rot = state.Rot @ SO2.exp(dtheta)
            
            # Position update
            dx = v / omega * (np.sin(dtheta))
            dy = v / omega * (1 - np.cos(dtheta))
            local_displacement = np.array([dx, dy])
            global_displacement = state.Rot @ local_displacement
            new_state.p = state.p + global_displacement
        
        return new_state
    
    def _measurement_model(self, state: RobotState) -> np.ndarray:
        """Measurement model: returns [x, y] position"""
        return state.p.copy()

class BagDataProcessor:
    """Process ROS2 bag database files"""
    
    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.conn = sqlite3.connect(bag_path)
        self.topics = self._get_topics()
    
    def _get_topics(self) -> Dict:
        """Get available topics from bag"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {}
        for row in cursor.fetchall():
            topics[row[1]] = {'id': row[0], 'type': row[2]}
        return topics
    
    def extract_odometry_data(self) -> pd.DataFrame:
        """Extract odometry data"""
        if '/odom' not in self.topics:
            raise ValueError("No /odom topic found in bag file")
        
        topic_id = self.topics['/odom']['id']
        cursor = self.conn.cursor()
        
        query = """
        SELECT timestamp, data 
        FROM messages 
        WHERE topic_id = ? 
        ORDER BY timestamp
        """
        
        cursor.execute(query, (topic_id,))
        
        odom_data = []
        for row in cursor.fetchall():
            timestamp = row[0] * 1e-9  # Convert to seconds
            # This is a simplified parser - you might need to use rosbag2_py for proper deserialization
            # For now, we'll assume the data format or use a placeholder
            odom_data.append({
                'timestamp': timestamp,
                'x': 0.0,  # These would need proper message parsing
                'y': 0.0,
                'theta': 0.0,
                'v': 0.0,
                'omega': 0.0
            })
        
        return pd.DataFrame(odom_data)
    
    def extract_scan_data(self) -> pd.DataFrame:
        """Extract laser scan data"""
        if '/scan' not in self.topics:
            raise ValueError("No /scan topic found in bag file")
        
        # Similar implementation as odometry
        # Return placeholder for now
        return pd.DataFrame()

class GroundTruthProcessor:
    """Process ground truth CSV data"""
    
    def __init__(self, gt_path: str):
        self.gt_path = gt_path
        self.data = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load ground truth data"""
        df = pd.read_csv(self.gt_path)
        
        # Convert quaternion to euler angles
        if 'qx' in df.columns:
            quats = df[['qx', 'qy', 'qz', 'qw']].values
            eulers = []
            
            for q in quats:
                r = R.from_quat(q)
                euler = r.as_euler('xyz', degrees=False)
                eulers.append(euler[2])  # Yaw angle
            
            df['theta'] = eulers
        
        return df
    
    def interpolate_at_timestamps(self, timestamps: np.ndarray) -> pd.DataFrame:
        """Interpolate ground truth at given timestamps"""
        gt_interp = pd.DataFrame()
        gt_interp['timestamp'] = timestamps
        
        for col in ['x', 'y', 'theta']:
            if col in self.data.columns:
                gt_interp[col] = np.interp(timestamps, self.data['time'], self.data[col])
        
        return gt_interp

class MapProcessor:
    """Process occupancy grid maps"""
    
    def __init__(self, map_path: str):
        self.map_path = map_path
        self.map_yaml = self._load_yaml()
        self.map_image = self._load_image()
    
    def _load_yaml(self) -> Dict:
        """Load map YAML file"""
        yaml_path = Path(self.map_path)
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_image(self) -> np.ndarray:
        """Load map image"""
        yaml_dir = Path(self.map_path).parent
        image_path = yaml_dir / self.map_yaml['image']
        return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to map coordinates"""
        resolution = self.map_yaml['resolution']
        origin = self.map_yaml['origin']
        
        map_x = int((x - origin[0]) / resolution)
        map_y = int((y - origin[1]) / resolution)
        
        return map_x, map_y
    
    def is_valid_position(self, x: float, y: float) -> bool:
        """Check if position is in free space"""
        map_x, map_y = self.world_to_map(x, y)
        
        if (0 <= map_x < self.map_image.shape[1] and 
            0 <= map_y < self.map_image.shape[0]):
            # 0 = obstacle, 255 = free space, 127 = unknown
            return self.map_image[map_y, map_x] > 127
        
        return False

class LocalizationBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, bag_path: str, map_path: str, gt_path: str):
        self.bag_processor = BagDataProcessor(bag_path)
        self.map_processor = MapProcessor(map_path)
        self.gt_processor = GroundTruthProcessor(gt_path)
        
        # Initialize filters
        self.filters = {}
        self._setup_filters()
        
        # Results storage
        self.results = {}
    
    def _setup_filters(self):
        """Setup different UKF variants"""
        # Initial state (will be updated from data)
        initial_state = RobotState(0, 0, 0)
        
        # Covariance matrices
        P0 = np.diag([0.1, 0.1, 0.1])  # Initial uncertainty
        Q = np.diag([0.01, 0.01, 0.01])  # Process noise
        R = np.diag([0.1, 0.1])  # Measurement noise (GPS-like)
        
        # Different manifold representations
        self.filters['left_SE2'] = UKFManifold(
            initial_state, P0, Q, R, manifold_type='left_SE2')
        self.filters['right_SE2'] = UKFManifold(
            initial_state, P0, Q, R, manifold_type='right_SE2')
        self.filters['SO2xR2'] = UKFManifold(
            initial_state, P0, Q, R, manifold_type='SO2xR2')
    
    def run_benchmark(self) -> Dict:
        """Run the complete benchmark"""
        print("Starting UKF Localization Benchmark...")
        
        # Extract data
        print("Extracting odometry data...")
        odom_data = self.bag_processor.extract_odometry_data()
        
        if len(odom_data) == 0:
            print("Warning: No odometry data found. Using simulated data.")
            odom_data = self._generate_simulated_data()
        
        # Get ground truth at odometry timestamps
        print("Interpolating ground truth...")
        gt_data = self.gt_processor.interpolate_at_timestamps(odom_data['timestamp'].values)
        
        # Run filters
        results = {}
        for filter_name, ukf in self.filters.items():
            print(f"Running {filter_name} UKF...")
            estimates = self._run_filter(ukf, odom_data)
            results[filter_name] = estimates
        
        # Compute metrics
        print("Computing metrics...")
        metrics = self._compute_metrics(results, gt_data)
        
        # Generate plots
        print("Generating plots...")
        self._generate_plots(results, gt_data, odom_data, metrics)
        
        return metrics
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        """Generate simulated odometry data for testing"""
        print("Generating simulated trajectory data...")
        
        # Circular trajectory parameters
        T = 40  # Total time
        dt = 0.01  # Time step
        radius = 5  # Radius of circle
        angular_vel = 2 * np.pi / T  # Complete one circle
        
        timestamps = np.arange(0, T, dt)
        data = []
        
        x, y, theta = 0, radius, 0  # Start at (0, radius)
        v = radius * angular_vel  # Linear velocity
        
        for t in timestamps:
            # Add some noise to control inputs
            v_noisy = v + np.random.normal(0, 0.01)
            omega_noisy = angular_vel + np.random.normal(0, 0.01)
            
            data.append({
                'timestamp': t,
                'x': x,
                'y': y,
                'theta': theta,
                'v': v_noisy,
                'omega': omega_noisy
            })
            
            # Update true position
            theta += angular_vel * dt
            x = radius * np.sin(theta)
            y = radius * np.cos(theta)
        
        return pd.DataFrame(data)
    
    def _run_filter(self, ukf: UKFManifold, odom_data: pd.DataFrame) -> pd.DataFrame:
        """Run a single filter on the data"""
        estimates = []
        
        # Initialize with first odometry reading
        if len(odom_data) > 0:
            first_row = odom_data.iloc[0]
            ukf.state = RobotState(first_row['x'], first_row['y'], first_row['theta'])
        
        for i, row in odom_data.iterrows():
            if i == 0:
                # First iteration
                estimates.append({
                    'timestamp': row['timestamp'],
                    'x': ukf.state.p[0],
                    'y': ukf.state.p[1],
                    'theta': ukf.state.theta,
                    'P_trace': np.trace(ukf.P)
                })
                continue
            
            # Control input
            u = np.array([row['v'], row['omega']])
            dt = row['timestamp'] - odom_data.iloc[i-1]['timestamp']
            
            # Predict step
            ukf.predict(u, dt)
            
            # Simulated GPS measurement (every 1
            # seconds for realistic scenario)
            if i % 100 == 0:  # Every 100 odometry readings
                # Simulated GPS measurement with noise
                gps_measurement = np.array([
                    row['x'] + np.random.normal(0, 0.1),
                    row['y'] + np.random.normal(0, 0.1)
                ])
                ukf.update(gps_measurement)
            
            estimates.append({
                'timestamp': row['timestamp'],
                'x': ukf.state.p[0],
                'y': ukf.state.p[1],
                'theta': ukf.state.theta,
                'P_trace': np.trace(ukf.P)
            })
        
        return pd.DataFrame(estimates)
    
    def _compute_metrics(self, results: Dict, gt_data: pd.DataFrame) -> Dict:
        """Compute performance metrics"""
        metrics = {}
        
        for filter_name, estimates in results.items():
            # Ensure same length for comparison
            min_len = min(len(estimates), len(gt_data))
            
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            # Position RMSE
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            pos_rmse = np.sqrt(np.mean(pos_errors**2))
            
            # Orientation RMSE
            angle_errors = np.abs(est_subset['theta'] - gt_subset['theta'])
            # Handle angle wrapping
            angle_errors = np.minimum(angle_errors, 2*np.pi - angle_errors)
            angle_rmse = np.sqrt(np.mean(angle_errors**2))
            
            # Mean absolute errors
            mae_x = np.mean(np.abs(est_subset['x'] - gt_subset['x']))
            mae_y = np.mean(np.abs(est_subset['y'] - gt_subset['y']))
            mae_theta = np.mean(angle_errors)
            
            # Maximum errors
            max_pos_error = np.max(pos_errors)
            max_angle_error = np.max(angle_errors)
            
            # Final position error
            final_pos_error = pos_errors.iloc[-1] if len(pos_errors) > 0 else 0
            
            metrics[filter_name] = {
                'position_rmse': pos_rmse,
                'orientation_rmse': angle_rmse,
                'mae_x': mae_x,
                'mae_y': mae_y,
                'mae_theta': mae_theta,
                'max_position_error': max_pos_error,
                'max_orientation_error': max_angle_error,
                'final_position_error': final_pos_error,
                'mean_uncertainty': np.mean(est_subset['P_trace'])
            }
        
        return metrics
    
    def _generate_plots(self, results: Dict, gt_data: pd.DataFrame, 
                       odom_data: pd.DataFrame, metrics: Dict):
        """Generate comprehensive plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Trajectory comparison
        ax1 = plt.subplot(2, 3, 1)
        
        # Plot ground truth
        plt.plot(gt_data['x'], gt_data['y'], 'k-', linewidth=2, 
                label='Ground Truth', alpha=0.8)
        
        # Plot odometry
        plt.plot(odom_data['x'], odom_data['y'], 'r--', linewidth=1, 
                label='Odometry', alpha=0.7)
        
        # Plot filter estimates
        colors = ['blue', 'green', 'orange']
        for i, (filter_name, estimates) in enumerate(results.items()):
            plt.plot(estimates['x'], estimates['y'], 
                    color=colors[i], linewidth=1.5,
                    label=f'{filter_name} UKF', alpha=0.8)
        
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Trajectory Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 2. Position errors over time
        ax2 = plt.subplot(2, 3, 2)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            
            plt.plot(est_subset['timestamp'], pos_errors, 
                    color=colors[i], linewidth=1.5, 
                    label=f'{filter_name} (RMSE: {metrics[filter_name]["position_rmse"]:.3f}m)')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Position Error [m]')
        plt.title('Position Error vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Orientation errors over time
        ax3 = plt.subplot(2, 3, 3)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            angle_errors = np.abs(est_subset['theta'] - gt_subset['theta'])
            angle_errors = np.minimum(angle_errors, 2*np.pi - angle_errors)
            angle_errors = np.degrees(angle_errors)
            
            plt.plot(est_subset['timestamp'], angle_errors, 
                    color=colors[i], linewidth=1.5,
                    label=f'{filter_name} (RMSE: {np.degrees(metrics[filter_name]["orientation_rmse"]):.1f}°)')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Orientation Error [degrees]')
        plt.title('Orientation Error vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Uncertainty evolution
        ax4 = plt.subplot(2, 3, 4)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            plt.plot(estimates['timestamp'], estimates['P_trace'], 
                    color=colors[i], linewidth=1.5, label=filter_name)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Trace of Covariance Matrix')
        plt.title('Uncertainty Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 5. RMSE comparison bar chart
        ax5 = plt.subplot(2, 3, 5)
        
        filter_names = list(metrics.keys())
        pos_rmse_values = [metrics[name]['position_rmse'] for name in filter_names]
        
        bars = plt.bar(filter_names, pos_rmse_values, color=colors[:len(filter_names)], alpha=0.7)
        plt.xlabel('Filter Type')
        plt.ylabel('Position RMSE [m]')
        plt.title('Position RMSE Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, pos_rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Metrics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Metric', 'left_SE2', 'right_SE2', 'SO2xR2']
        
        metric_names = [
            'Position RMSE [m]',
            'Orientation RMSE [deg]',
            'Max Position Error [m]',
            'Final Position Error [m]',
            'Mean Uncertainty'
        ]
        
        metric_keys = [
            'position_rmse',
            'orientation_rmse',
            'max_position_error', 
            'final_position_error',
            'mean_uncertainty'
        ]
        
        for name, key in zip(metric_names, metric_keys):
            row = [name]
            for filter_name in filter_names:
                value = metrics[filter_name][key]
                if key == 'orientation_rmse':
                    value = np.degrees(value)
                row.append(f'{value:.3f}')
            table_data.append(row)
        
        table = ax6.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = int(time.time())
        plot_filename = f'ukf_localization_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Results plot saved as: {plot_filename}")
        
        plt.show()
        
        # Print summary
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict):
        """Print performance summary"""
        print("\n" + "="*70)
        print("UKF LOCALIZATION BENCHMARK RESULTS")
        print("="*70)
        
        for filter_name, metric in metrics.items():
            print(f"\n{filter_name} UKF:")
            print(f"  Position RMSE:      {metric['position_rmse']:.4f} m")
            print(f"  Orientation RMSE:   {np.degrees(metric['orientation_rmse']):.2f} degrees")
            print(f"  Max Position Error: {metric['max_position_error']:.4f} m")
            print(f"  Final Position Error: {metric['final_position_error']:.4f} m")
            print(f"  Mean Uncertainty:   {metric['mean_uncertainty']:.6f}")
        
        # Find best performing filter
        best_filter = min(metrics.keys(), key=lambda x: metrics[x]['position_rmse'])
        print(f"\nBest performing filter: {best_filter}")
        print(f"Best Position RMSE: {metrics[best_filter]['position_rmse']:.4f} m")


class ROS2BagParser:
    """Enhanced ROS2 bag parser for TurtleBot 4 data"""
    
    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.conn = sqlite3.connect(bag_path)
        self.topics = self._get_topics()
    
    def _get_topics(self) -> Dict:
        """Get available topics from bag"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {}
        for row in cursor.fetchall():
            topics[row[1]] = {'id': row[0], 'type': row[2]}
        return topics
    
    def extract_odometry_data(self) -> pd.DataFrame:
        """Extract and parse odometry data from /odom topic"""
        if '/odom' not in self.topics:
            print("Warning: No /odom topic found. Available topics:", list(self.topics.keys()))
            return pd.DataFrame()
        
        topic_id = self.topics['/odom']['id']
        cursor = self.conn.cursor()
        
        query = """
        SELECT timestamp, data 
        FROM messages 
        WHERE topic_id = ? 
        ORDER BY timestamp
        """
        
        cursor.execute(query, (topic_id,))
        rows = cursor.fetchall()
        
        if not rows:
            print("No odometry data found in bag file")
            return pd.DataFrame()
        
        odom_data = []
        prev_time = None
        
        for row in rows:
            timestamp = row[0] * 1e-9  # Convert to seconds
            data = row[1]
            
            try:
                # Parse odometry message (simplified CDR parsing)
                parsed_data = self._parse_odometry_message(data)
                
                # Calculate linear and angular velocities
                v = np.sqrt(parsed_data['twist_linear_x']**2 + parsed_data['twist_linear_y']**2)
                omega = parsed_data['twist_angular_z']
                
                odom_data.append({
                    'timestamp': timestamp,
                    'x': parsed_data['pose_x'],
                    'y': parsed_data['pose_y'],
                    'z': parsed_data['pose_z'],
                    'qx': parsed_data['pose_qx'],
                    'qy': parsed_data['pose_qy'], 
                    'qz': parsed_data['pose_qz'],
                    'qw': parsed_data['pose_qw'],
                    'v': v,
                    'omega': omega,
                    'theta': self._quaternion_to_yaw(
                        parsed_data['pose_qx'], parsed_data['pose_qy'], 
                        parsed_data['pose_qz'], parsed_data['pose_qw'])
                })
                
            except Exception as e:
                print(f"Warning: Could not parse odometry message: {e}")
                continue
        
        if not odom_data:
            print("No valid odometry data could be parsed")
            return pd.DataFrame()
        
        print(f"Extracted {len(odom_data)} odometry messages")
        return pd.DataFrame(odom_data)
    
    def _parse_odometry_message(self, data: bytes) -> Dict:
        """Parse odometry message from CDR format (simplified)"""
        # This is a simplified parser. In practice, you might want to use
        # proper ROS2 message deserialization libraries
        
        import struct
        
        try:
            # CDR header (4 bytes)
            offset = 4
            
            # Skip frame_id and child_frame_id strings (simplified)
            # In real implementation, you'd properly parse string lengths
            offset += 32  # Approximate offset for header fields
            
            # Parse pose position (3 * 8 bytes for double)
            pose_x, pose_y, pose_z = struct.unpack_from('<3d', data, offset)
            offset += 24
            
            # Parse pose orientation quaternion (4 * 8 bytes for double) 
            pose_qx, pose_qy, pose_qz, pose_qw = struct.unpack_from('<4d', data, offset)
            offset += 32
            
            # Skip pose covariance (36 * 8 bytes)
            offset += 288
            
            # Parse twist linear (3 * 8 bytes for double)
            twist_linear_x, twist_linear_y, twist_linear_z = struct.unpack_from('<3d', data, offset)
            offset += 24
            
            # Parse twist angular (3 * 8 bytes for double)
            twist_angular_x, twist_angular_y, twist_angular_z = struct.unpack_from('<3d', data, offset)
            
            return {
                'pose_x': pose_x,
                'pose_y': pose_y,
                'pose_z': pose_z,
                'pose_qx': pose_qx,
                'pose_qy': pose_qy,
                'pose_qz': pose_qz,
                'pose_qw': pose_qw,
                'twist_linear_x': twist_linear_x,
                'twist_linear_y': twist_linear_y,
                'twist_linear_z': twist_linear_z,
                'twist_angular_x': twist_angular_x,
                'twist_angular_y': twist_angular_y,
                'twist_angular_z': twist_angular_z
            }
            
        except Exception as e:
            # Fallback: return zeros if parsing fails
            print(f"CDR parsing failed: {e}. Using fallback values.")
            return {
                'pose_x': 0.0, 'pose_y': 0.0, 'pose_z': 0.0,
                'pose_qx': 0.0, 'pose_qy': 0.0, 'pose_qz': 0.0, 'pose_qw': 1.0,
                'twist_linear_x': 0.0, 'twist_linear_y': 0.0, 'twist_linear_z': 0.0,
                'twist_angular_x': 0.0, 'twist_angular_y': 0.0, 'twist_angular_z': 0.0
            }
    
    def _quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """Convert quaternion to yaw angle"""
        # Yaw calculation from quaternion
        return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))


class EnhancedLocalizationBenchmark:
    """Enhanced benchmarking class with proper data handling"""
    
    def __init__(self, bag_path: str, map_path: str, gt_path: str):
        self.bag_parser = ROS2BagParser(bag_path)
        self.map_processor = MapProcessor(map_path) if map_path else None
        self.gt_processor = GroundTruthProcessor(gt_path)
        
        # Initialize filters
        self.filters = {}
        self._setup_filters()
        
        # Results storage
        self.results = {}
    
    def _setup_filters(self):
        """Setup different UKF variants with tuned parameters"""
        # Initial state (will be updated from data)
        initial_state = RobotState(0, 0, 0)
        
        # Tuned covariance matrices for TurtleBot 4
        P0 = np.diag([0.01, 0.01, 0.01])  # Initial uncertainty
        
        # Process noise - adjusted for different velocities
        Q = np.diag([0.001, 0.001, 0.001])  # Process noise
        
        # Measurement noise - GPS/localization system
        R = np.diag([0.05, 0.05])  # Measurement noise
        
        # UKF parameters
        alpha = 1e-3
        beta = 2.0
        kappa = 0.0
        
        # Different manifold representations
        self.filters['left_SE2'] = UKFManifold(
            initial_state, P0, Q, R, alpha, beta, kappa, manifold_type='left_SE2')
        self.filters['right_SE2'] = UKFManifold(
            initial_state, P0, Q, R, alpha, beta, kappa, manifold_type='right_SE2')
        self.filters['SO2xR2'] = UKFManifold(
            initial_state, P0, Q, R, alpha, beta, kappa, manifold_type='SO2xR2')
    
    def run_benchmark(self) -> Dict:
        """Run the complete benchmark with real data"""
        print("Starting Enhanced UKF Localization Benchmark...")
        
        # Extract odometry data
        print("Extracting odometry data from bag file...")
        odom_data = self.bag_parser.extract_odometry_data()
        
        if len(odom_data) == 0:
            raise ValueError("No odometry data could be extracted from bag file")
        
        print(f"Successfully extracted {len(odom_data)} odometry measurements")
        print(f"Time range: {odom_data['timestamp'].min():.2f} - {odom_data['timestamp'].max():.2f} seconds")
        print(f"Average velocity: {odom_data['v'].mean():.3f} m/s")
        print(f"Max velocity: {odom_data['v'].max():.3f} m/s")
        
        # Get ground truth at odometry timestamps
        print("Interpolating ground truth data...")
        gt_data = self.gt_processor.interpolate_at_timestamps(odom_data['timestamp'].values)
        
        print(f"Ground truth interpolated for {len(gt_data)} points")
        
        # Run filters
        results = {}
        for filter_name, ukf in self.filters.items():
            print(f"Running {filter_name} UKF...")
            start_time = time.time()
            estimates = self._run_filter(ukf, odom_data)
            end_time = time.time()
            results[filter_name] = estimates
            print(f"  Completed in {end_time - start_time:.2f} seconds")
        
        # Compute metrics
        print("Computing performance metrics...")
        metrics = self._compute_detailed_metrics(results, gt_data, odom_data)
        
        # Generate comprehensive plots
        print("Generating comprehensive plots...")
        self._generate_comprehensive_plots(results, gt_data, odom_data, metrics)
        
        return metrics
    
    def _run_filter(self, ukf: UKFManifold, odom_data: pd.DataFrame) -> pd.DataFrame:
        """Run filter with adaptive measurement updates"""
        estimates = []
        
        # Initialize with first odometry reading
        first_row = odom_data.iloc[0]
        ukf.state = RobotState(first_row['x'], first_row['y'], first_row['theta'])
        
        measurement_counter = 0
        measurement_interval = 50  # Update every 50 odometry readings (adjust as needed)
        
        for i, row in odom_data.iterrows():
            if i == 0:
                # First iteration - just record initial state
                estimates.append({
                    'timestamp': row['timestamp'],
                    'x': ukf.state.p[0],
                    'y': ukf.state.p[1],
                    'theta': ukf.state.theta,
                    'P_xx': ukf.P[0, 0],
                    'P_yy': ukf.P[1, 1],
                    'P_tt': ukf.P[2, 2],
                    'P_trace': np.trace(ukf.P)
                })
                continue
            
            # Control input
            u = np.array([row['v'], row['omega']])
            dt = row['timestamp'] - odom_data.iloc[i-1]['timestamp']
            
            # Prediction step
            if dt > 0 and dt < 1.0:  # Sanity check for dt
                ukf.predict(u, dt)
            
            # Measurement update (simulated GPS/localization)
            measurement_counter += 1
            if measurement_counter >= measurement_interval:
                # Add some noise to simulate real measurement system
                measurement_noise = np.random.multivariate_normal([0, 0], ukf.R)
                gps_measurement = np.array([row['x'], row['y']]) + measurement_noise
                ukf.update(gps_measurement)
                measurement_counter = 0
            
            # Record estimate
            estimates.append({
                'timestamp': row['timestamp'],
                'x': ukf.state.p[0],
                'y': ukf.state.p[1],
                'theta': ukf.state.theta,
                'P_xx': ukf.P[0, 0],
                'P_yy': ukf.P[1, 1], 
                'P_tt': ukf.P[2, 2],
                'P_trace': np.trace(ukf.P)
            })
        
        return pd.DataFrame(estimates)
    
    def _compute_detailed_metrics(self, results: Dict, gt_data: pd.DataFrame, 
                                 odom_data: pd.DataFrame) -> Dict:
        """Compute comprehensive performance metrics"""
        metrics = {}
        
        for filter_name, estimates in results.items():
            # Ensure same length for comparison
            min_len = min(len(estimates), len(gt_data))
            
            est_subset = estimates.iloc[:min_len].copy()
            gt_subset = gt_data.iloc[:min_len].copy()
            odom_subset = odom_data.iloc[:min_len].copy()
            
            # Position errors
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            
            # Orientation errors
            angle_errors = np.abs(est_subset['theta'] - gt_subset['theta'])
            # Handle angle wrapping
            angle_errors = np.minimum(angle_errors, 2*np.pi - angle_errors)
            
            # Component-wise errors
            x_errors = np.abs(est_subset['x'] - gt_subset['x'])
            y_errors = np.abs(est_subset['y'] - gt_subset['y'])
            
            # Velocity-dependent analysis
            low_vel_mask = odom_subset['v'] < 0.1  # Stationary/slow
            med_vel_mask = (odom_subset['v'] >= 0.1) & (odom_subset['v'] < 0.3)
            high_vel_mask = odom_subset['v'] >= 0.3
            
            # Time-based analysis
            total_time = est_subset['timestamp'].max() - est_subset['timestamp'].min()
            total_distance = np.sum(odom_subset['v'] * np.diff(np.append([0], odom_subset['timestamp'])))
            
            metrics[filter_name] = {
                # Overall RMSE
                'position_rmse': np.sqrt(np.mean(pos_errors**2)),
                'orientation_rmse': np.sqrt(np.mean(angle_errors**2)),
                'x_rmse': np.sqrt(np.mean(x_errors**2)),
                'y_rmse': np.sqrt(np.mean(y_errors**2)),
                
                # Mean absolute errors
                'mae_position': np.mean(pos_errors),
                'mae_x': np.mean(x_errors),
                'mae_y': np.mean(y_errors),
                'mae_theta': np.mean(angle_errors),
                
                # Maximum errors
                'max_position_error': np.max(pos_errors),
                'max_x_error': np.max(x_errors),
                'max_y_error': np.max(y_errors),
                'max_orientation_error': np.max(angle_errors),
                
                # Final errors
                'final_position_error': pos_errors.iloc[-1] if len(pos_errors) > 0 else 0,
                'final_x_error': x_errors.iloc[-1] if len(x_errors) > 0 else 0,
                'final_y_error': y_errors.iloc[-1] if len(y_errors) > 0 else 0,
                'final_orientation_error': angle_errors.iloc[-1] if len(angle_errors) > 0 else 0,
                
                # Velocity-dependent RMSE
                'low_vel_pos_rmse': np.sqrt(np.mean(pos_errors[low_vel_mask]**2)) if np.any(low_vel_mask) else 0,
                'med_vel_pos_rmse': np.sqrt(np.mean(pos_errors[med_vel_mask]**2)) if np.any(med_vel_mask) else 0,
                'high_vel_pos_rmse': np.sqrt(np.mean(pos_errors[high_vel_mask]**2)) if np.any(high_vel_mask) else 0,
                
                # Uncertainty metrics
                'mean_uncertainty_x': np.mean(est_subset['P_xx']),
                'mean_uncertainty_y': np.mean(est_subset['P_yy']),
                'mean_uncertainty_theta': np.mean(est_subset['P_tt']),
                'mean_uncertainty_trace': np.mean(est_subset['P_trace']),
                
                # Performance relative to distance traveled
                'rmse_per_meter': np.sqrt(np.mean(pos_errors**2)) / max(total_distance, 1e-6),
                'rmse_per_second': np.sqrt(np.mean(pos_errors**2)) / max(total_time, 1e-6),
                
                # Statistical measures
                'std_position_error': np.std(pos_errors),
                'percentile_95_pos_error': np.percentile(pos_errors, 95),
                'percentile_99_pos_error': np.percentile(pos_errors, 99),
                
                # Additional metrics
                'total_time': total_time,
                'total_distance': total_distance,
                'average_velocity': np.mean(odom_subset['v']),
                'max_velocity': np.max(odom_subset['v'])
            }
        
        return metrics
    
    def _generate_comprehensive_plots(self, results: Dict, gt_data: pd.DataFrame, 
                                    odom_data: pd.DataFrame, metrics: Dict):
        """Generate comprehensive visualization"""
        fig = plt.figure(figsize=(24, 18))
        colors = ['blue', 'green', 'orange']
        
        # 1. Trajectory comparison with map
        ax1 = plt.subplot(3, 4, 1)
        
        # Plot ground truth
        plt.plot(gt_data['x'], gt_data['y'], 'k-', linewidth=3, 
                label='Ground Truth', alpha=0.9)
        
        # Plot odometry
        plt.plot(odom_data['x'], odom_data['y'], 'r:', linewidth=2, 
                label='Odometry', alpha=0.7)
        
        # Plot filter estimates
        for i, (filter_name, estimates) in enumerate(results.items()):
            plt.plot(estimates['x'], estimates['y'], 
                    color=colors[i], linewidth=2,
                    label=f'{filter_name}', alpha=0.8)
        
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Trajectory Comparison')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 2. Position errors over time
        ax2 = plt.subplot(3, 4, 2)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            
            plt.plot(est_subset['timestamp'], pos_errors, 
                    color=colors[i], linewidth=2, 
                    label=f'{filter_name} (RMSE: {metrics[filter_name]["position_rmse"]:.3f}m)')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Position Error [m]')
        plt.title('Position Error vs Time')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 3. Orientation errors over time
        ax3 = plt.subplot(3, 4, 3)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            angle_errors = np.abs(est_subset['theta'] - gt_subset['theta'])
            angle_errors = np.minimum(angle_errors, 2*np.pi - angle_errors)
            angle_errors = np.degrees(angle_errors)
            
            plt.plot(est_subset['timestamp'], angle_errors, 
                    color=colors[i], linewidth=2,
                    label=f'{filter_name} (RMSE: {np.degrees(metrics[filter_name]["orientation_rmse"]):.1f}°)')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Orientation Error [degrees]')
        plt.title('Orientation Error vs Time')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 4. Velocity vs Position RMSE
        ax4 = plt.subplot(3, 4, 4)
        
        # Create velocity bins for analysis
        velocities = odom_data['v'].values
        vel_bins = np.linspace(0, np.max(velocities), 20)
        bin_centers = (vel_bins[:-1] + vel_bins[1:]) / 2
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data), len(odom_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            vel_subset = velocities[:min_len]
            
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            
            # Compute RMSE for each velocity bin
            bin_rmse = []
            for j in range(len(vel_bins)-1):
                mask = (vel_subset >= vel_bins[j]) & (vel_subset < vel_bins[j+1])
                if np.sum(mask) > 0:
                    bin_rmse.append(np.sqrt(np.mean(pos_errors[mask]**2)))
                else:
                    bin_rmse.append(0)
            
            plt.plot(bin_centers, bin_rmse, 
                    color=colors[i], linewidth=2, marker='o',
                    label=f'{filter_name}')
        
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Position RMSE [m]')
        plt.title('RMSE vs Velocity')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 5. Uncertainty evolution
        ax5 = plt.subplot(3, 4, 5)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            plt.plot(estimates['timestamp'], estimates['P_trace'], 
                    color=colors[i], linewidth=2, label=filter_name)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Trace of Covariance Matrix')
        plt.title('Uncertainty Evolution')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 6. Error distribution histogram
        ax6 = plt.subplot(3, 4, 6)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            
            plt.hist(pos_errors, bins=30, alpha=0.5, 
                    color=colors[i], label=filter_name, density=True)
        
        plt.xlabel('Position Error [m]')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 7. RMSE comparison bar chart
        ax7 = plt.subplot(3, 4, 7)
        
        filter_names = list(metrics.keys())
        pos_rmse_values = [metrics[name]['position_rmse'] for name in filter_names]
        
        bars = plt.bar(filter_names, pos_rmse_values, color=colors[:len(filter_names)], alpha=0.7)
        plt.xlabel('Filter Type')
        plt.ylabel('Position RMSE [m]')
        plt.title('Position RMSE Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, pos_rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 8. Component-wise RMSE
        ax8 = plt.subplot(3, 4, 8)
        
        x_rmse_values = [metrics[name]['x_rmse'] for name in filter_names]
        y_rmse_values = [metrics[name]['y_rmse'] for name in filter_names]
        
        x = np.arange(len(filter_names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, x_rmse_values, width, 
                       label='X RMSE', color='lightblue', alpha=0.7)
        bars2 = plt.bar(x + width/2, y_rmse_values, width,
                       label='Y RMSE', color='lightcoral', alpha=0.7)
        
        plt.xlabel('Filter Type')
        plt.ylabel('RMSE [m]')
        plt.title('Component-wise RMSE')
        plt.xticks(x, filter_names, rotation=45)
        plt.legend(fontsize=9)
        
        # 9. Cumulative error plot
        ax9 = plt.subplot(3, 4, 9)
        
        for i, (filter_name, estimates) in enumerate(results.items()):
            min_len = min(len(estimates), len(gt_data))
            est_subset = estimates.iloc[:min_len]
            gt_subset = gt_data.iloc[:min_len]
            
            pos_errors = np.sqrt((est_subset['x'] - gt_subset['x'])**2 + 
                               (est_subset['y'] - gt_subset['y'])**2)
            
            cumulative_error = np.cumsum(pos_errors)
            
            plt.plot(est_subset['timestamp'], cumulative_error, 
                    color=colors[i], linewidth=2, label=filter_name)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Cumulative Error [m]')
        plt.title('Cumulative Position Error')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 10. Velocity profile
        ax10 = plt.subplot(3, 4, 10)
        
        plt.plot(odom_data['timestamp'], odom_data['v'], 'b-', linewidth=2, label='Linear Velocity')
        plt.plot(odom_data['timestamp'], np.abs(odom_data['omega']), 'r-', linewidth=2, label='Angular Velocity')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity')
        plt.title('Robot Velocity Profile')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 11. Path length comparison
        ax11 = plt.subplot(3, 4, 11)
        
        # Calculate path lengths
        path_lengths = {}
        
        # Ground truth path length
        gt_diffs = np.diff(gt_data[['x', 'y']].values, axis=0)
        gt_path_length = np.sum(np.linalg.norm(gt_diffs, axis=1))
        path_lengths['Ground Truth'] = gt_path_length
        
        # Odometry path length
        odom_diffs = np.diff(odom_data[['x', 'y']].values, axis=0)
        odom_path_length = np.sum(np.linalg.norm(odom_diffs, axis=1))
        path_lengths['Odometry'] = odom_path_length
        
        # Filter path lengths
        for filter_name, estimates in results.items():
            est_diffs = np.diff(estimates[['x', 'y']].values, axis=0)
            est_path_length = np.sum(np.linalg.norm(est_diffs, axis=1))
            path_lengths[filter_name] = est_path_length
        
        methods = list(path_lengths.keys())
        lengths = list(path_lengths.values())
        colors_extended = ['black', 'red'] + colors[:len(results)]
        
        bars = plt.bar(methods, lengths, color=colors_extended, alpha=0.7)
        plt.xlabel('Method')
        plt.ylabel('Path Length [m]')
        plt.title('Total Path Length Comparison')
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, lengths):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 12. Detailed metrics table
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('tight')
        ax12.axis('off')
        
        # Create comprehensive table data
        table_data = []
        headers = ['Metric'] + list(filter_names)
        
        metric_rows = [
            ('Position RMSE [m]', 'position_rmse', '{:.3f}'),
            ('Orientation RMSE [deg]', 'orientation_rmse', '{:.1f}'),
            ('Max Position Error [m]', 'max_position_error', '{:.3f}'),
            ('Final Position Error [m]', 'final_position_error', '{:.3f}'),
            ('Low Vel RMSE [m]', 'low_vel_pos_rmse', '{:.3f}'),
            ('High Vel RMSE [m]', 'high_vel_pos_rmse', '{:.3f}'),
            ('95th Percentile Error [m]', 'percentile_95_pos_error', '{:.3f}'),
            ('Mean Uncertainty', 'mean_uncertainty_trace', '{:.4f}'),
            ('RMSE per meter [m/m]', 'rmse_per_meter', '{:.4f}')
        ]
        
        for metric_name, metric_key, format_str in metric_rows:
            row = [metric_name]
            for filter_name in filter_names:
                value = metrics[filter_name][metric_key]
                if metric_key == 'orientation_rmse':
                    value = np.degrees(value)
                row.append(format_str.format(value))
            table_data.append(row)
        
        table = ax12.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)
        
        # Color code best performance
        for i in range(1, len(table_data) + 1):
            values = [float(table_data[i-1][j]) for j in range(1, len(headers))]
            best_idx = np.argmin(values) + 1
            table[(i, best_idx)].set_facecolor('#90EE90')  # Light green
        
        plt.tight_layout(pad=2.0)
        
        # Save comprehensive plot
        timestamp = int(time.time())
        plot_filename = f'comprehensive_ukf_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comprehensive results plot saved as: {plot_filename}")
        
        plt.show()
        
        # Print detailed summary
        self._print_detailed_summary(metrics, odom_data)
    
    def _print_detailed_summary(self, metrics: Dict, odom_data: pd.DataFrame):
        """Print comprehensive performance summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE UKF LOCALIZATION BENCHMARK RESULTS")
        print("="*80)
        
        # Dataset summary
        print(f"\nDataset Summary:")
        print(f"  Total time:         {odom_data['timestamp'].max() - odom_data['timestamp'].min():.1f} seconds")
        print(f"  Total data points:  {len(odom_data)}")
        print(f"  Average velocity:   {odom_data['v'].mean():.3f} m/s")
        print(f"  Max velocity:       {odom_data['v'].max():.3f} m/s")
        print(f"  Average ang. vel:   {np.abs(odom_data['omega']).mean():.3f} rad/s")
        
        # Performance comparison
        print(f"\nPerformance Comparison:")
        print(f"{'Filter':<15} {'Pos RMSE':<10} {'Ori RMSE':<10} {'Max Err':<10} {'Final Err':<10}")
        print("-" * 65)
        
        for filter_name, metric in metrics.items():
            print(f"{filter_name:<15} "
                  f"{metric['position_rmse']:<10.3f} "
                  f"{np.degrees(metric['orientation_rmse']):<10.1f} "
                  f"{metric['max_position_error']:<10.3f} "
                  f"{metric['final_position_error']:<10.3f}")
        
        # Velocity-dependent analysis
        print(f"\nVelocity-Dependent Analysis:")
        print(f"{'Filter':<15} {'Low Vel':<10} {'Med Vel':<10} {'High Vel':<10}")
        print("-" * 55)
        
        for filter_name, metric in metrics.items():
            print(f"{filter_name:<15} "
                  f"{metric['low_vel_pos_rmse']:<10.3f} "
                  f"{metric['med_vel_pos_rmse']:<10.3f} "
                  f"{metric['high_vel_pos_rmse']:<10.3f}")
        
        # Statistical analysis
        print(f"\nStatistical Analysis:")
        print(f"{'Filter':<15} {'Std Dev':<10} {'95th %ile':<10} {'99th %ile':<10}")
        print("-" * 55)
        
        for filter_name, metric in metrics.items():
            print(f"{filter_name:<15} "
                  f"{metric['std_position_error']:<10.3f} "
                  f"{metric['percentile_95_pos_error']:<10.3f} "
                  f"{metric['percentile_99_pos_error']:<10.3f}")
        
        # Find best performing filter overall
        best_filter = min(metrics.keys(), key=lambda x: metrics[x]['position_rmse'])
        print(f"\n🏆 Best Overall Performance: {best_filter}")
        print(f"   Position RMSE: {metrics[best_filter]['position_rmse']:.3f} m")
        print(f"   Orientation RMSE: {np.degrees(metrics[best_filter]['orientation_rmse']):.1f}°")
        
        # Efficiency metrics
        print(f"\nEfficiency Metrics:")
        for filter_name, metric in metrics.items():
            print(f"  {filter_name}:")
            print(f"    RMSE per meter traveled: {metric['rmse_per_meter']:.4f}")
            print(f"    RMSE per second:         {metric['rmse_per_second']:.4f}")


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='UKF Localization Benchmark')
    parser.add_argument('--bag', required=True, help='Path to ROS2 bag file (.db3)')
    parser.add_argument('--map', help='Path to map YAML file')
    parser.add_argument('--groundtruth', required=True, help='Path to ground truth CSV file')
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = EnhancedLocalizationBenchmark(
            bag_path=args.bag,
            map_path=args.map,
            gt_path=args.groundtruth
        )
        
        # Run benchmark
        metrics = benchmark.run_benchmark()
        
        print("Benchmark completed successfully!")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()