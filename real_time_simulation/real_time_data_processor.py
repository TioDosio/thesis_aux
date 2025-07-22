#!/usr/bin/env python3
"""
Real-Time Data Processor for MonoTransmotion Model
This script processes ROS topic data in real-time and prepares it for trajectory prediction.
Based on summarize_data.py but adapted for streaming real-time data.
"""

import numpy as np
import json
import torch
import time
import math
from collections import deque, defaultdict
from pyquaternion import Quaternion
try:
    from pytorch3d.transforms import quaternion_to_matrix
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False

# Camera intrinsic matrix (Vizzy robot camera)
VIZZY_CAMERA_K = torch.tensor([[335.491, 0, 329.763],
                               [0, 376.188, 239.821],
                               [0, 0, 1]], dtype=torch.float32)

# Model configuration - Must match model's expected parameters
SEQ_LEN = 10   # Must match model config (not 5)
INTERVAL = 5   # Reduced from 15
OBS_LEN = 4    # Must match model config (not 3)
PRED_LEN = 6

# Keypoint order
CORRECT_ORDER = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
]


def quaternion_yaw(q, in_image_frame=True):
    """Calculate yaw angle from quaternion"""
    if isinstance(q, dict):
        # Convert from dict format {q1, q2, q3, q4} -> Quaternion(w, x, y, z)
        quat = Quaternion(q['q4'], q['q1'], q['q2'], q['q3'])
    else:
        quat = q
    
    if in_image_frame:
        v = np.dot(quat.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(quat.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
    return float(yaw)


def quaternion_to_matrix_manual(quaternions):
    """Manual quaternion to rotation matrix conversion if pytorch3d not available"""
    if PYTORCH3D_AVAILABLE:
        return quaternion_to_matrix(quaternions)
    
    # Manual implementation for [w, x, y, z] format
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    # Normalize quaternions
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrix
    R = torch.zeros((*quaternions.shape[:-1], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    R[..., 0, 0] = 1 - 2*(yy + zz)
    R[..., 0, 1] = 2*(xy - wz)
    R[..., 0, 2] = 2*(xz + wy)
    R[..., 1, 0] = 2*(xy + wz)
    R[..., 1, 1] = 1 - 2*(xx + zz)
    R[..., 1, 2] = 2*(yz - wx)
    R[..., 2, 0] = 2*(xz - wy)
    R[..., 2, 1] = 2*(yz + wx)
    R[..., 2, 2] = 1 - 2*(xx + yy)
    
    return R


def local2global(traj_estimated, ego_pose):
    """Transform local coordinates to global coordinates"""
    # Ego translation
    ego_translation = ego_pose[:, :, 4:]
    # Ego rotation (quaternion format: [q1, q2, q3, q4] -> [x, y, z, w])
    ego_quaternion = ego_pose[:, :, :4]
    
    # Convert to [w, x, y, z] format for quaternion_to_matrix
    ego_quaternion_wxyz = torch.stack([
        ego_quaternion[:, :, 3],  # w
        ego_quaternion[:, :, 0],  # x  
        ego_quaternion[:, :, 1],  # y
        ego_quaternion[:, :, 2]   # z
    ], dim=-1)
    
    # Quaternion to rotation matrix
    ego_rotation_matrix = quaternion_to_matrix_manual(ego_quaternion_wxyz)
    
    # Reshape traj_estimated for matrix multiplication
    traj_estimated = traj_estimated.unsqueeze(-1)
    
    # Transform to global coordinates
    traj_estimated = torch.matmul(ego_rotation_matrix, traj_estimated)
    traj_estimated = traj_estimated.squeeze(-1) + ego_translation
    
    return traj_estimated


def extract_keypoints_from_person_dict(person_dict):
    """Extract keypoints from person dictionary in correct order"""
    #print(f"DEBUG: Extracting keypoints from person_dict with keys: {list(person_dict.keys())}")
    
    # Create mapping from part_id to coordinates
    part_map = {}
    
    # From the examples in image_detections.txt, the keypoints are stored as:
    # body_parts with part_id as string names like "Nose", "RShoulder", etc.
    body_parts = person_dict.get('body_parts', [])
    #print(f"DEBUG: Found {len(body_parts)} body parts")
    
    for body_part in body_parts:
        part_id = body_part['part_id']  # This is a string like "Nose", "RShoulder"
        part_map[part_id] = [
            float(body_part['x']), 
            float(body_part['y']), 
            float(body_part['confidence'])
        ]
        #print(f"  {part_id}: ({body_part['x']}, {body_part['y']}, conf={body_part['confidence']:.3f})")
    
    #print(f"DEBUG: Mapped {len(part_map)} keypoints: {list(part_map.keys())}")
    
    # Map from the message format to our expected format
    # The message format uses different names than our CORRECT_ORDER
    keypoint_mapping = {
        "Nose": "Nose",
        "REye": "REye", 
        "LEye": "LEye",
        "REar": "REar",
        "LEar": "LEar", 
        "RShoulder": "RShoulder",
        "LShoulder": "LShoulder",
        "RElbow": "RElbow",
        "LElbow": "LElbow",
        "RWrist": "RWrist", 
        "LWrist": "LWrist",
        "RHip": "RHip",
        "LHip": "LHip",
        "RKnee": "RKnee",
        "LKnee": "LKnee",
        "RAnkle": "RAnkle",
        "LAnkle": "LAnkle"
    }
    
    # Build keypoints in correct order
    keypoints = []
    for expected_name in CORRECT_ORDER:
        if expected_name in keypoint_mapping and keypoint_mapping[expected_name] in part_map:
            keypoints.extend(part_map[keypoint_mapping[expected_name]])
        else:
            keypoints.extend([0.0, 0.0, 0.0])
    
    #print(f"DEBUG: Final keypoints array length: {len(keypoints)} (expected {len(CORRECT_ORDER)*3})")
    return keypoints


def compute_bbox_from_keypoints(keypoints):
    """Compute bounding box from keypoints"""
    xs, ys = [], []
    
    # Extract x, y coordinates from keypoints with confidence > 0
    for i in range(0, len(keypoints), 3):
        x, y, confidence = keypoints[i:i+3]
        if confidence > 0:
            xs.append(x)
            ys.append(y)
    
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    
    return [min(xs), min(ys), max(xs), max(ys)]


def project_3d_simple(box_obj, kk):
    """Simple 3D to 2D projection"""
    # Extract 3D center
    x, y, z = box_obj[0], box_obj[1], box_obj[2]
    
    # Project to 2D using camera intrinsics
    u = (x * kk[0, 0] / z) + kk[0, 2]
    v = (y * kk[1, 1] / z) + kk[1, 2]
    
    # Use reasonable box dimensions for projection
    w, l, h = box_obj[3], box_obj[4], box_obj[5]
    
    # Simple 2D box estimation
    box_2d = [u - w/2, v - h/2, u + w/2, v + h/2]
    return box_2d


def extract_ground_truth_simple(box_obj, kk, spherical=True):
    """Simplified version of extract_ground_truth for real-time processing"""
    # 2D projection
    boxes_2d = project_3d_simple(box_obj, kk)
    
    # 3D box
    boxes_3d = box_obj[:6]
    
    # Angle processing
    yaw = box_obj[6] if len(box_obj) > 6 else 0.0
    
    # Ensure yaw is in valid range
    while yaw > math.pi:
        yaw -= 2 * math.pi
    while yaw < -math.pi:
        yaw += 2 * math.pi
    
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    
    # Height, width, length
    hwl = [box_obj[5], box_obj[3], box_obj[4]]  # [h, w, l]
    
    # Spherical coordinates
    xyz = list(box_obj[:3])
    dd = np.linalg.norm(box_obj[:3])
    
    if spherical:
        # Convert to spherical coordinates
        r = dd
        theta = math.atan2(xyz[1], xyz[0])  # azimuth
        phi = math.atan2(xyz[2], math.sqrt(xyz[0]**2 + xyz[1]**2))  # elevation
        loc = [theta, phi, xyz[2], r]  # [theta, psi, z, r]
    else:
        loc = xyz + [dd]
    
    # Final output format
    output = loc + hwl + [sin_yaw, cos_yaw, yaw]
    
    return boxes_2d, boxes_3d, output


def preprocess_monoloco_simple(keypoints, kk):
    """Simplified preprocessing for MonoLoco format"""
    # Reshape keypoints from flat list to (3, 17) format
    kps = np.array(keypoints).reshape(3, 17)  # [x, y, conf] for 17 keypoints
    
    # Normalize by image dimensions (assuming 640x480 or similar)
    kps[0] /= 640.0  # x coordinates
    kps[1] /= 480.0  # y coordinates
    
    # Convert to tensor and flatten
    kps_tensor = torch.tensor(kps, dtype=torch.float32)
    return kps_tensor.view(-1)  # Flatten to 1D


class RealTimeDataProcessor:
    """Real-time data processor for MonoTransmotion model"""
    
    def __init__(self):
        # Data storage with sequence length limits
        self.ego_poses = deque(maxlen=SEQ_LEN * INTERVAL * 2)  # Extra buffer
        self.person_data = defaultdict(lambda: deque(maxlen=SEQ_LEN * INTERVAL * 2))
        self.frame_count = 0
        
        # Track person appearances
        self.person_frame_tracker = defaultdict(list)
        
    def add_ego_pose(self, ego_data):
        """Add ego pose data"""
        self.ego_poses.append({
            'frame': self.frame_count,
            'timestamp': ego_data.get('timestamp', time.time()),
            'x': ego_data['x'],
            'y': ego_data['y'],
            'z': ego_data['z'],
            'q1': ego_data['q1'],
            'q2': ego_data['q2'],
            'q3': ego_data['q3'],
            'q4': ego_data['q4']
        })
    
    def add_person_detection(self, person_dict, timestamp):
        """Add person detection data"""
        person_id = person_dict.get('id', 0)
        #print(f"DEBUG: Adding person detection for ID: {person_id}")
        
        # Extract keypoints
        keypoints = extract_keypoints_from_person_dict(person_dict)
        #print(f"DEBUG: Extracted {len(keypoints)//3} keypoints, total length: {len(keypoints)}")
        
        # Compute bounding box
        bbox = compute_bbox_from_keypoints(keypoints)
        #print(f"DEBUG: Computed bbox: {bbox}")
        
        # Estimate 3D position (simplified)
        bbox_height = bbox[3] - bbox[1] if bbox[3] > bbox[1] else 100
        bbox_width = bbox[2] - bbox[0] if bbox[2] > bbox[0] else 50
        
        # Simple depth estimation
        if bbox_height > 0:
            estimated_depth = max(1.5, min(8.0, 1.7 * VIZZY_CAMERA_K[1,1] / bbox_height))
        else:
            estimated_depth = 3.0
        
        # Convert to world coordinates (simplified)
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        
        world_x = (x_center - VIZZY_CAMERA_K[0,2]) * estimated_depth / VIZZY_CAMERA_K[0,0]
        world_y = (y_center - VIZZY_CAMERA_K[1,2]) * estimated_depth / VIZZY_CAMERA_K[1,1]
        
        person_data = {
            'frame': self.frame_count,
            'timestamp': timestamp,
            'id': person_id,
            'x': float(estimated_depth),
            'y': float(-world_x),
            'z': float(-world_y + 1.2),  # Add camera height
            'keypoints': keypoints,
            'bbox': bbox
        }
        
        self.person_data[person_id].append(person_data)
        self.person_frame_tracker[person_id].append(self.frame_count)
        #print(f"DEBUG: Person {person_id} now has {len(self.person_data[person_id])} detections")
    
    def can_generate_sequence(self, person_id):
        """Check if we can generate a sequence for a person"""
        if person_id not in self.person_data:
            return False
        
        person_frames = self.person_frame_tracker[person_id]
        #print(f"DEBUG: Person {person_id} has {len(person_frames)} frames: {person_frames}")
        
        # Minimum requirement: OBS_LEN frames (4 frames for observation)
        if len(person_frames) < OBS_LEN:
            #print(f"DEBUG: Person {person_id} needs minimum {OBS_LEN} frames, has {len(person_frames)}")
            return False
        
        # Use up to SEQ_LEN frames, but at least OBS_LEN frames
        frames_to_use = min(len(person_frames), SEQ_LEN)
        recent_frames = person_frames[-frames_to_use:]
        
        #print(f"DEBUG: Person {person_id} using {len(recent_frames)} frames out of {len(person_frames)} available")
        
        # Check frame intervals (should be roughly INTERVAL apart)
        if len(recent_frames) > 1:
            for i in range(1, len(recent_frames)):
                frame_diff = recent_frames[i] - recent_frames[i-1]
                if frame_diff > INTERVAL * 3:  # Allow more tolerance for minimum sequences
                    #print(f"DEBUG: Person {person_id} frame gap too large: {frame_diff} > {INTERVAL * 3}")
                    return False
        
        #print(f"DEBUG: Person {person_id} CAN generate sequence with {len(recent_frames)} frames!")
        return True
    
    def generate_model_input(self, person_id):
        """Generate input data for MonoTransmotion model"""
        if not self.can_generate_sequence(person_id):
            return None
        
        # Determine how many frames we actually have and can use
        available_person_frames = len(self.person_data[person_id])
        available_ego_frames = len(self.ego_poses)
        
        # Use up to SEQ_LEN frames, but at least OBS_LEN frames
        frames_to_use = min(available_person_frames, available_ego_frames, SEQ_LEN)
        frames_to_use = max(frames_to_use, OBS_LEN)  # Ensure minimum
        
        # Get recent data
        person_history = list(self.person_data[person_id])[-frames_to_use:]
        ego_history = list(self.ego_poses)[-frames_to_use:]
        
        #print(f"DEBUG: Using {frames_to_use} frames for person {person_id} (available: person={available_person_frames}, ego={available_ego_frames})")
        
        if len(person_history) < OBS_LEN or len(ego_history) < OBS_LEN:
            #print(f"DEBUG: Insufficient data after selection: person={len(person_history)}, ego={len(ego_history)}")
            return None
        
        # Prepare model input tensors
        X_seq = []
        Y_seq = []
        kps_seq = []
        boxes_3d_seq = []
        boxes_2d_seq = []
        ego_pose_seq = []
        
        # Process available frames
        for i in range(len(person_history)):
            person_frame = person_history[i]
            ego_frame = ego_history[i]
            
            # 2D center from bounding box
            bbox = person_frame['bbox']
            x_2d = (bbox[0] + bbox[2]) / 2
            y_2d = (bbox[1] + bbox[3]) / 2
            X_seq.append([x_2d, y_2d])
            
            # Keypoints
            keypoints = person_frame['keypoints']
            
            # Reshape keypoints to (3, 17) format for model
            kps_reshaped = []
            for j in range(3):  # x, y, confidence
                kps_reshaped.append([keypoints[k*3 + j] for k in range(17)])
            kps_seq.append(kps_reshaped)
            
            # 3D box with reasonable pedestrian dimensions
            yaw = quaternion_yaw(ego_frame)
            box_3d = [
                person_frame['x'], person_frame['y'], person_frame['z'],
                0.6, 0.6, 1.7,  # width, length, height for pedestrian
                yaw
            ]
            boxes_3d_seq.append(box_3d)
            
            # 2D box and Y (ground truth processing)
            try:
                boxes_2d, boxes_3d_gt, y_output = extract_ground_truth_simple(box_3d, VIZZY_CAMERA_K)
                boxes_2d_seq.append(boxes_2d)
                
                # Preprocess for Y
                preprocessed = preprocess_monoloco_simple(keypoints, VIZZY_CAMERA_K)
                Y_seq.append(preprocessed.tolist())
                
            except Exception as e:
                #print(f"Warning: Failed to process ground truth for frame {i}: {e}")
                boxes_2d_seq.append([x_2d-25, y_2d-50, x_2d+25, y_2d+50])
                Y_seq.append([0.0] * 34)  # Default size for preprocessed data
            
            # Ego pose [q1, q2, q3, q4, x, y, z]
            ego_pose_seq.append([
                ego_frame['q1'], ego_frame['q2'], ego_frame['q3'], ego_frame['q4'],
                ego_frame['x'], ego_frame['y'], ego_frame['z']
            ])
        
        # Pad sequences to SEQ_LEN if we have fewer frames
        while len(X_seq) < SEQ_LEN:
            # Repeat the last frame for padding
            last_idx = len(X_seq) - 1
            X_seq.append(X_seq[last_idx].copy())
            Y_seq.append(Y_seq[last_idx].copy())
            kps_seq.append([row.copy() for row in kps_seq[last_idx]])
            boxes_3d_seq.append(boxes_3d_seq[last_idx].copy())
            boxes_2d_seq.append(boxes_2d_seq[last_idx].copy())
            ego_pose_seq.append(ego_pose_seq[last_idx].copy())
            
        #print(f"DEBUG: Generated sequences of length {len(X_seq)} (padded from {frames_to_use} actual frames)")
        
        # Convert to tensors
        model_input = {
            'X': torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 2]
            'Y': torch.tensor(Y_seq, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, features]
            'kps': torch.tensor(kps_seq, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 3, 17]
            'boxes_3d': torch.tensor([box[:3] for box in boxes_3d_seq], dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 3]
            'boxes_2d': torch.tensor(boxes_2d_seq, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 4]
            'ego_pose': torch.tensor(ego_pose_seq, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 7]
            'K': VIZZY_CAMERA_K.unsqueeze(0).unsqueeze(0).repeat(1, SEQ_LEN, 1, 1),  # [1, seq_len, 3, 3]
        }
        
        return model_input
    
    def update_frame_count(self):
        """Update frame counter"""
        self.frame_count += 1
    
    def get_active_persons(self):
        """Get list of persons that can generate sequences"""
        return [person_id for person_id in self.person_data.keys() 
                if self.can_generate_sequence(person_id)]


# Utility functions for message processing
def process_tf_message_dict(tf_msg_dict):
    """Process TF message dictionary to extract ego pose from odom frame"""
    transforms = tf_msg_dict.get('transforms', [])
    
    # Look for the last odom transform (most recent in the message)
    odom_transform = None
    for transform in transforms:
        if transform.get('header', {}).get('frame_id') == 'odom':
            odom_transform = transform
    
    if odom_transform:
        translation = odom_transform['transform']['translation']
        rotation = odom_transform['transform']['rotation']
        
        # Calculate yaw angle for trajectory processing
        quat = {
            'q1': rotation['x'],
            'q2': rotation['y'], 
            'q3': rotation['z'],
            'q4': rotation['w']
        }
        yaw = quaternion_yaw(quat, in_image_frame=False)
        
        return {
            'x': translation['x'],
            'y': translation['y'],  # Keep original coordinate system
            'z': translation['z'],
            'q1': rotation['x'],
            'q2': rotation['y'],
            'q3': rotation['z'],
            'q4': rotation['w'],
            'yaw': yaw,
            'frame_id': odom_transform.get('header', {}).get('frame_id', 'odom'),
            'child_frame_id': odom_transform.get('child_frame_id', 'unknown'),
            'timestamp': time.time()
        }
    
    return None


def process_image_detections_dict(detections_dict):
    """Process image detections dictionary"""
    persons = detections_dict.get('persons', [])
    processed_persons = []
    
    for person in persons:
        processed_persons.append({
            'id': person.get('id', 0),
            'id_confidence': person.get('id_confidence', 0.5),
            'body_parts': person.get('body_parts', [])
        })
    
    return processed_persons


def process_raw_bodies_dict(bodies_dict):
    """Process raw bodies dictionary to extract pose data"""
    poses = bodies_dict.get('poses', [])
    processed_poses = []
    
    for i, pose in enumerate(poses):
        position = pose['position']
        processed_poses.append({
            'id': i,
            'x': position['x'],
            'y': position['y'],
            'z': position['z'],
            'timestamp': time.time()
        })
    
    return processed_poses


if __name__ == "__main__":
    # Example usage
    processor = RealTimeDataProcessor()
    
    #print("Real-Time Data Processor initialized")
    #print(f"Sequence length: {SEQ_LEN}")
    #print(f"Observation length: {OBS_LEN}")
    #print(f"Prediction length: {PRED_LEN}")
    #print(f"Frame interval: {INTERVAL}")
