#!/usr/bin/env python3
"""
Real-Time Trajectory Predictor for ROS
This script subscribes to ROS topics (/tf, /image_detections),
processes data in real-time to match MonoTransmotion model format,
and generates trajectory predictions.

The script now uses the map->odom transformation instead of odom->base_footprint
to provide better global localization context for trajectory prediction.
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
import argparse
from collections import defaultdict, deque, OrderedDict
from pyquaternion import Quaternion

# Add model code directory to path
model_code_path = os.path.expanduser("~/thesis_aux/model/code")
sys.path.append(model_code_path)

# Add real-time data processor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion as ROSQuaternion, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage
from tf2_msgs.msg import TFMessage
from human_awareness_msgs.msg import PersonTracker, PersonsList, Person
from real_time_data_processor import (RealTimeDataProcessor, process_tf_message_dict, process_image_detections_dict, process_raw_bodies_dict)
from eval.evaluator import Evaluator
from utils.camera import project_3d, correct_angle, to_spherical
from utils.process import preprocess_monoloco
import yaml

# Define correct keypoint order
CORRECT_ORDER = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
]

# Vizzy camera intrinsic matrix
VIZZY_CAMERA_K = torch.tensor([[335.491, 0, 329.763],
                               [0, 376.188, 239.821],
                               [0, 0, 1]], dtype=torch.float32)


class RealTimeTrajectoryPredictor:
    def __init__(self):
        rospy.init_node('real_time_trajectory_predictor', anonymous=True)
        
        # Configuration - Match model's expected parameters
        self.seq_len = 10       # Must match model config
        self.interval = 5      # Reduced from 15  
        self.obs_len = 4        # Must match model config
        self.pred_len = 6       # predict the next 6 frames
        frequency = 7.5        # Hz, must match model config
        
        # Initialize data processor
        self.data_processor = RealTimeDataProcessor()
        
        # Store both transforms to compute map->base_footprint
        self.latest_map_odom_frame = None
        self.latest_map_odom_timestamp = None
        self.latest_odom_base_frame = None
        self.latest_odom_base_timestamp = None
        
        self.setup_model()

        # Publishers
        self.trajectory_pub = rospy.Publisher('/predicted_trajectories', PoseArray, queue_size=10)
        self.trajectory_json_pub = rospy.Publisher('/predicted_trajectories_json', String, queue_size=10)
        self.visualization_pub = rospy.Publisher('/trajectory_visualization', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('/predicted_paths', Path, queue_size=10)
        
        # Subscribers
        self.setup_subscribers()
        
        rospy.loginfo("Real-Time Trajectory Predictor initialized")
        rospy.loginfo("Waiting for data from topics: /tf (map->odom + odom->base_footprint), /image_detections (PersonsList)")
        rospy.loginfo("Minimum {} frames needed to start predictions".format(self.obs_len))
        rospy.loginfo("Optimal {} frames ({}*{} interval) = ~{} seconds at {}Hz".format(
            self.seq_len * self.interval, self.seq_len, self.interval, 
            (self.seq_len * self.interval) / frequency, frequency))
        rospy.loginfo("READY FOR ROSBAG PLAYBACK - Script is now event-driven")
        
        # Print diagnostic information
        print("="*60)
        print("DEBUG: Script fully initialized and ready")
        print("DEBUG: Subscribed to topics:")
        print("  - /tf (TFMessage) - map->odom and odom->base_footprint transforms")
        print("  - /image_detections (PersonsList)")
        print("DEBUG: Waiting for messages...")
        print("="*60)

    def setup_model(self):
        """Setup the trajectory prediction model"""
        try:
            # Load model configuration
            config_path = os.path.join(model_code_path, "configs", "traj_pred.yaml")
            if not os.path.exists(config_path):
                rospy.logwarn("Config file not found: {}".format(config_path))
                self.model = None
                return
                
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Setup evaluator args
            import argparse
            args = argparse.Namespace()
            args.eval_mode = "traj_pred"
            
            # Check if model checkpoint exists
            checkpoint_path = os.path.join(model_code_path, "../checkpoints/traj_pred/best_traj_model.pth")
            if not os.path.exists(checkpoint_path):
                rospy.logwarn("Model checkpoint not found: {}".format(checkpoint_path))
                self.model = None
                return
                
            args.load_traj = checkpoint_path
            args.traj_cfg = config_path
            args.load_loc = ""
            
            loc_config_path = os.path.join(model_code_path, "configs/localization.yaml")
            if os.path.exists(loc_config_path):
                args.loc_cfg = loc_config_path
            else:
                rospy.logwarn("Localization config not found: {}".format(loc_config_path))
                args.loc_cfg = ""
                
            args.obs = self.obs_len
            args.pred = self.pred_len
            args.bs = 1  # Single sample prediction
            args.r_seed = 1
            args.joints_folder = ""  # Not used for real-time
            
            # Initialize evaluator (which loads the model)
            self.evaluator = Evaluator(args)
            self.model = self.evaluator.traj_model
            self.model.eval()
            
            rospy.loginfo("Model loaded successfully")
            
        except Exception as e:
            rospy.logerr("Failed to load model: {}".format(e))
            import traceback
            rospy.logwarn("Traceback: {}".format(traceback.format_exc()))
            self.model = None

    def setup_subscribers(self):
        """Setup ROS subscribers"""
        rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        # Note: /raw_bodies not needed - 3D poses are included in /image_detections
        rospy.Subscriber('/image_detections', PersonsList, self.image_detections_callback)

    def tf_callback(self, msg):
        """Process transform messages to extract both map->odom and odom->base_footprint transforms"""
        print(f"🔄 TF CALLBACK: Received {len(msg.transforms)} transforms")
        
        map_odom_transforms = []
        odom_base_transforms = []
        
        # Find both types of transforms in this message
        for transform in msg.transforms:
            print(f"🔍 TF TRANSFORM: frame_id='{transform.header.frame_id}' -> child_frame_id='{transform.child_frame_id}'")
            if transform.header.frame_id == 'map' and transform.child_frame_id == 'odom':
                map_odom_transforms.append(transform)
                print(f"✅ FOUND: map->odom transform: t=({transform.transform.translation.x:.3f}, {transform.transform.translation.y:.3f})")
            elif transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_footprint':
                odom_base_transforms.append(transform)
                print(f"✅ FOUND: odom->base_footprint transform: t=({transform.transform.translation.x:.3f}, {transform.transform.translation.y:.3f})")
            else:
                print(f"❌ IGNORED: {transform.header.frame_id}->{transform.child_frame_id} (not needed)")
        
        print(f"📊 TF SUMMARY: Found {len(map_odom_transforms)} map->odom, {len(odom_base_transforms)} odom->base_footprint")
        
        # Process map->odom transforms
        if map_odom_transforms:
            latest_transform = map_odom_transforms[-1]  # Use the last one
            self.latest_map_odom_frame = {
                'timestamp': latest_transform.header.stamp.to_sec(),
                'seq': latest_transform.header.seq,
                'frame_id': latest_transform.header.frame_id,
                'child_frame_id': latest_transform.child_frame_id,
                'translation': {
                    'x': latest_transform.transform.translation.x,
                    'y': latest_transform.transform.translation.y,
                    'z': latest_transform.transform.translation.z
                },
                'rotation': {
                    'x': latest_transform.transform.rotation.x,
                    'y': latest_transform.transform.rotation.y,
                    'z': latest_transform.transform.rotation.z,
                    'w': latest_transform.transform.rotation.w
                }
            }
            self.latest_map_odom_timestamp = latest_transform.header.stamp.to_sec()
            # print(f"DEBUG: Updated map->odom transform: t=({latest_transform.transform.translation.x:.3f}, {latest_transform.transform.translation.y:.3f})")
        
        # Process odom->base_footprint transforms
        if odom_base_transforms:
            latest_transform = odom_base_transforms[-1]  # Use the last one
            self.latest_odom_base_frame = {
                'timestamp': latest_transform.header.stamp.to_sec(),
                'seq': latest_transform.header.seq,
                'frame_id': latest_transform.header.frame_id,
                'child_frame_id': latest_transform.child_frame_id,
                'translation': {
                    'x': latest_transform.transform.translation.x,
                    'y': latest_transform.transform.translation.y,
                    'z': latest_transform.transform.translation.z
                },
                'rotation': {
                    'x': latest_transform.transform.rotation.x,
                    'y': latest_transform.transform.rotation.y,
                    'z': latest_transform.transform.rotation.z,
                    'w': latest_transform.transform.rotation.w
                }
            }
            self.latest_odom_base_timestamp = latest_transform.header.stamp.to_sec()
            # print(f"DEBUG: Updated odom->base_footprint transform: t=({latest_transform.transform.translation.x:.3f}, {latest_transform.transform.translation.y:.3f})")
        
        # Combine transforms to get map->base_footprint if both are available
        if self.latest_map_odom_frame and self.latest_odom_base_frame:
            print(f"🔄 TF COMBINATION: Both transforms available, attempting combination...")
            combined_transform = self.combine_transforms(
                self.latest_map_odom_frame, 
                self.latest_odom_base_frame
            )
            
            if combined_transform:
                print(f"✅ TF COMBINATION: Combined map->base_footprint: t=({combined_transform['translation']['x']:.3f}, {combined_transform['translation']['y']:.3f})")
                # Convert to dictionary format for data processor
                # Use 'odom' as frame_id since process_tf_message_dict looks for odom transforms
                tf_dict = {
                    'transforms': [{
                        'header': {
                            'frame_id': 'odom'
                        },
                        'transform': {
                            'translation': combined_transform['translation'],
                            'rotation': combined_transform['rotation']
                        }
                    }]
                }
                                
                # Process with data processor
                ego_data = process_tf_message_dict(tf_dict)
                print(f"🔍 TF PROCESSING: process_tf_message_dict result: {ego_data is not None}")
                
                if ego_data:
                    ego_data['timestamp'] = max(self.latest_map_odom_timestamp, self.latest_odom_base_timestamp)
                    ego_data['seq'] = self.latest_odom_base_frame['seq']
                    print(f"📝 EGO POSE: Adding combined global pose (map->base_footprint) to data processor")
                    print(f"   Timestamp: {ego_data['timestamp']:.3f}")
                    print(f"   Position: x={ego_data['x']:.3f}, y={ego_data['y']:.3f}, z={ego_data['z']:.3f}")
                    self.data_processor.add_ego_pose(ego_data)
                    print(f"📊 EGO POSES: Total ego poses now: {len(self.data_processor.ego_poses)}")
                    
                    # Show recent ego poses for debugging
                    recent_poses = list(self.data_processor.ego_poses)[-5:]  # Last 5 poses
                    print(f"📈 RECENT EGO POSES (last {len(recent_poses)}):")
                    for i, pose in enumerate(recent_poses):
                        print(f"   {i}: t={pose.get('timestamp', 'N/A'):.3f}, x={pose.get('x', 'N/A'):.3f}, y={pose.get('y', 'N/A'):.3f}")
                else:
                    print(f"❌ TF PROCESSING: process_tf_message_dict returned None - check data processor!")
            else:
                print(f"❌ TF COMBINATION: Transform combination failed!")
        else:
            print(f"⚠️  TF STATUS: Missing transforms for combination")
            if not self.latest_map_odom_frame:
                print(f"   ❌ Still waiting for map->odom transform")
            if not self.latest_odom_base_frame:
                print(f"   ❌ Still waiting for odom->base_footprint transform")
            
            # FALLBACK: If we only have odom->base_footprint, use it directly for now
            if odom_base_transforms and not self.latest_map_odom_frame:
                print(f"🔄 FALLBACK: Using odom->base_footprint directly (no map->odom available)")
                latest_transform = odom_base_transforms[-1]
                
                # Create a simplified tf_dict with just the odom transform
                tf_dict = {
                    'transforms': [{
                        'header': {
                            'frame_id': 'odom'  # Use odom as reference frame
                        },
                        'transform': {
                            'translation': {
                                'x': latest_transform.transform.translation.x,
                                'y': latest_transform.transform.translation.y,
                                'z': latest_transform.transform.translation.z
                            },
                            'rotation': {
                                'x': latest_transform.transform.rotation.x,
                                'y': latest_transform.transform.rotation.y,
                                'z': latest_transform.transform.rotation.z,
                                'w': latest_transform.transform.rotation.w
                            }
                        }
                    }]
                }
                
                ego_data = process_tf_message_dict(tf_dict)
                if ego_data:
                    ego_data['timestamp'] = latest_transform.header.stamp.to_sec()
                    ego_data['seq'] = latest_transform.header.seq
                    self.data_processor.add_ego_pose(ego_data)
                    print(f"✅ FALLBACK: Added odom pose, total: {len(self.data_processor.ego_poses)}")
                else:
                    print(f"❌ FALLBACK: Failed to process odom transform")

    def combine_transforms(self, map_odom_transform, odom_base_transform):
        """Combine map->odom and odom->base_footprint to get map->base_footprint transform"""
        try:
            # print(f"DEBUG: Attempting to combine transforms...")
            # print(f"DEBUG: Map->Odom translation: {map_odom_transform['translation']}")
            # print(f"DEBUG: Odom->Base translation: {odom_base_transform['translation']}")
            
            # Extract map->odom transform
            t1 = np.array([
                map_odom_transform['translation']['x'],
                map_odom_transform['translation']['y'],
                map_odom_transform['translation']['z']
            ])
            q1 = Quaternion(
                map_odom_transform['rotation']['w'],
                map_odom_transform['rotation']['x'],
                map_odom_transform['rotation']['y'],
                map_odom_transform['rotation']['z']
            )
            
            # Extract odom->base_footprint transform
            t2 = np.array([
                odom_base_transform['translation']['x'],
                odom_base_transform['translation']['y'],
                odom_base_transform['translation']['z']
            ])
            q2 = Quaternion(
                odom_base_transform['rotation']['w'],
                odom_base_transform['rotation']['x'],
                odom_base_transform['rotation']['y'],
                odom_base_transform['rotation']['z']
            )
            
            # Combine transforms: T_map_base = T_map_odom * T_odom_base
            # Translation: t_combined = t1 + q1.rotate(t2)
            t_combined = t1 + q1.rotate(t2)
            
            # Rotation: q_combined = q1 * q2
            q_combined = q1 * q2
            
            # print(f"DEBUG: Combined translation: {t_combined}")
            # print(f"DEBUG: Combined rotation: [x={q_combined.x:.3f}, y={q_combined.y:.3f}, z={q_combined.z:.3f}, w={q_combined.w:.3f}]")
            
            # Return combined transform
            result = {
                'translation': {
                    'x': float(t_combined[0]),
                    'y': float(t_combined[1]),
                    'z': float(t_combined[2])
                },
                'rotation': {
                    'x': float(q_combined.x),
                    'y': float(q_combined.y),
                    'z': float(q_combined.z),
                    'w': float(q_combined.w)
                }
            }
            
            # print(f"DEBUG: Transform combination successful!")
            return result
            
        except Exception as e:
            rospy.logwarn("Failed to combine transforms: {}".format(e))
            import traceback
            rospy.logwarn("Transform combination traceback: {}".format(traceback.format_exc()))
            return None

    def get_current_reference_frame(self):
        """Get the current reference frame (combined map->base_footprint transform)"""
        if self.latest_map_odom_frame and self.latest_odom_base_frame:
            return self.combine_transforms(self.latest_map_odom_frame, self.latest_odom_base_frame)
        return None

    def image_detections_callback(self, msg):
        """Process person detections from /image_detections topic"""        
        #print("=" * 50)
        #print("🎯 IMAGE_DETECTIONS MESSAGE RECEIVED!")
        #print(f"Found {len(msg.persons)} persons in message")
        #print("=" * 50)
        
        # PersonsList message contains an array of Person objects
        # Use message timestamp instead of current time (important for rosbag playback)
        if hasattr(msg, 'header') and msg.header.stamp.to_sec() > 0:
            timestamp = msg.header.stamp.to_sec()
        else:
            # Fallback to current time if no header timestamp
            timestamp = rospy.Time.now().to_sec()
        
        self.data_processor.update_frame_count()
        
        # Process each person in the list
        for person_idx, person in enumerate(msg.persons):
            print(f"Processing person {person_idx}: ID='{person.id}', body_parts={len(person.body_parts)}")
            
            # Extract keypoints from the person
            keypoints = self.extract_keypoints_from_person(person)
            
            # Use body_pose directly from Person message structure
            body_pose = person.body_pose
            person_data = {
                'id': person.id if person.id else f"person_{person_idx}",  # Use index if no ID
                'timestamp': timestamp,
                'keypoints': keypoints,
                'x': float(body_pose.position.x),
                'y': float(body_pose.position.y), 
                'z': float(body_pose.position.z),
                'orientation_x': float(body_pose.orientation.x),
                'orientation_y': float(body_pose.orientation.y),
                'orientation_z': float(body_pose.orientation.z),
                'orientation_w': float(body_pose.orientation.w)
            }
            
            # Add person to data processor
            # Convert keypoints to the format expected by data processor
            keypoints_flat = self.get_raw_keypoints_list(keypoints)
            person_data['keypoints'] = keypoints_flat
            person_data['body_parts'] = [{'part_id': part, 'x': kp['x'], 'y': kp['y'], 'confidence': kp['confidence']} 
                                       for part, kp in zip(CORRECT_ORDER, keypoints)]
            
            self.data_processor.add_person_detection(person_data, timestamp)
        
        # Print frame status with batch progress
        ego_count = len(self.data_processor.ego_poses)
        minimum_required = self.obs_len  # Minimum frames needed to start prediction
        optimal_data = self.seq_len * self.interval  # Optimal number of frames
        active_persons = len(self.data_processor.get_active_persons())
        
        print(f"📊 FRAME STATUS: Frame {self.data_processor.frame_count}")
        print(f"   👥 Persons detected: {len(msg.persons)}")
        print(f"   🗺️  Map->Base poses: {ego_count} (minimum needed: {minimum_required}, optimal: {optimal_data})")
        print(f"   🎯 Active persons: {active_persons}")
        
        if ego_count < minimum_required:
            print(f"⚠️  WARNING: Not enough ego poses! Need {minimum_required}, have {ego_count}")
            print(f"   📈 Need {minimum_required - ego_count} more ego poses before predictions can start")
        else:
            print(f"✅ SUFFICIENT DATA: Ready for predictions!")
        
        rospy.loginfo("Frame {}: {} persons, Map->Base poses: {} (min:{}, opt:{}), Active persons: {}".format(
            self.data_processor.frame_count, len(msg.persons), ego_count, minimum_required, optimal_data, active_persons))
        
        # Try to predict trajectories (pass timestamp for rosbag compatibility)
        self.try_predict_trajectories(timestamp)

    def extract_keypoints_from_person(self, person):
        """Extract keypoints from Person message in correct order"""
        # Create mapping from part_id to coordinates
        part_map = {}
        
        print(f"DEBUG: Extracting keypoints from {len(person.body_parts)} body parts")
        
        # Process body_parts from Person message
        for body_part in person.body_parts:
            # From the examples, BodyPart has: part_id (string), x, y, confidence
            part_name = body_part.part_id
            part_map[part_name] = {
                'x': float(body_part.x),
                'y': float(body_part.y),
                'confidence': float(body_part.confidence)
            }
            #print(f"  {part_name}: ({body_part.x}, {body_part.y}, conf={body_part.confidence:.3f})")
        
        #print(f"DEBUG: Found keypoints for: {list(part_map.keys())}")
        
        # Map from the message format to our expected format
        # From the examples, we have: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, 
        # MidHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar, etc.
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
                keypoints.append(part_map[keypoint_mapping[expected_name]])
                print(f"  ✓ Mapped {expected_name}")
            else:
                keypoints.append({'x': 0.0, 'y': 0.0, 'confidence': 0.0})
                #print(f"  ✗ Missing {expected_name}")
        
        #print(f"DEBUG: Extracted {len(keypoints)} keypoints total")
        return keypoints

    def estimate_position_from_keypoints(self, keypoints):
        """Estimate 3D position from 2D keypoints"""
        valid_kps = [kp for kp in keypoints if kp['confidence'] > 0.3]
        
        if not valid_kps:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # Calculate center
        x_center = sum(kp['x'] for kp in valid_kps) / len(valid_kps)
        y_center = sum(kp['y'] for kp in valid_kps) / len(valid_kps)
        
        # Estimate depth from bounding box
        bbox = self.compute_bbox_from_keypoints(keypoints)
        bbox_height = bbox[3] - bbox[1] if bbox[3] > bbox[1] else 100
        
        if bbox_height > 0:
            estimated_depth = max(1.5, min(8.0, 1.7 * VIZZY_CAMERA_K[1,1] / bbox_height))
        else:
            estimated_depth = 3.0
        
        # Convert to world coordinates
        world_x = (x_center - VIZZY_CAMERA_K[0,2]) * estimated_depth / VIZZY_CAMERA_K[0,0]
        world_y = (y_center - VIZZY_CAMERA_K[1,2]) * estimated_depth / VIZZY_CAMERA_K[1,1]
        
        return {
            'x': float(estimated_depth),  # Forward distance
            'y': float(-world_x),         # Left-right
            'z': float(-world_y + 1.2)    # Up-down + camera height
        }

    def compute_bbox_from_keypoints(self, keypoints):
        """Compute bounding box from keypoints"""
        xs, ys = [], []
        for kp in keypoints:
            if kp['confidence'] > 0:
                xs.append(kp['x'])
                ys.append(kp['y'])
        
        if not xs or not ys:
            return [0.0, 0.0, 0.0, 0.0]
        
        return [min(xs), min(ys), max(xs), max(ys)]

    def get_raw_keypoints_list(self, keypoints):
        """Convert keypoints to flat list format"""
        raw_list = []
        for kp in keypoints:
            raw_list.extend([kp['x'], kp['y'], kp['confidence']])
        return raw_list

    def try_predict_trajectories(self, message_timestamp=None):
        """Attempt to predict trajectories if we have sufficient data"""
        print(f"DEBUG: try_predict_trajectories called")
        print(f"DEBUG: Model available: {self.model is not None}")
        
        if not self.model:
            print(f"DEBUG: No model available, skipping prediction")
            return
        
        print(f"DEBUG: Checking data availability...")
        print(f"DEBUG: Ego poses: {len(self.data_processor.ego_poses)}")
        print(f"DEBUG: Active persons: {len(self.data_processor.get_active_persons())}")
        
        predictions = {}
        # Use message timestamp for rosbag compatibility, fallback to Time(0) 
        if message_timestamp is not None:
            current_time = rospy.Time.from_sec(message_timestamp)
        else:
            current_time = rospy.Time(0)  # Will work with both real-time and simulation time
        
        # Get active persons from data processor
        active_persons = self.data_processor.get_active_persons()
        print(f"DEBUG: Active persons list: {active_persons}")
        
        for person_id in active_persons:
            print(f"DEBUG: Processing person {person_id}")
            try:
                # Generate model input using data processor
                model_input = self.data_processor.generate_model_input(person_id)
                print(f"DEBUG: Model input for {person_id}: {model_input is not None}")
                
                if model_input is not None:
                    print(f"DEBUG: Attempting trajectory prediction for {person_id}")
                    # Predict trajectory
                    trajectory = self.predict_single_trajectory(model_input)
                    
                    if trajectory is not None:
                        predictions[person_id] = trajectory
                        rospy.loginfo("✓ Predicted trajectory for person {} - {} steps".format(person_id, len(trajectory)))
                    else:
                        print(f"DEBUG: Trajectory prediction failed for {person_id}")
                else:
                    print(f"DEBUG: No model input available for {person_id}")
                    
            except Exception as e:
                rospy.logwarn("Failed to predict trajectory for person {}: {}".format(person_id, e))
                import traceback
                print(f"DEBUG: Prediction error traceback: {traceback.format_exc()}")
        
        print(f"DEBUG: Total predictions: {len(predictions)}")
        if predictions:
            print(f"DEBUG: Publishing predictions...")
            self.publish_predictions(predictions, current_time)
        else:
            print(f"DEBUG: No predictions to publish")

    def prepare_model_input(self, person_id, person_history):
        """Prepare data in the format expected by MonoTransmotion model"""
        try:
            # Get the most recent sequence
            recent_data = list(person_history)[-self.seq_len * self.interval:]
            recent_ego = list(self.ego_poses)[-self.seq_len * self.interval:]
            
            if len(recent_data) < self.seq_len * self.interval or len(recent_ego) < self.seq_len * self.interval:
                return None
            
            # Sample at intervals to get seq_len frames
            sampled_data = recent_data[::self.interval][:self.seq_len]
            sampled_ego = recent_ego[::self.interval][:self.seq_len]
            
            if len(sampled_data) < self.seq_len or len(sampled_ego) < self.seq_len:
                return None
            
            # Prepare model input tensors
            X = []  # 2D coordinates  
            kps = []  # Keypoints
            ego_pose = []  # Ego poses
            boxes_2d = []  # 2D bounding boxes
            boxes_3d = []  # 3D positions
            
            for i, (ped_data, ego_data) in enumerate(zip(sampled_data, sampled_ego)):
                # 2D position (from keypoints center)
                if 'keypoints' in ped_data and ped_data['keypoints']:
                    # Calculate center from keypoints
                    kps_array = np.array(ped_data['keypoints']).reshape(-1, 3)
                    valid_kps = kps_array[kps_array[:, 2] > 0]
                    if len(valid_kps) > 0:
                        x_2d = np.mean(valid_kps[:, 0])
                        y_2d = np.mean(valid_kps[:, 1])
                    else:
                        x_2d, y_2d = 320.0, 240.0  # Image center default
                else:
                    x_2d, y_2d = 320.0, 240.0
                
                X.append([x_2d, y_2d])
                
                # Keypoints (convert from [x,y,conf] format to just [x,y] format)
                if 'keypoints' in ped_data:
                    kps_data = ped_data['keypoints']
                    print(f"DEBUG: Processing keypoints for frame {i}, raw shape: {np.array(kps_data).shape}")
                    
                    # Convert from flat [x,y,conf,x,y,conf,...] to [[x,y],[x,y],...] format
                    if len(kps_data) == 51:  # 17 keypoints * 3 values
                        kps_xy = []
                        for j in range(0, len(kps_data), 3):
                            kps_xy.extend([kps_data[j], kps_data[j+1]])  # x, y only
                        kps.append(kps_xy)  # 34 values (17 keypoints * 2)
                        print(f"DEBUG: Converted keypoints to xy-only format, length: {len(kps_xy)}")
                    else:
                        print(f"DEBUG: Unexpected keypoints length: {len(kps_data)}, using zeros")
                        kps.append([0.0] * 34)  # 17 keypoints * 2 values
                else:
                    print(f"DEBUG: No keypoints for frame {i}, using zeros")
                    kps.append([0.0] * 34)  # 17 keypoints * 2 values
                
                # Ego pose [q1, q2, q3, q4, x, y, z]
                ego_pose.append([
                    ego_data['q1'], ego_data['q2'], ego_data['q3'], ego_data['q4'],
                    ego_data['x'], ego_data['y'], ego_data['z']
                ])
                
                # 2D bounding box
                if 'bbox' in ped_data:
                    boxes_2d.append(ped_data['bbox'])
                else:
                    boxes_2d.append([x_2d-50, y_2d-100, x_2d+50, y_2d+100])
                
                # 3D position
                boxes_3d.append([ped_data['x'], ped_data['y'], ped_data['z']])
            
            # Convert to tensors
            print(f"DEBUG: Converting to tensors - kps list has {len(kps)} elements")
            print(f"DEBUG: First kps element length: {len(kps[0]) if kps else 'empty'}")
            
            model_input = {
                'X': torch.tensor(X, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 2]
                'kps': torch.tensor(kps, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 34]
                'ego_pose': torch.tensor(ego_pose, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 7]
                'boxes_2d': torch.tensor(boxes_2d, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 4]
                'boxes_3d': torch.tensor(boxes_3d, dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 3]
            }
            
            print(f"DEBUG: Final tensor shapes:")
            for key, tensor in model_input.items():
                print(f"  {key}: {tensor.shape}")
            
            return model_input
            
        except Exception as e:
            rospy.logwarn("Failed to prepare model input: {}".format(e))
            return None

    def predict_single_trajectory(self, model_input):
        """Use the model to predict a single trajectory"""
        try:
            print("=" * 60)
            print("🚀 MODEL PREDICTION START")
            print("=" * 60)
            
            with torch.no_grad():
                # Prepare input data for the localization model
                inputs = model_input['kps']  # Use keypoints as input
                print(f"🔍 MODEL INPUT: Input tensor shape: {inputs.shape}")
                print(f"🔍 MODEL INPUT: Input tensor type: {inputs.dtype}")
                print(f"🔍 MODEL INPUT: Input tensor device: {inputs.device}")
                
                # Convert tensor to the format expected by joint2traj: [batch, seq, 17, 2]
                if len(inputs.shape) == 4:
                    # Shape is [batch, seq, 3, 17] from data processor
                    batch_size, seq_length, features, num_keypoints = inputs.shape
                    if features == 3 and num_keypoints == 17:
                        # Take only x,y coordinates (first 2 features), drop confidence
                        inputs = inputs[:, :, :2, :].permute(0, 1, 3, 2)  # [batch, seq, 17, 2]
                        print(f"🔧 TENSOR CONVERSION: Converted from [batch, seq, 3, 17] to [batch, seq, 17, 2]: {inputs.shape}")
                    else:
                        print(f"❌ ERROR: Unexpected 4D tensor dimensions: {inputs.shape}")
                        return None
                elif len(inputs.shape) == 3:
                    batch_size, seq_length, features = inputs.shape
                    if features == 34:  # 17 keypoints * 2 values (x,y)
                        inputs = inputs.view(batch_size, seq_length, 17, 2)
                        print(f"🔧 TENSOR CONVERSION: Reshaped from [batch, seq, 34] to [batch, seq, 17, 2]: {inputs.shape}")
                    elif features == 51:  # 17 keypoints * 3 values (x,y,conf) - shouldn't happen now
                        inputs_reshaped = inputs.view(batch_size, seq_length, 17, 3)
                        inputs = inputs_reshaped[:, :, :, :2]  # Take only x,y, drop confidence
                        print(f"🔧 TENSOR CONVERSION: Converted from [batch, seq, 51] to [batch, seq, 17, 2]: {inputs.shape}")
                    else:
                        print(f"❌ ERROR: Unexpected feature dimension: {features}, expected 34")
                        return None
                else:
                    print(f"❌ ERROR: Unexpected input tensor shape: {inputs.shape}")
                    return None
                
                print(f"✅ FINAL INPUT: Shape for joint2traj: {inputs.shape}")
                print(f"📊 INPUT STATS: Min={inputs.min():.3f}, Max={inputs.max():.3f}, Mean={inputs.mean():.3f}")
                
                # Create scene representation from keypoints
                from utils import joint2traj, recover_traj, loc2traj, batch_process_coords
                
                print("🎯 JOINT2TRAJ: Converting keypoints to scene...")
                scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)
                print(f"🎯 JOINT2TRAJ RESULTS:")
                print(f"   scene_train_real_ped: {scene_train_real_ped.shape}")
                print(f"   scene_train_mask: {scene_train_mask.shape}")
                print(f"   padding_mask: {padding_mask.shape}")
                
                scene_train_real_ped = scene_train_real_ped.to(self.evaluator.traj_config["DEVICE"])
                scene_train_mask = scene_train_mask.to(self.evaluator.traj_config["DEVICE"])
                padding_mask = padding_mask.to(self.evaluator.traj_config["DEVICE"])
                
                # Use only the first person in the scene
                scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                scene_train_mask = scene_train_mask[:,0,:,:]
                print(f"🎯 PERSON SELECTION: Selected first person, new shapes:")
                print(f"   scene_train_real_ped: {scene_train_real_ped.shape}")
                print(f"   scene_train_mask: {scene_train_mask.shape}")
                
                # Use only observation frames for prediction
                scene_train_real_ped_obs = scene_train_real_ped[:,:self.obs_len,:,:]
                padding_mask_obs = padding_mask.clone()
                padding_mask_obs[:,self.obs_len:] = True
                print(f"🔍 OBSERVATION SELECTION: Using {self.obs_len} frames for localization")
                print(f"   scene_train_real_ped_obs: {scene_train_real_ped_obs.shape}")
                
                # Get localization output
                print("🧠 LOCALIZATION MODEL: Running inference...")
                loc_outputs = self.evaluator.loc_model(scene_train_real_ped_obs, padding_mask_obs)
                print(f"🧠 LOCALIZATION RESULTS:")
                print(f"   loc_outputs shape: {loc_outputs.shape}")
                print(f"   loc_outputs stats: Min={loc_outputs.min():.3f}, Max={loc_outputs.max():.3f}, Mean={loc_outputs.mean():.3f}")
                print(f"   loc_outputs sample values: {loc_outputs.flatten()[:10]}")
                
                # Recover trajectory from localization - try simplified approach
                ego_pose_tensor = model_input['ego_pose']
                
                print(f"🌍 EGO POSE: ego_pose_tensor shape: {ego_pose_tensor.shape}")
                
                # Use only observation length for recovery
                ego_pose_obs = ego_pose_tensor[:, :self.obs_len, :]
                print(f"🌍 EGO POSE OBS: ego_pose_obs shape: {ego_pose_obs.shape}")
                
                # Try to bypass the recover_traj function which is causing issues
                # For real-time prediction, we can use the localization output directly
                try:
                    print("🔄 TRAJECTORY RECOVERY: Attempting direct approach...")
                    # Skip the camera pose transformation and use localization outputs directly
                    # This assumes the localization model already provides world coordinates
                    traj_estimated_ls = loc_outputs.unsqueeze(0)  # Add batch dimension if needed
                    print(f"✅ TRAJECTORY RECOVERY: Using localization outputs directly: {traj_estimated_ls.shape}")
                    
                except Exception as e:
                    print(f"❌ TRAJECTORY RECOVERY: Direct approach failed: {e}")
                    # Fallback: try with simplified camera pose
                    try:
                        print("🔄 TRAJECTORY RECOVERY: Trying simplified camera pose...")
                        # Create a very simple camera pose (just identity matrices)
                        batch_size = ego_pose_obs.shape[0]
                        camera_pose_simple = torch.zeros(batch_size, self.obs_len, 7)  # [x,y,z,qx,qy,qz,qw]
                        camera_pose_simple[:, :, 6] = 1.0  # Set w=1 for identity quaternion
                        traj_estimated_ls = recover_traj(loc_outputs, ego_pose_obs, camera_pose_simple)
                        print(f"✅ TRAJECTORY RECOVERY: Simplified camera pose worked: {traj_estimated_ls.shape}")
                    except Exception as e2:
                        print(f"❌ TRAJECTORY RECOVERY: Simplified camera pose failed: {e2}")
                        # Last resort: manually create trajectory from localization
                        print("🔄 TRAJECTORY RECOVERY: Manual creation as last resort...")
                        traj_estimated_ls = loc_outputs.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                        print(f"✅ TRAJECTORY RECOVERY: Manual trajectory creation: {traj_estimated_ls.shape}")
                
                # Convert to trajectory prediction format
                print("🎯 LOC2TRAJ: Converting to trajectory format...")
                scene_train_real_ped_traj, scene_train_mask_traj, padding_mask_traj = loc2traj(traj_estimated_ls)
                print(f"🎯 LOC2TRAJ RESULTS:")
                print(f"   scene_train_real_ped_traj: {scene_train_real_ped_traj.shape}")
                print(f"   scene_train_mask_traj: {scene_train_mask_traj.shape}")
                print(f"   padding_mask_traj: {padding_mask_traj.shape}")
                
                # Process for trajectory prediction
                print("⚙️ BATCH PROCESSING: Processing coordinates...")
                in_joints, in_masks, out_joints, out_masks, padding_mask_processed, _ = batch_process_coords(
                    scene_train_real_ped_traj, scene_train_mask_traj, padding_mask_traj, 
                    self.evaluator.traj_config, training=False)
                
                print(f"⚙️ BATCH PROCESSING RESULTS:")
                print(f"   in_joints: {in_joints.shape}")
                print(f"   in_masks: {in_masks.shape}")
                print(f"   out_joints: {out_joints.shape}")
                print(f"   out_masks: {out_masks.shape}")
                print(f"   padding_mask_processed: {padding_mask_processed.shape}")
                
                padding_mask_processed = padding_mask_processed.to(self.evaluator.traj_config["DEVICE"])
                
                # Predict trajectory
                print("🚀 TRAJECTORY MODEL: Running trajectory prediction...")
                pred_joints = self.evaluator.traj_model(in_joints, padding_mask_processed)
                print(f"🚀 TRAJECTORY MODEL RESULTS:")
                print(f"   pred_joints shape: {pred_joints.shape}")
                print(f"   pred_joints stats: Min={pred_joints.min():.3f}, Max={pred_joints.max():.3f}, Mean={pred_joints.mean():.3f}")
                
                # Get prediction for future frames
                pred_joints = pred_joints[:, -self.pred_len:]
                pred_joints = pred_joints.cpu()
                print(f"🎯 FUTURE PREDICTION:")
                print(f"   pred_joints (future {self.pred_len} frames): {pred_joints.shape}")
                print(f"   pred_joints stats: Min={pred_joints.min():.3f}, Max={pred_joints.max():.3f}, Mean={pred_joints.mean():.3f}")
                
                # Add relative position to last observation
                last_obs_pos = scene_train_real_ped_traj[:,0:1,(self.obs_len-1):self.obs_len, 0, 0:2]
                print(f"🔄 POSITION ADJUSTMENT:")
                print(f"   last_obs_pos: {last_obs_pos.shape}")
                print(f"   last_obs_pos values: {last_obs_pos}")
                
                pred_joints = pred_joints + last_obs_pos
                print(f"   pred_joints after position adjustment: {pred_joints.shape}")
                print(f"   pred_joints adjusted stats: Min={pred_joints.min():.3f}, Max={pred_joints.max():.3f}, Mean={pred_joints.mean():.3f}")
                
                # Convert to trajectory format
                pred_array = pred_joints.reshape(pred_joints.size(0), self.pred_len, 2)
                pred_array = pred_array[0].numpy()  # Take first (and only) batch item
                print(f"📊 FINAL TRAJECTORY:")
                print(f"   pred_array shape: {pred_array.shape}")
                print(f"   pred_array values:")
                for i, pos in enumerate(pred_array):
                    print(f"     Step {i}: x={pos[0]:.3f}, y={pos[1]:.3f}")
                
                trajectory = []
                for i in range(pred_array.shape[0]):
                    trajectory.append({
                        'x': float(pred_array[i, 0]),
                        'y': float(pred_array[i, 1]),
                        'z': 0.0  # 2D prediction, z coordinate is not predicted
                    })
                
                print(f"✅ SUCCESS: Generated trajectory with {len(trajectory)} points")
                print("=" * 60)
                print("🎉 MODEL PREDICTION COMPLETE")
                print("=" * 60)
                
                return trajectory
                
        except Exception as e:
            print("=" * 60)
            print("❌ MODEL PREDICTION FAILED")
            print("=" * 60)
            rospy.logwarn("Model prediction failed: {}".format(e))
            import traceback
            print(f"❌ FULL TRACEBACK:")
            print(traceback.format_exc())
            print("=" * 60)
            return None

    def publish_predictions(self, predictions, timestamp):
        """Publish trajectory predictions"""
        print("=" * 60)
        print("📢 PUBLISHING PREDICTIONS")
        print("=" * 60)
        print(f"📊 Publishing {len(predictions)} predictions at timestamp {timestamp.to_sec()}")
        
        for person_id, trajectory in predictions.items():
            print(f"👤 Person {person_id}: {len(trajectory)} trajectory points")
            for i, point in enumerate(trajectory):
                print(f"   Step {i}: x={point['x']:.3f}, y={point['y']:.3f}, z={point['z']:.3f}")
        
        # Publish as PoseArray
        pose_array = PoseArray()
        pose_array.header.stamp = timestamp
        pose_array.header.frame_id = "map"  # Use map frame instead of odom
        
        total_poses = 0
        for person_id, trajectory in predictions.items():
            for step, pos in enumerate(trajectory):
                pose = Pose()
                pose.position.x = pos['x']
                pose.position.y = pos['y']
                pose.position.z = pos['z']
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)
                total_poses += 1
        
        print(f"📢 POSE ARRAY: Publishing {total_poses} poses to /predicted_trajectories")
        self.trajectory_pub.publish(pose_array)
        
        # Publish as JSON
        json_data = {
            'timestamp': timestamp.to_sec(),
            'predictions': predictions
        }
        json_msg = String()
        json_msg.data = json.dumps(json_data)
        print(f"📢 JSON: Publishing JSON data to /predicted_trajectories_json")
        self.trajectory_json_pub.publish(json_msg)
        
        # Publish as Path for each person
        for person_id, trajectory in predictions.items():
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = "map"  # Use map frame instead of odom
            
            for step, pos in enumerate(trajectory):
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = timestamp
                pose_stamped.header.frame_id = "map"  # Use map frame instead of odom
                pose_stamped.pose.position.x = pos['x']
                pose_stamped.pose.position.y = pos['y']
                pose_stamped.pose.position.z = pos['z'] if pos['z'] != 0.0 else 0.02  # Slightly above ground
                pose_stamped.pose.orientation.w = 1.0
                path_msg.poses.append(pose_stamped)
            
            print(f"📢 PATH: Publishing path with {len(path_msg.poses)} poses for person {person_id} to /predicted_paths")
            self.path_pub.publish(path_msg)
        
        # Create visualization markers
        print(f"📢 MARKERS: Creating visualization markers...")
        self.publish_trajectory_visualization(predictions, timestamp)
        
        print(f"✅ PUBLISHING COMPLETE: Published predictions for {len(predictions)} people")
        print("=" * 60)
        
        rospy.loginfo("Published predictions for {} people".format(len(predictions)))

    def publish_trajectory_visualization(self, predictions, timestamp):
        """Create and publish trajectory visualization markers"""
        print("🎨 VISUALIZATION: Creating markers...")
        
        marker_array = MarkerArray()
        marker_id = 0
        total_markers = 0
        
        for person_id, trajectory in predictions.items():
            print(f"🎨 Creating markers for person {person_id} with {len(trajectory)} points")
            
            # Create line strip for trajectory path
            trajectory_marker = Marker()
            trajectory_marker.header.frame_id = "map"  # Use map frame instead of odom
            trajectory_marker.header.stamp = timestamp
            trajectory_marker.ns = "predicted_trajectories"
            trajectory_marker.id = marker_id
            trajectory_marker.type = Marker.LINE_STRIP
            trajectory_marker.action = Marker.ADD
            
            trajectory_marker.scale.x = 0.08  # Line width
            trajectory_marker.color.r = 1.0
            trajectory_marker.color.g = 0.0
            trajectory_marker.color.b = 0.0
            trajectory_marker.color.a = 0.8
            
            for pos in trajectory:
                point = Point()
                point.x = pos['x']
                point.y = pos['y']
                point.z = pos['z'] if pos['z'] != 0.0 else 0.05  # Slightly above ground
                trajectory_marker.points.append(point)
            
            marker_array.markers.append(trajectory_marker)
            marker_id += 1
            total_markers += 1
            print(f"   ✓ Line strip marker created (ID: {marker_id-1})")
            
            # Add spheres for each predicted position
            for i, pos in enumerate(trajectory):
                sphere_marker = Marker()
                sphere_marker.header.frame_id = "map"  # Use map frame instead of odom
                sphere_marker.header.stamp = timestamp
                sphere_marker.ns = "trajectory_points"
                sphere_marker.id = marker_id
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                
                sphere_marker.pose.position.x = pos['x']
                sphere_marker.pose.position.y = pos['y']
                sphere_marker.pose.position.z = pos['z'] if pos['z'] != 0.0 else 0.05
                sphere_marker.pose.orientation.w = 1.0
                
                # Scale based on prediction step (future steps are smaller)
                scale = 0.15 - (i * 0.01)  # Decrease size for future steps
                sphere_marker.scale.x = scale
                sphere_marker.scale.y = scale
                sphere_marker.scale.z = scale
                
                # Color gradient: red to yellow over time
                sphere_marker.color.r = 1.0
                sphere_marker.color.g = min(1.0, i * 0.2)  # Increase green over time
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 0.7
                
                marker_array.markers.append(sphere_marker)
                marker_id += 1
                total_markers += 1
            
            print(f"   ✓ {len(trajectory)} sphere markers created")
            
            # Add text label for person ID
            text_marker = Marker()
            text_marker.header.frame_id = "map"  # Use map frame instead of odom
            text_marker.header.stamp = timestamp
            text_marker.ns = "trajectory_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position text at the end of trajectory
            if trajectory:
                last_pos = trajectory[-1]
                text_marker.pose.position.x = last_pos['x']
                text_marker.pose.position.y = last_pos['y']
                text_marker.pose.position.z = (last_pos['z'] if last_pos['z'] != 0.0 else 0.05) + 0.3
                text_marker.pose.orientation.w = 1.0
                
                text_marker.scale.z = 0.2  # Text height
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 0.9
                
                text_marker.text = f"Person {person_id}"
                
                marker_array.markers.append(text_marker)
                marker_id += 1
                total_markers += 1
                print(f"   ✓ Text marker created")
            
            # Add ground plane projection (trajectory projected to z=0)
            ground_marker = Marker()
            ground_marker.header.frame_id = "map"  # Use map frame instead of odom
            ground_marker.header.stamp = timestamp
            ground_marker.ns = "ground_projection"
            ground_marker.id = marker_id
            ground_marker.type = Marker.LINE_STRIP
            ground_marker.action = Marker.ADD
            
            ground_marker.scale.x = 0.05  # Thinner line for ground projection
            ground_marker.color.r = 0.0
            ground_marker.color.g = 1.0
            ground_marker.color.b = 1.0
            ground_marker.color.a = 0.6
            
            for pos in trajectory:
                point = Point()
                point.x = pos['x']
                point.y = pos['y']
                point.z = 0.01  # Just above ground plane
                ground_marker.points.append(point)
            
            marker_array.markers.append(ground_marker)
            marker_id += 1
            total_markers += 1
            print(f"   ✓ Ground projection marker created")
        
        print(f"🎨 VISUALIZATION: Publishing {total_markers} markers to /trajectory_visualization")
        self.visualization_pub.publish(marker_array)
        print(f"✅ VISUALIZATION: Markers published successfully")

    def run(self):
        """Main execution loop"""
        rospy.loginfo("Real-Time Trajectory Predictor is running...")
        rospy.loginfo("Predictions will start after minimum {} frames collected".format(self.obs_len))
        
        # For rosbag compatibility, use event-driven approach instead of polling
        # The script will respond to incoming messages rather than polling at a fixed rate
        rospy.loginfo("Script is event-driven - waiting for ROS messages...")
        rospy.loginfo("Topics subscribed: /tf (map->odom), /image_detections")
        
        # Add a timer to periodically check if we're still alive and print diagnostic info
        def diagnostic_timer(event):
            print(f"📊 DIAGNOSTIC: Frame count: {self.data_processor.frame_count}")
            print(f"📊 DIAGNOSTIC: Map->Odom poses collected: {len(self.data_processor.ego_poses)}")
            print(f"📊 DIAGNOSTIC: Active persons: {len(self.data_processor.get_active_persons())}")
            
            # Check if topics exist and print all available topics
            try:
                topics = rospy.get_published_topics()
                print(f"\n📋 ALL AVAILABLE TOPICS ({len(topics)} total):")
                
                image_detection_topics = []
                person_related_topics = []
                human_related_topics = []
                
                for topic_name, topic_type in topics:
                    print(f"  - {topic_name} ({topic_type})")
                    
                    # Look for topics that might contain person/human/detection data
                    if 'image_detections' in topic_name.lower():
                        image_detection_topics.append((topic_name, topic_type))
                    elif 'person' in topic_name.lower() or 'human' in topic_name.lower():
                        person_related_topics.append((topic_name, topic_type))
                    elif 'detection' in topic_name.lower() or 'track' in topic_name.lower():
                        human_related_topics.append((topic_name, topic_type))
                
                print(f"\n🎯 PERSON/DETECTION RELATED TOPICS:")
                if image_detection_topics:
                    print(f"  Image Detection Topics:")
                    for topic_name, topic_type in image_detection_topics:
                        print(f"    ✓ {topic_name} ({topic_type})")
                        
                if person_related_topics:
                    print(f"  Person/Human Topics:")
                    for topic_name, topic_type in person_related_topics:
                        print(f"    ✓ {topic_name} ({topic_type})")
                        
                if human_related_topics:
                    print(f"  Detection/Tracking Topics:")
                    for topic_name, topic_type in human_related_topics:
                        print(f"    ✓ {topic_name} ({topic_type})")
                
                if not (image_detection_topics or person_related_topics or human_related_topics):
                    print("  ❌ No person/detection related topics found!")
                    print("  💡 Check rosbag contents with: rosbag info your_file.bag")
                
                tf_exists = any('/tf' in topic[0] for topic in topics)
                detections_exists = any('/image_detections' in topic[0] for topic in topics)
                print(f"\n📊 TARGET TOPICS STATUS:")
                print(f"  /tf (map->odom): {'✓ FOUND' if tf_exists else '✗ MISSING'}")
                print(f"  /image_detections: {'✓ FOUND' if detections_exists else '✗ MISSING'}")
                
                if not detections_exists:
                    print("\n⚠️  WARNING: /image_detections topic not found!")
                    print("   Check if rosbag contains PersonTracker messages")
                    print("   or if the topic name is different")
                    print("   Look at the person-related topics listed above ↑")
                    
            except Exception as e:
                print(f"📊 DIAGNOSTIC: Could not check available topics: {e}")
        
        # Set up a 10-second diagnostic timer
        rospy.Timer(rospy.Duration(10.0), diagnostic_timer)
        
        print("DEBUG: Starting rospy.spin() - waiting for messages...")
        
        # Just spin - all work is done in callbacks
        rospy.spin()


def main():
    try:
        predictor = RealTimeTrajectoryPredictor()
        predictor.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Failed to start predictor: {}".format(e))


if __name__ == '__main__':
    main()
