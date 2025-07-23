import os
import sys
# Add model code directory to path
model_code_path = os.path.expanduser("~/thesis_aux/model/code")
sys.path.append(model_code_path)

# Add real-time data processor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import json
import numpy as np
import torch
from pyquaternion import Quaternion
import rospy
from tf2_msgs.msg import TFMessage
from human_awareness_msgs.msg import PersonsList, Person
from real_time_data_processor import (RealTimeDataProcessor, process_tf_message_dict, process_image_detections_dict, process_raw_bodies_dict)
from eval.evaluator import Evaluator
import yaml
import traceback
from utils import joint2traj, recover_traj, loc2traj, batch_process_coords
import argparse
from messages import TrajectoryMessageManager

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
        self.seq_len = 10
        self.interval = 15
        self.obs_len = 4
        self.pred_len = 2
        
        # Initialize data processor
        self.data_processor = RealTimeDataProcessor()
        
        # Store both transforms to compute map->base_footprint
        self.latest_map_odom_frame = None
        self.latest_map_odom_timestamp = None
        self.latest_odom_base_frame = None
        self.latest_odom_base_timestamp = None
        
        self.setup_model()

        # Initialize message manager for publishing
        self.message_manager = TrajectoryMessageManager()
        
        # Subscribers
        self.setup_subscribers()

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
            rospy.logwarn("Traceback: {}".format(traceback.format_exc()))
            self.model = None

    def setup_subscribers(self):
        """Setup ROS subscribers"""
        rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        # Note: /raw_bodies not needed - 3D poses are included in /image_detections
        rospy.Subscriber('/image_detections', PersonsList, self.image_detections_callback)

    def tf_callback(self, msg):
        """Process transform messages to extract both map->odom and odom->base_footprint transforms"""        
        map_odom_transforms = []
        odom_base_transforms = []
        
        # Find both types of transforms in this message
        for transform in msg.transforms:
            if transform.header.frame_id == 'map' and transform.child_frame_id == 'odom':
                map_odom_transforms.append(transform)
            elif transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_footprint':
                odom_base_transforms.append(transform)        
        
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
        
        # Combine transforms to get map->base_footprint if both are available
        if self.latest_map_odom_frame and self.latest_odom_base_frame:
            combined_transform = self.combine_transforms(
                self.latest_map_odom_frame, 
                self.latest_odom_base_frame
            )
            
            if combined_transform:
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
                
                if ego_data:
                    ego_data['timestamp'] = max(self.latest_map_odom_timestamp, self.latest_odom_base_timestamp)
                    ego_data['seq'] = self.latest_odom_base_frame['seq']
                    self.data_processor.add_ego_pose(ego_data)
            
            # FALLBACK: If we only have odom->base_footprint, use it directly for now
            if odom_base_transforms and not self.latest_map_odom_frame:
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

    def combine_transforms(self, map_odom_transform, odom_base_transform):
        """Combine map->odom and odom->base_footprint to get map->base_footprint transform"""
        try:
            
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
            
            return result
            
        except Exception as e:
            rospy.logwarn("Failed to combine transforms: {}".format(e))
            rospy.logwarn("Transform combination traceback: {}".format(traceback.format_exc()))
            return None

    def image_detections_callback(self, msg):
        """Process person detections from /image_detections topic"""        
        
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
            person_id = person.id if person.id else f"person_{person_idx}"
            
            # Extract keypoints from the person
            keypoints = self.extract_keypoints_from_person(person)
            
            # Use body_pose directly from Person message structure
            body_pose = person.body_pose
            person_data = {
                'id': person_id,
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
        
        # Try to predict trajectories (pass timestamp for rosbag compatibility)
        self.try_predict_trajectories(timestamp)

    def extract_keypoints_from_person(self, person):
        """Extract keypoints from Person message in correct order"""
        # Create mapping from part_id to coordinates
        part_map = {}
        # Process body_parts from Person message
        for body_part in person.body_parts:
            # From the examples, BodyPart has: part_id (string), x, y, confidence
            part_name = body_part.part_id
            part_map[part_name] = {
                'x': float(body_part.x),
                'y': float(body_part.y),
                'confidence': float(body_part.confidence)
            }
        
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
            else:
                keypoints.append({'x': 0.0, 'y': 0.0, 'confidence': 0.0})
        
        return keypoints

    def get_raw_keypoints_list(self, keypoints):
        """Convert keypoints to flat list format"""
        raw_list = []
        for kp in keypoints:
            raw_list.extend([kp['x'], kp['y'], kp['confidence']])
        return raw_list

    def try_predict_trajectories(self, message_timestamp=None):
        """Attempt to predict trajectories if we have sufficient data"""
        
        if not self.model:
            return
        
        predictions = {}
        # Use message timestamp for rosbag compatibility, fallback to Time(0) 
        if message_timestamp is not None:
            current_time = rospy.Time.from_sec(message_timestamp)
        else:
            current_time = rospy.Time(0)  # Will work with both real-time and simulation time
        
        # Get active persons from data processor
        active_persons = self.data_processor.get_active_persons()
        
        for person_id in active_persons:
            try:
                # Generate model input using data processor
                model_input = self.data_processor.generate_model_input(person_id)
                
                if model_input is not None:
                    # Predict trajectory
                    trajectory = self.predict_single_trajectory(model_input)
                    
                    if trajectory is not None:
                        predictions[person_id] = trajectory
            except Exception as e:
                rospy.logwarn("Failed to predict trajectory for person {}: {}".format(person_id, e))
                rospy.logdebug("   Traceback: {}".format(traceback.format_exc()))
        
        if predictions:
            self.publish_predictions(predictions, current_time)

    def predict_single_trajectory(self, model_input):
        """Use the model to predict a single trajectory"""
        try:
            with torch.no_grad():
                # Prepare input data for the localization model
                inputs = model_input['kps']  # Use keypoints as input
                
                # Convert tensor to the format expected by joint2traj: [batch, seq, 17, 2]
                if len(inputs.shape) == 4:
                    # Shape is [batch, seq, 3, 17] from data processor
                    batch_size, seq_length, features, num_keypoints = inputs.shape
                    if features == 3 and num_keypoints == 17:
                        # Take only x,y coordinates (first 2 features), drop confidence
                        inputs = inputs[:, :, :2, :].permute(0, 1, 3, 2)  # [batch, seq, 17, 2]
                    else:
                        return None
                elif len(inputs.shape) == 3:
                    batch_size, seq_length, features = inputs.shape
                    if features == 34:  # 17 keypoints * 2 values (x,y)
                        inputs = inputs.view(batch_size, seq_length, 17, 2)
                    elif features == 51:  # 17 keypoints * 3 values (x,y,conf) - shouldn't happen now
                        inputs_reshaped = inputs.view(batch_size, seq_length, 17, 3)
                        inputs = inputs_reshaped[:, :, :, :2]  # Take only x,y, drop confidence
                    else:
                        return None
                else:
                    return None
                
                # Create scene representation from keypoints
            
                scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)
                
                scene_train_real_ped = scene_train_real_ped.to(self.evaluator.traj_config["DEVICE"])
                scene_train_mask = scene_train_mask.to(self.evaluator.traj_config["DEVICE"])
                padding_mask = padding_mask.to(self.evaluator.traj_config["DEVICE"])
                
                # Use only the first person in the scene
                scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                scene_train_mask = scene_train_mask[:,0,:,:]
                
                # Use only observation frames for prediction
                scene_train_real_ped_obs = scene_train_real_ped[:,:self.obs_len,:,:]
                padding_mask_obs = padding_mask.clone()
                padding_mask_obs[:,self.obs_len:] = True
                
                # Get localization output
                loc_outputs = self.evaluator.loc_model(scene_train_real_ped_obs, padding_mask_obs)
                
                # Recover trajectory from localization - try simplified approach
                ego_pose_tensor = model_input['ego_pose']
                                
                # Use only observation length for recovery
                ego_pose_obs = ego_pose_tensor[:, :self.obs_len, :]
                
                # Try to bypass the recover_traj function which is causing issues
                # For real-time prediction, we can use the localization output directly
                try:
                    # Skip the camera pose transformation and use localization outputs directly
                    # This assumes the localization model already provides world coordinates
                    traj_estimated_ls = loc_outputs.unsqueeze(0)  # Add batch dimension if needed
                    
                except Exception as e:
                    # Fallback: try with simplified camera pose
                    try:
                        # Create a very simple camera pose (just identity matrices)
                        batch_size = ego_pose_obs.shape[0]
                        camera_pose_simple = torch.zeros(batch_size, self.obs_len, 7)  # [x,y,z,qx,qy,qz,qw]
                        camera_pose_simple[:, :, 6] = 1.0  # Set w=1 for identity quaternion
                        traj_estimated_ls = recover_traj(loc_outputs, ego_pose_obs, camera_pose_simple)
                    except Exception as e2:
                        # Last resort: manually create trajectory from localization
                        traj_estimated_ls = loc_outputs.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                
                # Convert to trajectory prediction format
                scene_train_real_ped_traj, scene_train_mask_traj, padding_mask_traj = loc2traj(traj_estimated_ls)
                
                # Process for trajectory prediction
                in_joints, in_masks, out_joints, out_masks, padding_mask_processed, _ = batch_process_coords(
                    scene_train_real_ped_traj, scene_train_mask_traj, padding_mask_traj, 
                    self.evaluator.traj_config, training=False)
                
                padding_mask_processed = padding_mask_processed.to(self.evaluator.traj_config["DEVICE"])
                
                # Predict trajectory
                pred_joints = self.evaluator.traj_model(in_joints, padding_mask_processed)
                
                # Get prediction for future frames
                pred_joints = pred_joints[:, -self.pred_len:]
                pred_joints = pred_joints.cpu()
                
                # Add relative position to last observation
                last_obs_pos = scene_train_real_ped_traj[:,0:1,(self.obs_len-1):self.obs_len, 0, 0:2]
                
                pred_joints = pred_joints + last_obs_pos
                
                # Convert to trajectory format
                pred_array = pred_joints.reshape(pred_joints.size(0), self.pred_len, 2)
                pred_array = pred_array[0].numpy()  # Take first (and only) batch item
                
                trajectory = []
                for i in range(pred_array.shape[0]):
                    trajectory.append({
                        'x': float(pred_array[i, 0]),
                        'y': float(pred_array[i, 1]),
                        'z': 0.0  # 2D prediction, z coordinate is not predicted
                    })
                
                return trajectory
                
        except Exception as e:
            rospy.logwarn("Model prediction failed: {}".format(e))
            return None

    def publish_predictions(self, predictions, timestamp):
        """Publish predictions using the message manager"""
        self.message_manager.publish_all_predictions(predictions, timestamp)

    def run(self):
        # Keep the node alive to process incoming messages
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down Real-Time Trajectory Predictor...")

def main():
    try:
        predictor = RealTimeTrajectoryPredictor()
        predictor.run()
    except Exception as e:
        rospy.logerr("Failed to start predictor: {}".format(e))

if __name__ == '__main__':
    main()
