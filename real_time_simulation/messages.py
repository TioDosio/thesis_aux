#!/usr/bin/env python3
"""
Message Publishing and Management Module for Real-Time Trajectory Prediction
Handles all ROS message creation, publishing, and visualization functions.
"""

import json
import math
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray


class TrajectoryMessageManager:
    """Manages all trajectory-related message publishing and visualization"""
    
    def __init__(self):
        # Publishers
        self.trajectory_pub = rospy.Publisher('/predicted_trajectories', PoseArray, queue_size=10)
        self.trajectory_json_pub = rospy.Publisher('/predicted_trajectories_json', String, queue_size=10)
        self.visualization_pub = rospy.Publisher('/trajectory_visualization', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('/predicted_paths', Path, queue_size=10)
    
    def publish_all_predictions(self, predictions, timestamp, last_person_positions=None):
        """
        Main function to publish predictions in all formats
        
        Args:
            predictions: Dictionary of person_id -> trajectory predictions
            timestamp: ROS timestamp for the predictions
            last_person_positions: Dictionary of person_id -> last known position {'x': x, 'y': y, 'z': z}
        """
        if not predictions:
            return
            
        # Publish in multiple formats
        self.publish_pose_array(predictions, timestamp, last_person_positions)
        self.publish_json_format(predictions, timestamp)
        self.publish_path_messages(predictions, timestamp, last_person_positions)
        self.publish_visualization_markers(predictions, timestamp)
        
        rospy.loginfo("Published predictions for {} people".format(len(predictions)))
    
    def publish_pose_array(self, predictions, timestamp, last_person_positions=None):
        """
        Publish predictions as PoseArray message with proper orientations
        
        Args:
            predictions: Dictionary of person_id -> trajectory predictions
            timestamp: ROS timestamp
            last_person_positions: Dictionary of person_id -> last known position
        """
        pose_array = PoseArray()
        pose_array.header.stamp = timestamp
        pose_array.header.frame_id = "map"
        
        for person_id, trajectory in predictions.items():
            # Get the reference position (last known position or first trajectory point)
            if last_person_positions and person_id in last_person_positions:
                ref_pos = last_person_positions[person_id]
            elif trajectory:
                # If no last position, use the first trajectory point as reference
                # This assumes the trajectory starts from current position
                ref_pos = trajectory[0]
            else:
                continue
            
            # Create poses for each trajectory point
            for step, pos in enumerate(trajectory):
                pose = Pose()
                pose.position.x = pos['x']
                pose.position.y = pos['y']
                pose.position.z = pos['z']
                
                # Calculate orientation based on movement direction
                if step == 0:
                    # First prediction point: direction from last known position to first prediction
                    dx = pos['x'] - ref_pos['x']
                    dy = pos['y'] - ref_pos['y']
                else:
                    # Subsequent points: direction from previous trajectory point
                    prev_pos = trajectory[step - 1]
                    dx = pos['x'] - prev_pos['x']
                    dy = pos['y'] - prev_pos['y']
                
                # Calculate yaw angle (rotation around z-axis)
                # Add small threshold to avoid issues with zero movement
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    # No movement, use default orientation (facing forward)
                    yaw = 0.0
                else:
                    yaw = math.atan2(dy, dx)
                
                # Convert yaw to quaternion (only rotation around z-axis)
                pose.orientation = self._yaw_to_quaternion(yaw)
                
                pose_array.poses.append(pose)
        
        self.trajectory_pub.publish(pose_array)
    
    def publish_json_format(self, predictions, timestamp):
        """Publish predictions as JSON string message"""
        json_data = {
            'timestamp': timestamp.to_sec(),
            'predictions': predictions
        }
        json_msg = String()
        json_msg.data = json.dumps(json_data)
        self.trajectory_json_pub.publish(json_msg)
    
    def publish_path_messages(self, predictions, timestamp, last_person_positions=None):
        """Publish Path messages for each person with proper orientations"""
        for person_id, trajectory in predictions.items():
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = "map"
            
            # Get reference position for orientation calculation
            if last_person_positions and person_id in last_person_positions:
                ref_pos = last_person_positions[person_id]
            elif trajectory:
                ref_pos = trajectory[0]
            else:
                continue
            
            for step, pos in enumerate(trajectory):
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = timestamp
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = pos['x']
                pose_stamped.pose.position.y = pos['y']
                pose_stamped.pose.position.z = pos['z'] if pos['z'] != 0.0 else 0.02
                
                # Calculate orientation based on movement direction
                if step == 0:
                    # First prediction point: direction from last known position to first prediction
                    dx = pos['x'] - ref_pos['x']
                    dy = pos['y'] - ref_pos['y']
                else:
                    # Subsequent points: direction from previous trajectory point
                    prev_pos = trajectory[step - 1]
                    dx = pos['x'] - prev_pos['x']
                    dy = pos['y'] - prev_pos['y']
                
                # Calculate yaw angle and convert to quaternion
                # Add small threshold to avoid issues with zero movement
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    # No movement, use default orientation (facing forward)
                    yaw = 0.0
                else:
                    yaw = math.atan2(dy, dx)
                
                pose_stamped.pose.orientation = self._yaw_to_quaternion(yaw)
                
                path_msg.poses.append(pose_stamped)
            
            self.path_pub.publish(path_msg)
    
    def publish_visualization_markers(self, predictions, timestamp):
        """
        Create and publish comprehensive RViz visualization markers
        
        The predictions should already be in world coordinates (map frame) after proper
        inverse transformation from the model output. These markers will visualize
        the trajectory at the correct world positions.
        """
        marker_array = MarkerArray()
        marker_id = 0
        
        # Only visualize person 0 (configurable)
        for person_id, trajectory in predictions.items():
            if str(person_id) != '0' and person_id != 0:
                continue
            
            # Add trajectory path line
            marker_id = self._add_trajectory_line(marker_array, trajectory, timestamp, marker_id, person_id)
            
            # Add trajectory points as spheres
            marker_id = self._add_trajectory_spheres(marker_array, trajectory, timestamp, marker_id)
            
            # Add person ID label
            marker_id = self._add_person_label(marker_array, trajectory, timestamp, marker_id, person_id)
            
            # Add ground plane projection
            marker_id = self._add_ground_projection(marker_array, trajectory, timestamp, marker_id)
        
        self.visualization_pub.publish(marker_array)
    
    def _add_trajectory_line(self, marker_array, trajectory, timestamp, marker_id, person_id):
        """Add line strip marker for trajectory path"""
        trajectory_marker = Marker()
        trajectory_marker.header.frame_id = "map"
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
            point.z = pos['z'] if pos['z'] != 0.0 else 0.05
            trajectory_marker.points.append(point)
        
        marker_array.markers.append(trajectory_marker)
        return marker_id + 1
    
    def _add_trajectory_spheres(self, marker_array, trajectory, timestamp, marker_id):
        """Add sphere markers for each predicted position"""
        for i, pos in enumerate(trajectory):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "map"
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
            scale = 0.15 - (i * 0.01)
            sphere_marker.scale.x = scale
            sphere_marker.scale.y = scale
            sphere_marker.scale.z = scale
            
            # Color gradient: red to yellow over time
            sphere_marker.color.r = 1.0
            sphere_marker.color.g = min(1.0, i * 0.2)
            sphere_marker.color.b = 0.0
            sphere_marker.color.a = 0.7
            
            marker_array.markers.append(sphere_marker)
            marker_id += 1
        
        return marker_id
    
    def _add_person_label(self, marker_array, trajectory, timestamp, marker_id, person_id):
        """Add text label for person ID"""
        if not trajectory:
            return marker_id
            
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = timestamp
        text_marker.ns = "trajectory_labels"
        text_marker.id = marker_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # Position text at the end of trajectory
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
        
        text_marker.text = "Person {}".format(person_id)
        
        marker_array.markers.append(text_marker)
        return marker_id + 1
    
    def _add_ground_projection(self, marker_array, trajectory, timestamp, marker_id):
        """Add ground plane projection of trajectory"""
        ground_marker = Marker()
        ground_marker.header.frame_id = "map"
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
        return marker_id + 1
    
    def _yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle (rotation around z-axis) to quaternion
        
        Args:
            yaw: Yaw angle in radians
            
        Returns:
            geometry_msgs/Quaternion
        """
        # For rotation around z-axis only: q = [0, 0, sin(yaw/2), cos(yaw/2)]
        half_yaw = yaw * 0.5
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = math.sin(half_yaw)
        quat.w = math.cos(half_yaw)
        return quat


# Utility functions for message creation
def create_pose_from_trajectory_point(point):
    """Create a ROS Pose from trajectory point"""
    pose = Pose()
    pose.position.x = point['x']
    pose.position.y = point['y']
    pose.position.z = point['z']
    pose.orientation.w = 1.0
    return pose


def create_point_from_trajectory_point(point, z_offset=0.0):
    """Create a ROS Point from trajectory point with optional z offset"""
    ros_point = Point()
    ros_point.x = point['x']
    ros_point.y = point['y']
    ros_point.z = point['z'] + z_offset if point['z'] != 0.0 else z_offset
    return ros_point


def format_predictions_for_json(predictions, timestamp):
    """Format predictions for JSON serialization"""
    return {
        'timestamp': timestamp.to_sec() if hasattr(timestamp, 'to_sec') else timestamp,
        'predictions': predictions,
        'num_persons': len(predictions),
        'frame_id': 'map'
    }