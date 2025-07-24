#!/usr/bin/env python
"""
Complete Pipeline Processing Script
This script contains all the functionality to:
1. Process rosbag files to EPFL ego format
2. Process rosbag files to EPFL local format  
3. Organize the resulting JSON files into paired folders

All functionality is contained within this single script - no external script calls needed.

Usage:
    python complete_pipeline.py --input /path/to/rosbag/files --output /path/to/output/directory
    python complete_pipeline.py -i /path/to/rosbag/files -o /path/to/output/directory

The input directory should contain 'invited' and 'wild' subdirectories with .bag files.
The output directory will be created if it doesn't exist.
"""

import os
import sys
import glob
import shutil
import json
import argparse
from collections import defaultdict, OrderedDict
from datetime import datetime

# Try to import rosbag
try:
    import rosbag
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import PoseArray
    ROSBAG_AVAILABLE = True
except ImportError:
    print("Warning: rosbag module not found. Some functionality may be limited.")
    ROSBAG_AVAILABLE = False


# Define the correct order of keypoints as specified by the pose detection model
CORRECT_ORDER = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
]


def log_message(message, level="INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{0}] {1}: {2}".format(timestamp, level, message))


def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ========================
# EGO FORMAT PROCESSING
# ========================

def find_closest_odom_transform(timestamp, odom_transforms):
    """
    Find the latest odom transform that occurs before or at the given timestamp.
    """
    if not odom_transforms:
        return None
    
    # Filter transforms that occur before or at the target timestamp
    valid_transforms = [
        (tf_timestamp, transform) 
        for tf_timestamp, transform in odom_transforms 
        if tf_timestamp <= timestamp
    ]
    
    # If no transforms before the timestamp, return None
    if not valid_transforms:
        return None
    
    # Find the transform with the latest (maximum) timestamp
    latest_transform = max(valid_transforms, key=lambda x: x[0])
    return latest_transform[1]


def extract_epfl_ego_data(bag_file, output_file):
    """
    Extract /raw_bodies and /tf data from ROS bag file and create EPFL ego format JSON.
    """
    if not ROSBAG_AVAILABLE:
        log_message("rosbag module not available - skipping ego format processing", "ERROR")
        return False
    
    try:
        # Open the bag file
        with rosbag.Bag(bag_file, 'r') as bag:
            # Check if required topics exist in the bag
            topics = bag.get_type_and_topic_info()[1].keys()
            if '/raw_bodies' not in topics:
                log_message("Error: /raw_bodies topic not found in the bag file.", "ERROR")
                log_message("Available topics: {0}".format(list(topics)), "INFO")
                return False
            
            if '/tf' not in topics:
                log_message("Error: /tf topic not found in the bag file.", "ERROR")
                return False
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            log_message("Reading /tf topic for odom transforms...", "INFO")
            # First pass: collect all odom transforms from /tf
            odom_transforms = []
            for topic, msg, t in bag.read_messages(topics=['/tf']):
                for transform in msg.transforms:
                    if transform.header.frame_id == 'odom':
                        timestamp = t.to_sec()
                        odom_transforms.append((timestamp, transform))
            
            log_message("Found {0} odom transforms".format(len(odom_transforms)), "INFO")
            
            # Second pass: process /raw_bodies messages (one frame per message)
            log_message("Processing /raw_bodies messages...", "INFO")
            frame_count = 0
            processed_count = 0
            
            with open(output_file, 'w') as f:
                for topic, msg, t in bag.read_messages(topics=['/raw_bodies']):
                    timestamp = t.to_sec()
                    
                    # Find the closest odom transform for this message
                    closest_odom = find_closest_odom_transform(timestamp, odom_transforms)
                    
                    # Process this message (each message becomes one frame)
                    if closest_odom:
                        # If message has poses, use the first pose coordinates
                        # If no poses, use zero coordinates but keep the orientation from odom
                        x, y, z = 0.0, 0.0, 0.0
                        if msg.poses:
                            # Use the first pose in the message
                            first_pose = msg.poses[0]
                            x = first_pose.position.x
                            y = first_pose.position.y
                            z = first_pose.position.z
                        
                        ego_entry = OrderedDict([
                            ("frame", frame_count),
                            ("coordinates", OrderedDict([
                                ("x", x),
                                ("y", z), # EPFL ego format where y is the height
                                ("z", y),
                                ("q1", closest_odom.transform.rotation.x),
                                ("q2", closest_odom.transform.rotation.y),
                                ("q3", closest_odom.transform.rotation.z),
                                ("q4", closest_odom.transform.rotation.w)
                            ]))
                        ])
                        
                        # Write JSON line with consistent ordering
                        json_string = json.dumps(ego_entry, sort_keys=False)
                        f.write(json_string + '\n')
                    else:
                        # No odom transform found, but still create a frame entry with zero values
                        ego_entry = OrderedDict([
                            ("frame", frame_count),
                            ("coordinates", OrderedDict([
                                ("x", 0.0),
                                ("y", 0.0),
                                ("z", 0.0),
                                ("q1", 0.0),
                                ("q2", 0.0),
                                ("q3", 0.0),
                                ("q4", 1.0)  # Default quaternion
                            ]))
                        ])
                        
                        # Write JSON line with consistent ordering
                        json_string = json.dumps(ego_entry, sort_keys=False)
                        f.write(json_string + '\n')
                    
                    # Increment frame count once per message
                    frame_count += 1
                    processed_count += 1
                    if processed_count % 100 == 0:
                        log_message("Processed {0} /raw_bodies messages...".format(processed_count), "INFO")
            
            log_message("Successfully processed {0} /raw_bodies messages".format(processed_count), "INFO")
            log_message("Generated {0} frames (one per message) in EPFL ego format".format(frame_count), "INFO")
            return True
            
    except Exception as e:
        log_message("Error processing bag file: {0}".format(e), "ERROR")
        return False


# ========================
# LOCAL FORMAT PROCESSING
# ========================

def read_rosbag_topics(bag_file, topics_to_read):
    """
    Read specified topics from a ROS bag file.
    """
    if not ROSBAG_AVAILABLE:
        raise ImportError("rosbag module is not available.")
    
    topic_data = {topic: [] for topic in topics_to_read}
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            log_message("Reading bag file: {0}".format(bag_file), "INFO")
            
            # Get bag info
            info = bag.get_type_and_topic_info()
            available_topics = list(info[1].keys())
            
            log_message("Available topics: {0}".format(available_topics), "INFO")
            
            # Check which requested topics are available
            missing_topics = [topic for topic in topics_to_read if topic not in available_topics]
            if missing_topics:
                log_message("Warning: Topics not found in bag: {0}".format(missing_topics), "WARNING")
            
            # Read messages from the specified topics
            for topic, msg, timestamp in bag.read_messages(topics=topics_to_read):
                topic_data[topic].append({
                    'msg': msg,
                    'timestamp': timestamp
                })
                
        log_message("Successfully read {0} messages".format(sum(len(msgs) for msgs in topic_data.values())), "INFO")
        return topic_data
        
    except Exception as e:
        log_message("Error reading bag file {0}: {1}".format(bag_file, e), "ERROR")
        return topic_data


def extract_image_detections_data(messages):
    """
    Extract keypoint detection data from /image_detections messages.
    """
    detections_data = []
    
    for msg_data in messages:
        msg = msg_data['msg']
        timestamp = msg_data['timestamp']
        
        # Extract sequence number from header
        seq = msg.header.seq
        
        # Convert persons data to dictionary format similar to original
        persons = []
        for person in msg.persons:
            person_dict = {
                'id': person.id,
                'id_confidence': person.id_confidence,
                'body_parts': []
            }
            
            for body_part in person.body_parts:
                body_part_dict = {
                    'part_id': body_part.part_id,
                    'x': body_part.x,
                    'y': body_part.y,
                    'confidence': body_part.confidence
                }
                person_dict['body_parts'].append(body_part_dict)
            
            persons.append(person_dict)
        
        detection_entry = {
            'timestamp': timestamp.to_sec(),
            'seq': seq,
            'header': {
                'seq': seq,
                'stamp': {
                    'secs': msg.header.stamp.secs,
                    'nsecs': msg.header.stamp.nsecs
                },
                'frame_id': msg.header.frame_id
            },
            'persons': persons
        }
        
        detections_data.append(detection_entry)
    
    # Sort by timestamp to ensure consistent ordering
    detections_data.sort(key=lambda x: x['timestamp'])
    return detections_data


def extract_raw_bodies_data(messages):
    """
    Extract pose data from /raw_bodies messages.
    """
    bodies_data = []
    
    for msg_data in messages:
        msg = msg_data['msg']
        timestamp = msg_data['timestamp']
        
        # Extract sequence number from header
        seq = msg.header.seq
        
        # Convert poses data to dictionary format
        poses = []
        for pose in msg.poses:
            pose_dict = {
                'position': {
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z
                },
                'orientation': {
                    'x': pose.orientation.x,
                    'y': pose.orientation.y,
                    'z': pose.orientation.z,
                    'w': pose.orientation.w
                }
            }
            poses.append(pose_dict) 
        
        body_entry = {
            'timestamp': timestamp.to_sec(),
            'seq': seq,
            'header': {
                'seq': seq,
                'stamp': {
                    'secs': msg.header.stamp.secs,
                    'nsecs': msg.header.stamp.nsecs
                },
                'frame_id': msg.header.frame_id
            },
            'poses': poses
        }
        
        bodies_data.append(body_entry)
    
    # Sort by timestamp to ensure consistent ordering
    bodies_data.sort(key=lambda x: x['timestamp'])
    return bodies_data


def extract_keypoints(person):
    """
    Extract keypoints from a person detection in the correct order.
    """
    # Create a mapping from part_id to coordinates [x, y, confidence]
    part_map = {
        bp['part_id']: [float(bp['x']), float(bp['y']), float(bp['confidence'])]
        for bp in person.get('body_parts', [])
    }

    # Build keypoints list in the correct order
    keypoints = []
    for part in CORRECT_ORDER:
        # If keypoint exists, use it; otherwise use [0.0, 0.0, 0.0]
        keypoints.extend(part_map.get(part, [0.0, 0.0, 0.0]))

    return keypoints


def compute_bbox(keypoints):
    """
    Compute bounding box from valid keypoints.
    """
    xs, ys = [], []

    # Extract x, y coordinates from keypoints with confidence > 0
    for i in range(0, len(keypoints), 3):
        x, y, confidence = keypoints[i], keypoints[i+1], keypoints[i+2]
        if confidence > 0:
            xs.append(x)
            ys.append(y)

    # Return zero bounding box if no valid keypoints found
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]

    # Calculate bounding box coordinates
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return [x_min, y_min, x_max, y_max]


def extract_poses(timestamp, bodies_data):
    """
    Extract pose information for a specific timestamp from processed data.
    """
    # Find the closest body data by timestamp
    closest_body = None
    min_time_diff = float('inf')
    
    for body_entry in bodies_data:
        time_diff = abs(body_entry['timestamp'] - timestamp)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_body = body_entry
    
    if closest_body:
        return closest_body['poses']
    
    return []


def process_rosbag_to_local_format(bag_file, output_file):
    """
    Main processing function that reads bag file and combines keypoint and pose data.
    """
    if not ROSBAG_AVAILABLE:
        log_message("rosbag module not available - skipping local format processing", "ERROR")
        return False
    
    # Define the topics to read
    topics_to_read = ['/image_detections', '/raw_bodies']

    log_message("Processing bag file: {0}".format(bag_file), "INFO")
    log_message("Topics to read: {0}".format(topics_to_read), "INFO")

    # Read data from bag file
    topic_data = read_rosbag_topics(bag_file, topics_to_read)
    
    # Extract structured data from each topic (now returns lists sorted by timestamp)
    detections_data = extract_image_detections_data(topic_data.get('/image_detections', []))
    bodies_data = extract_raw_bodies_data(topic_data.get('/raw_bodies', []))
    
    try:
        with open(output_file, 'w') as out_f:
            # Process each detection message as a separate frame
            for frame_idx, detection in enumerate(detections_data):
                coordinates = []
                
                persons = detection.get('persons', [])
                detection_timestamp = detection['timestamp']
                
                if persons:
                    # Get poses for this frame (find closest by timestamp)
                    poses_result = extract_poses(detection_timestamp, bodies_data)
                    
                    # Process each person in this frame
                    for person_idx, person in enumerate(persons):
                        # Extract keypoints in the correct order
                        keypoints = extract_keypoints(person)
                        
                        # Compute bounding box from keypoints
                        bbox = compute_bbox(keypoints)
                        
                        # Extract position coordinates from corresponding pose if available
                        x, y, z = 0.0, 0.0, 0.0
                        if poses_result and person_idx < len(poses_result):
                            position = poses_result[person_idx].get('position', {})
                            x = position.get('x', 0.0)
                            y = position.get('y', 0.0)
                            z = position.get('z', 0.0)
                        
                        # Create coordinate entry for this person
                        # Assign ID in reverse order: if 4 persons, IDs are 4, 3, 2, 1
                        person_id = len(persons) - person_idx
                        coordinate_entry = OrderedDict([
                            ("id", person_id),
                            ("x", x),
                            ("y", z),   # y is the height coordinate in EPFL format
                            ("z", y),
                            ("bbox", bbox),
                            ("keypoints", keypoints)
                        ])
                        
                        coordinates.append(coordinate_entry)
                
                # Create frame entry following EPFL format (frame number starts from 0)
                frame_entry = OrderedDict([
                    ("frame", frame_idx),
                    ("coordinates", coordinates)
                ])
                
                # Write as JSON line
                json_string = json.dumps(frame_entry, sort_keys=False)
                out_f.write(json_string + '\n')
        
        log_message("Successfully processed {0} frames".format(len(detections_data)), "INFO")
        return True
        
    except IOError as e:
        log_message("Error writing output file: {0}".format(e), "ERROR")
        return False


# ========================
# FOLDER ORGANIZATION
# ========================

def organize_json_pairs(base_path, folder_types=['invited', 'wild']):
    """
    Organize pairs of JSON files into folders based on their common name prefix.
    """
    log_message("Organizing JSON file pairs", "INFO")
    
    for folder_name in folder_types:
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            log_message("Folder {0} does not exist, skipping...".format(folder_path), "WARNING")
            continue
            
        log_message("Processing folder: {0}".format(folder_name), "INFO")
        
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        if not json_files:
            log_message("No JSON files found in {0}".format(folder_name), "INFO")
            continue
        
        # Group files by their common prefix
        file_groups = defaultdict(list)
        
        for file in json_files:
            # Remove the specific suffixes to get the common prefix
            if file.endswith('_ego_coordinates.json'):
                prefix = file[:-len('_ego_coordinates.json')]
                file_groups[prefix].append(file)
            elif file.endswith('_local_coordinates.json'):
                prefix = file[:-len('_local_coordinates.json')]
                file_groups[prefix].append(file)
        
        # Create folders and move files
        for prefix, files in file_groups.items():
            if len(files) == 2:  # Only process if we have a complete pair
                # Create the new folder
                new_folder_path = os.path.join(folder_path, prefix)
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                
                log_message("Created folder: {0}".format(prefix), "INFO")
                
                # Move both files to the new folder
                for file in files:
                    src_path = os.path.join(folder_path, file)
                    dst_path = os.path.join(new_folder_path, file)
                    try:
                        shutil.move(src_path, dst_path)
                        log_message("  Moved: {0}".format(file), "INFO")
                    except Exception as e:
                        log_message("  Error moving {0}: {1}".format(file, e), "ERROR")
            else:
                log_message("Warning: Found {0} files for prefix '{1}' (expected 2): {2}".format(len(files), prefix, files), "WARNING")


# ========================
# MAIN PROCESSING PIPELINE
# ========================

def process_bag_files_in_folder(input_folder, output_folder, folder_type):
    """
    Process all bag files in a folder to generate both ego and local format JSON files.
    """
    # Find all .bag files in the input folder
    bag_pattern = os.path.join(input_folder, "*.bag")
    bag_files = glob.glob(bag_pattern)
    
    if not bag_files:
        log_message("No .bag files found in {0}".format(input_folder), "WARNING")
        return 0, 0
    
    log_message("Found {0} .bag files in {1}".format(len(bag_files), folder_type), "INFO")
    
    # Ensure output directory exists
    ensure_directory_exists(output_folder)
    
    # Process each bag file
    successful_count = 0
    failed_count = 0
    
    for i, bag_file in enumerate(bag_files, 1):
        bag_name = os.path.splitext(os.path.basename(bag_file))[0]
        
        ego_output_file = os.path.join(output_folder, "{0}_ego_coordinates.json".format(bag_name))
        local_output_file = os.path.join(output_folder, "{0}_local_coordinates.json".format(bag_name))
        
        log_message("[{0}/{1}] Processing: {2}".format(i, len(bag_files), bag_name), "INFO")
        
        try:
            # Process ego format
            ego_success = extract_epfl_ego_data(bag_file, ego_output_file)
            
            # Process local format
            local_success = extract_epfl_local_format(bag_file, local_output_file)
            
            if ego_success and local_success:
                successful_count += 1
                log_message(" Successfully processed: {0}".format(bag_name), "INFO")
            else:
                failed_count += 1
                log_message("  Failed to process: {0}".format(bag_name), "ERROR")
            
        except Exception as e:
            failed_count += 1
            log_message("  Error processing {0}: {1}".format(bag_name, e), "ERROR")
    
    return successful_count, failed_count


def extract_epfl_local_format(bag_file, output_file):
    """
    Wrapper function for local format processing with proper error handling.
    """
    try:
        return process_rosbag_to_local_format(bag_file, output_file)
    except Exception as e:
        log_message("Error in local format processing: {0}".format(e), "ERROR")
        return False


def main():
    """Main pipeline function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Complete Pipeline Processing Script')
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory containing rosbag files (should have "invited" and "wild" subdirectories)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory where processed JSON files will be saved')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        log_message("ERROR: Input directory does not exist: {0}".format(args.input), "ERROR")
        return False
    
    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output)
    
    log_message("COMPLETE PIPELINE PROCESSING - STARTING", "INFO")
    log_message("="*60, "INFO")
    log_message("Input directory: {0}".format(args.input), "INFO")
    log_message("Output directory: {0}".format(args.output), "INFO")
    
    if not ROSBAG_AVAILABLE:
        log_message("ERROR: rosbag module not available. Please install ROS or rosbag package.", "ERROR")
        log_message("Installation: pip install rospkg rosbag", "INFO")
        return False
    
    # Define paths using command line arguments
    rosbag_input_base = args.input
    preprocessing_output_base = args.output
    
    # Check if rosbag input directories exist
    for folder_type in ['invited', 'wild']:
        input_folder = os.path.join(rosbag_input_base, folder_type)
        if not os.path.exists(input_folder):
            log_message("Input folder does not exist: {0}".format(input_folder), "WARNING")
    
    total_successful = 0
    total_failed = 0
    
    # Process both invited and wild folders
    for folder_type in ['invited', 'wild']:
        log_message("Processing {0} folder".format(folder_type.upper()), "INFO")
        log_message("="*50, "INFO")
        
        input_folder = os.path.join(rosbag_input_base, folder_type)
        output_folder = os.path.join(preprocessing_output_base, folder_type)
        
        # Skip if input folder doesn't exist
        if not os.path.exists(input_folder):
            log_message("Skipping {0} - input folder does not exist".format(folder_type), "WARNING")
            continue
        
        # Process bag files
        successful, failed = process_bag_files_in_folder(input_folder, output_folder, folder_type)
        total_successful += successful
        total_failed += failed
        
        log_message("{0} processing complete: {1} successful, {2} failed".format(folder_type, successful, failed), "INFO")
    
    # Organize JSON files into pairs
    log_message("Organizing JSON files into paired folders", "INFO")
    organize_json_pairs(preprocessing_output_base)
    
    # Final summary
    log_message("="*60, "INFO")
    log_message("PIPELINE PROCESSING COMPLETE", "INFO")
    log_message("="*60, "INFO")
    log_message("Total successful: {0}".format(total_successful), "INFO")
    log_message("Total failed: {0}".format(total_failed), "INFO")
    
    if total_failed == 0:
        log_message(" All operations completed successfully!", "INFO")
        return True
    else:
        log_message("{0} operations had issues".format(total_failed), "WARNING")
        return total_successful > 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log_message("Operation cancelled by user", "INFO")
        sys.exit(1)
    except Exception as e:
        log_message("Unexpected error: {0}".format(e), "ERROR")
        sys.exit(1)
