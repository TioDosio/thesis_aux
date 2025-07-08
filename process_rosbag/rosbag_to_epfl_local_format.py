import yaml
import json
import os
import glob
import sys
from collections import OrderedDict, defaultdict

# Try to import rosbag, provide helpful error message if not available
try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    print("Error: rosbag module not found. Please install it with:")
    print("pip install rospkg rosbag")
    print("or ensure ROS is properly installed and sourced.")
    ROSBAG_AVAILABLE = False
    # Create a dummy rosbag class for syntax checking
    class rosbag:
        class Bag:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def get_type_and_topic_info(self):
                return {}, {}
            def read_messages(self, topics=None):
                return []


# Define the correct order of keypoints as specified by the pose detection model
CORRECT_ORDER = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
]


def read_rosbag_topics(bag_file, topics_to_read):
    """
    Read specified topics from a ROS bag file.

    This function reads the specified topics from a bag file and returns
    a dictionary organized by topic name containing all messages.

    Args:
        bag_file (str): Path to the .bag file
        topics_to_read (list): List of topic names to read

    Returns:
        dict: Dictionary with topic names as keys and lists of messages as values
    """
    if not ROSBAG_AVAILABLE:
        raise ImportError("rosbag module is not available. Please install ROS or the rosbag package.")
    
    topic_data = {topic: [] for topic in topics_to_read}
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            print("Reading bag file: {}".format(bag_file))
            
            # Get bag info
            info = bag.get_type_and_topic_info()
            available_topics = list(info[1].keys())
            
            print("Available topics: {}".format(available_topics))
            
            # Check which requested topics are available
            missing_topics = [topic for topic in topics_to_read if topic not in available_topics]
            if missing_topics:
                print("Warning: Topics not found in bag: {}".format(missing_topics))
            
            # Read messages from the specified topics
            for topic, msg, timestamp in bag.read_messages(topics=topics_to_read):
                topic_data[topic].append({
                    'msg': msg,
                    'timestamp': timestamp
                })
                
        print("Successfully read {} messages".format(sum(len(msgs) for msgs in topic_data.values())))
        return topic_data
        
    except Exception as e:
        print("Error reading bag file {}: {}".format(bag_file, e))
        return topic_data


def extract_image_detections_data(messages):
    """
    Extract keypoint detection data from /image_detections messages.

    Args:
        messages (list): List of message dictionaries from /image_detections topic

    Returns:
        dict: Dictionary with sequence numbers as keys and detection data as values
    """
    detections_data = {}
    
    for msg_data in messages:
        msg = msg_data['msg']
        
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
        
        detections_data[seq] = {
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
    
    return detections_data


def extract_raw_bodies_data(messages):
    """
    Extract pose data from /raw_bodies messages.

    Args:
        messages (list): List of message dictionaries from /raw_bodies topic

    Returns:
        dict: Dictionary with sequence numbers as keys and pose data as values
    """
    bodies_data = {}
    
    for msg_data in messages:
        msg = msg_data['msg']
        
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
        
        bodies_data[seq] = {
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
    
    return bodies_data


def extract_tf_data(messages):
    """
    Extract transform data from /tf messages.

    Args:
        messages (list): List of message dictionaries from /tf topic

    Returns:
        dict: Dictionary with timestamps as keys and transform data as values
    """
    tf_data = {}
    
    for msg_data in messages:
        msg = msg_data['msg']
        timestamp = msg_data['timestamp']
        
        # Process all transforms in the message
        for transform in msg.transforms:
            tf_dict = {
                'header': {
                    'seq': transform.header.seq,
                    'stamp': {
                        'secs': transform.header.stamp.secs,
                        'nsecs': transform.header.stamp.nsecs
                    },
                    'frame_id': transform.header.frame_id
                },
                'child_frame_id': transform.child_frame_id,
                'transform': {
                    'translation': {
                        'x': transform.transform.translation.x,
                        'y': transform.transform.translation.y,
                        'z': transform.transform.translation.z
                    },
                    'rotation': {
                        'x': transform.transform.rotation.x,
                        'y': transform.transform.rotation.y,
                        'z': transform.transform.rotation.z,
                        'w': transform.transform.rotation.w
                    }
                }
            }
            
            # Use timestamp as key for tf data
            tf_data[timestamp] = tf_dict
    
    return tf_data


def parse_yaml_file(filename):
    """
    DEPRECATED: This function is kept for backward compatibility but is no longer used.
    Use read_rosbag_topics() instead.
    """
    print("Warning: parse_yaml_file() is deprecated. Use read_rosbag_topics() instead.")
    return []


def extract_keypoints(person):
    """
    Extract keypoints from a person detection in the correct order.

    This function takes a person dictionary containing body_parts and reorganizes
    the keypoints according to the CORRECT_ORDER specification. Missing keypoints
    are filled with [0.0, 0.0, 0.0] (x, y, confidence).

    Args:
        person (dict): Person detection dictionary containing 'body_parts' key

    Returns:
        list: Flattened list of keypoints in format [x1, y1, c1, x2, y2, c2, ...]
              where each triplet represents (x_coordinate, y_coordinate, confidence)
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

    This function calculates the minimum bounding box that contains all
    keypoints with confidence > 0. The bounding box is represented as
    [x_min, y_min, x_max, y_max].

    Args:
        keypoints (list): Flattened list of keypoints [x1, y1, c1, x2, y2, c2, ...]

    Returns:
        list: Bounding box coordinates [x_min, y_min, x_max, y_max]
              Returns [0.0, 0.0, 0.0, 0.0] if no valid keypoints found
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


def extract_poses(seq, bodies_data):
    """
    Extract pose information for a specific sequence number from processed data.

    This function searches through the processed bodies data to find the pose data
    corresponding to a specific sequence number (frame).

    Args:
        seq (int): Sequence number (frame number) to search for
        bodies_data (dict): Dictionary containing processed raw_bodies data

    Returns:
        list: List of pose dictionaries, each containing 'position' and 'orientation'
              Returns empty list if sequence not found
    """
    if seq in bodies_data:
        return bodies_data[seq]['poses']
    return []


def process_rosbag_data(bag_file, output_file):
    """
    Main processing function that reads bag file and combines keypoint and pose data.

    This function reads the bag file, processes the data frame by frame,
    and outputs the combined result in JSON format following the EPFL structure.

    Args:
        bag_file (str): Path to the .bag file to process
        output_file (str): Path for the output JSON file
    """
    # Define the topics to read
    topics_to_read = ['/image_detections', '/raw_bodies', '/tf']
    
    print("Processing bag file: {}".format(bag_file))
    print("Topics to read: {}".format(topics_to_read))
    
    # Read data from bag file
    topic_data = read_rosbag_topics(bag_file, topics_to_read)
    
    # Extract structured data from each topic
    detections_data = extract_image_detections_data(topic_data.get('/image_detections', []))
    bodies_data = extract_raw_bodies_data(topic_data.get('/raw_bodies', []))
    tf_data = extract_tf_data(topic_data.get('/tf', []))
    
    print("Extracted {} detection frames".format(len(detections_data)))
    print("Extracted {} body frames".format(len(bodies_data)))
    print("Extracted {} tf transforms".format(len(tf_data)))
    
    # Use defaultdict to group data by frame sequence number
    frame_data = defaultdict(list)
    
    # Process all detection frames
    for seq, detection in detections_data.items():
        persons = detection.get('persons', [])
        
        if persons:  # Only process if there are persons detected
            print("Processing frame {} with {} person(s)".format(seq, len(persons)))
            
            # Get poses for this frame
            poses_result = extract_poses(seq, bodies_data)
            
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
                coordinate_entry = OrderedDict([
                    ("id", person_idx + 1),  # Person ID starting from 1
                    ("x", x),
                    ("y", y),
                    ("z", z),
                    ("bbox", bbox),
                    ("keypoints", keypoints)
                ])
                
                frame_data[seq].append(coordinate_entry)
        else:
            # Frame exists but no persons detected - ensure frame is included with empty coordinates
            print("Frame {}: No persons detected".format(seq))
            frame_data[seq] = []
    
    # Write output in EPFL format
    print("Writing output to: {}".format(output_file))
    
    try:
        with open(output_file, 'w') as out_f:
            for frame_number in sorted(frame_data.keys()):
                coordinates = frame_data[frame_number]
                
                # Create frame entry following EPFL format
                frame_entry = OrderedDict([
                    ("frame", frame_number),
                    ("coordinates", coordinates)
                ])
                
                # Write as JSON line
                json_string = json.dumps(frame_entry, sort_keys=False)
                out_f.write(json_string + '\n')
        
        print("Successfully processed {} frames".format(len(frame_data)))
        print("Output saved to: {}".format(output_file))
        
    except IOError as e:
        print("Error writing output file: {}".format(e))


def process_bag_folder(folder_path, output_folder):
    """
    Process all .bag files in a folder.

    Args:
        folder_path (str): Path to folder containing .bag files
        output_folder (str): Path to folder for output JSON files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all .bag files in the folder
    bag_pattern = os.path.join(folder_path, "*.bag")
    bag_files = glob.glob(bag_pattern)
    
    if not bag_files:
        print("No .bag files found in {}".format(folder_path))
        return
    
    print("Found {} .bag files to process".format(len(bag_files)))
    
    # Process each bag file
    successful_count = 0
    failed_count = 0
    successful_files = []
    failed_files = []
    
    for i, bag_file in enumerate(bag_files, 1):
        bag_name = os.path.splitext(os.path.basename(bag_file))[0]
        output_file = os.path.join(output_folder, "{}_keypoints.json".format(bag_name))
        
        print("\n[{}/{}] Processing: {}".format(i, len(bag_files), bag_name))
        print("=" * 60)
        
        try:
            process_rosbag_data(bag_file, output_file)
            successful_count += 1
            successful_files.append(bag_name)
            print("Successfully processed: {}".format(bag_name))
            print("Progress: {}/{} done ({} successful, {} failed)".format(
                i, len(bag_files), successful_count, failed_count))
        except Exception as e:
            failed_count += 1
            failed_files.append((bag_name, str(e)))
            print("Error processing {}: {}".format(bag_name, e))
            print("Progress: {}/{} done ({} successful, {} failed)".format(
                i, len(bag_files), successful_count, failed_count))
        
        print("=" * 60)
    
    # Final summary
    print("\nFinal Results:")
    print("Total processed: {}".format(len(bag_files)))
    print("Successful: {}".format(successful_count))
    print("Failed: {}".format(failed_count))
    print("Success rate: {:.1f}%".format((successful_count * 100.0) / len(bag_files)))
    
    # Show failed files if any
    if failed_files:
        print("\nFailed files:")
        for i, (filename, error) in enumerate(failed_files, 1):
            print("  {}. {} - {}".format(i, filename, error))


def main():
    if len(sys.argv) < 2:
        print("Usage: python rosbag_to_epflFormat.py <folder>")
        print("<folder> - 'wild' or 'invited'")
        sys.exit(1)
    
    # Get arguments
    bag_folder = "../rosbag_automation/output/{}".format(sys.argv[1])
    output_folder = "../preprocessing_files/{}".format(sys.argv[1])

    # Validate input folder
    if not os.path.exists(bag_folder):
        print("Error: Bag folder does not exist: {}".format(bag_folder))
        sys.exit(1)
    
    print("=" * 60)
    print("ROS Bag Data Preprocessing Script")
    print("=" * 60)
    print("Input folder: {}".format(bag_folder))
    print("Output folder: {}".format(output_folder))
    print("=" * 60)
    
    # Process all bag files in the folder
    process_bag_folder(bag_folder, output_folder)
    
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
