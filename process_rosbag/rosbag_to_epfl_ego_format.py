#!/usr/bin/env python
"""
ROS Bag to EPFL Ego Format Converter
This script reads a ROS bag file and extracts the /tf topic data to a text file.
"""

import os
import sys
import argparse
import rosbag
from tf2_msgs.msg import TFMessage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract /tf topic data from ROS bag file to text file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input ROS bag file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output text file path for /tf data (can be a directory or full file path)"
    )
    
    return parser.parse_args()


def extract_tf_data(bag_file, output_file):
    """
    Extract /tf topic data from ROS bag file and save to text file.
    Only includes transforms with frame_id: map, odom, or base_footprint.
    
    Args:
        bag_file (str): Path to the input ROS bag file
        output_file (str): Path to the output text file
    """
    # Define the frame IDs we want to filter for
    target_frame_ids = {"map", "odom", "base_footprint"}
    
    try:
        # Open the bag file
        with rosbag.Bag(bag_file, 'r') as bag:
            # Check if /tf topic exists in the bag
            topics = bag.get_type_and_topic_info()[1].keys()
            if '/tf' not in topics:
                print("Error: /tf topic not found in the bag file.")
                print("Available topics:")
                for topic in topics:
                    print("  - {}".format(topic))
                return False
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Extract /tf data and write to text file
            with open(output_file, 'w') as f:
                # TF Data extracted from: bag_file
                # Filtered for frame_ids: map, odom, base_footprin
                # Format: timestamp, frame_id, child_frame_id, translation(x,y,z), rotation(x,y,z,w)
                # Each line represents one transform
                
                message_count = 0
                filtered_count = 0
                
                for topic, msg, t in bag.read_messages(topics=['/tf']):
                    # Handle TFMessage which contains multiple transforms
                    for transform in msg.transforms:
                        frame_id = transform.header.frame_id
                        
                        # Filter: only process transforms with target frame_ids
                        if frame_id in target_frame_ids:
                            timestamp = t.to_sec()
                            child_frame_id = transform.child_frame_id
                            
                            # Translation
                            tx = transform.transform.translation.x
                            ty = transform.transform.translation.y
                            tz = transform.transform.translation.z
                            
                            # Rotation (quaternion)
                            rx = transform.transform.rotation.x
                            ry = transform.transform.rotation.y
                            rz = transform.transform.rotation.z
                            rw = transform.transform.rotation.w
                            
                            # Write to file
                            f.write("{:.6f},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                                timestamp, frame_id, child_frame_id, tx, ty, tz, rx, ry, rz, rw
                            ))
                            
                            filtered_count += 1
                        
                        message_count += 1
                
            print("Total transform messages processed: {}".format(message_count))
            print("Filtered messages (map, odom, base_footprint): {}".format(filtered_count))
            print("Output saved to: {}".format(output_file))
            return True
            
    except Exception as e:
        print("Error processing bag file: {}".format(e))
        return False


def main():
    """Main function to extract /tf data from ROS bag file."""
    args = parse_arguments()
    
    input_bag = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    # Validate input file
    if not os.path.exists(input_bag):
        print("Error: Input bag file '{}' does not exist.".format(input_bag))
        sys.exit(1)
    
    if not input_bag.endswith('.bag'):
        print("Warning: Input file doesn't have .bag extension")
    
    # Handle output path - if it's a directory, generate filename
    if os.path.isdir(output_path):
        # Generate output filename based on input bag filename
        bag_basename = os.path.basename(input_bag)
        bag_name_without_ext = os.path.splitext(bag_basename)[0]
        output_filename = "{}_tf_data.txt".format(bag_name_without_ext)
        output_txt = os.path.join(output_path, output_filename)
    else:
        output_txt = output_path
    
    print("Input bag file: {}".format(input_bag))
    print("Output text file: {}".format(output_txt))
    print()
    
    # Extract /tf data
    success = extract_tf_data(input_bag, output_txt)
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()