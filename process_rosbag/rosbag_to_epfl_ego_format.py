#!/usr/bin/env python
"""
ROS Bag to EPFL Ego Format Converter
This script reads a ROS bag file and creates EPFL ego format JSON using /raw_bodies and /tf topics.
"""

import os
import sys
import argparse
import json
import glob
import rosbag
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseArray
from collections import OrderedDict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create EPFL ego format JSON from ROS bag files using /raw_bodies and /tf topics"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing .bag files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for EPFL ego format JSON files"
    )
    
    return parser.parse_args()


def find_closest_odom_transform(timestamp, odom_transforms):
    """
    Find the latest odom transform that occurs before or at the given timestamp.
    
    Args:
        timestamp (float): Target timestamp
        odom_transforms (list): List of (timestamp, transform) tuples
        
    Returns:
        transform: The latest transform before the timestamp or None if not found
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
    
    Args:
        bag_file (str): Path to the input ROS bag file
        output_file (str): Path to the output JSON file
    """
    try:
        # Open the bag file
        with rosbag.Bag(bag_file, 'r') as bag:
            # Check if required topics exist in the bag
            topics = bag.get_type_and_topic_info()[1].keys()
            if '/raw_bodies' not in topics:
                print("Error: /raw_bodies topic not found in the bag file.")
                print("Available topics:")
                for topic in topics:
                    print("  - {}".format(topic))
                return False
            
            if '/tf' not in topics:
                print("Error: /tf topic not found in the bag file.")
                return False
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            print("Reading /tf topic for odom transforms...")
            # First pass: collect all odom transforms from /tf
            odom_transforms = []
            for topic, msg, t in bag.read_messages(topics=['/tf']):
                for transform in msg.transforms:
                    if transform.header.frame_id == 'odom':
                        timestamp = t.to_sec()
                        odom_transforms.append((timestamp, transform))
            
            print("Found {} odom transforms".format(len(odom_transforms)))
            
            # Second pass: process /raw_bodies messages
            print("Processing /raw_bodies messages...")
            frame_count = 0
            processed_count = 0
            
            with open(output_file, 'w') as f:
                for topic, msg, t in bag.read_messages(topics=['/raw_bodies']):
                    timestamp = t.to_sec()
                    
                    # Process each pose in the PoseArray
                    for pose in msg.poses:
                        # Find the closest odom transform
                        closest_odom = find_closest_odom_transform(timestamp, odom_transforms)
                        
                        if closest_odom:
                            # Create EPFL ego format entry using OrderedDict for consistent ordering
                            ego_entry = OrderedDict([
                                ("frame", frame_count),
                                ("coordinates", OrderedDict([
                                    ("x", pose.position.x),
                                    ("y", pose.position.y),
                                    ("z", pose.position.z),
                                    ("q1", closest_odom.transform.rotation.x),
                                    ("q2", closest_odom.transform.rotation.y),
                                    ("q3", closest_odom.transform.rotation.z),
                                    ("q4", closest_odom.transform.rotation.w)
                                ]))
                            ])
                            
                            # Write JSON line with consistent ordering
                            json_string = json.dumps(ego_entry, sort_keys=False)
                            f.write(json_string + '\n')
                            
                            frame_count += 1
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print("Processed {} /raw_bodies messages...".format(processed_count))
            
            print("Successfully processed {} /raw_bodies messages".format(processed_count))
            print("Generated {} frames in EPFL ego format".format(frame_count))
            print("Output saved to: {}".format(output_file))
            return True
            
    except Exception as e:
        print("Error processing bag file: {}".format(e))
        return False


def process_bag_folder(input_folder, output_folder):
    """
    Process all .bag files in a folder and create EPFL ego format JSON files.
    
    Args:
        input_folder (str): Path to folder containing .bag files
        output_folder (str): Path to folder for output JSON files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all .bag files in the input folder
    bag_pattern = os.path.join(input_folder, "*.bag")
    bag_files = glob.glob(bag_pattern)
    
    if not bag_files:
        print("No .bag files found in {}".format(input_folder))
        return
    
    print("Found {} .bag files to process".format(len(bag_files)))
    
    # Process each bag file
    successful_count = 0
    failed_count = 0
    successful_files = []
    failed_files = []
    
    for i, bag_file in enumerate(bag_files, 1):
        bag_name = os.path.splitext(os.path.basename(bag_file))[0]
        output_file = os.path.join(output_folder, "{}_epfl_ego.json".format(bag_name))
        
        print("\n[{}/{}] Processing: {}".format(i, len(bag_files), bag_name))
        print("=" * 60)
        
        try:
            success = extract_epfl_ego_data(bag_file, output_file)
            if success:
                successful_count += 1
                successful_files.append(bag_name)
                print("Successfully processed: {}".format(bag_name))
            else:
                failed_count += 1
                failed_files.append((bag_name, "Processing failed"))
                print("Failed to process: {}".format(bag_name))
            
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
    """Main function to process all .bag files in a folder."""
    args = parse_arguments()
    
    input_folder = os.path.abspath(args.input)
    output_folder = os.path.abspath(args.output)
    
    # Validate input folder
    if not os.path.exists(input_folder):
        print("Error: Input folder '{}' does not exist.".format(input_folder))
        sys.exit(1)
    
    if not os.path.isdir(input_folder):
        print("Error: '{}' is not a directory.".format(input_folder))
        sys.exit(1)
    
    print("=" * 60)
    print("ROS Bag to EPFL Ego Format Converter")
    print("=" * 60)
    print("Input folder: {}".format(input_folder))
    print("Output folder: {}".format(output_folder))
    print("=" * 60)
    
    # Process all bag files in the folder
    process_bag_folder(input_folder, output_folder)
    
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()