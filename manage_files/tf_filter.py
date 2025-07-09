#!/usr/bin/env python
"""
TF Filter Script
This script reads a ROS bag file and extracts transforms from /tf topic,
filtering to only include transforms with frame_id "map" or "odom" or "base_footprint".
    print("1. Extract transforms to YAML format (default):")
    print("   python tf_filter.py -i input.bag -o filtered_transforms.yaml")
    print()
    
    print("2. Extract transforms to JSON format:")
    print("   python tf_filter.py -i input.bag -o filtered_transforms.json -f json")
    print()
    
    print("3. Full path example:")
    print("   python tf_filter.py \\")
    print("     --input /path/to/rosbag/file.bag \\")
    print("     --output /path/to/output/transforms.yaml \\")
    print("     --format yaml")
    print()
    
    print("Script Features:")
    print("- Reads /tf topic from ROS bag files")
    print("- Filters transforms with frame_id 'map' or 'odom'")
    print("- Outputs in YAML or JSON format")
    print("- Preserves the exact structure as shown in your example")
    print("- Each transform is written as a separate document (YAML) or array element (JSON)")
    print()
"""
import os
import sys
import argparse
import json
import yaml
from collections import OrderedDict

# Try to import rosbag, provide helpful error message if not available
try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    print("Error: rosbag module not found. Please install it with:")
    print("pip install rospkg rosbag")
    print("or ensure ROS is properly installed and sourced.")
    ROSBAG_AVAILABLE = False
    sys.exit(1)


def extract_tf_transforms(bag_file, output_file, output_format='yaml'):
    """
    Extract transforms from /tf topic and filter by frame_id.
    
    Args:
        bag_file (str): Path to the input ROS bag file
        output_file (str): Path to the output file
        output_format (str): Output format - 'yaml' or 'json'
    """
    if not ROSBAG_AVAILABLE:
        raise ImportError("rosbag module is not available. Please install ROS or the rosbag package.")
    
    try:
        filtered_transforms = []
        
        with rosbag.Bag(bag_file, 'r') as bag:
            # Check if /tf topic exists in the bag
            topics = bag.get_type_and_topic_info()[1].keys()
            if '/tf' not in topics:
                print("Error: /tf topic not found in the bag file.")
                print("Available topics:")
                for topic in topics:
                    print("  - {}".format(topic))
                return False
            
            print("Reading /tf topic from bag file: {}".format(bag_file))
            
            # Read all /tf messages
            tf_count = 0
            filtered_count = 0
            
            for topic, msg, timestamp in bag.read_messages(topics=['/tf']):
                tf_count += 1
                
                # Process each transform in the message
                for transform in msg.transforms:
                    frame_id = transform.header.frame_id
                    
                    # Filter transforms with frame_id "map" or "odom"
                    if frame_id in ["map", "odom", "base_footprint"]:
                        filtered_count += 1
                        
                        # Convert transform to dictionary
                        transform_dict = OrderedDict([
                            ("transforms", [OrderedDict([
                                ("header", OrderedDict([
                                    ("seq", transform.header.seq),
                                    ("stamp", OrderedDict([
                                        ("secs", transform.header.stamp.secs),
                                        ("nsecs", transform.header.stamp.nsecs)
                                    ])),
                                    ("frame_id", transform.header.frame_id)
                                ])),
                                ("child_frame_id", transform.child_frame_id),
                                ("transform", OrderedDict([
                                    ("translation", OrderedDict([
                                        ("x", float(transform.transform.translation.x)),
                                        ("y", float(transform.transform.translation.y)),
                                        ("z", float(transform.transform.translation.z))
                                    ])),
                                    ("rotation", OrderedDict([
                                        ("x", float(transform.transform.rotation.x)),
                                        ("y", float(transform.transform.rotation.y)),
                                        ("z", float(transform.transform.rotation.z)),
                                        ("w", float(transform.transform.rotation.w))
                                    ]))
                                ]))
                            ])])
                        ])
                        
                        filtered_transforms.append(transform_dict)
                
                if tf_count % 100 == 0:
                    print("Processed {} /tf messages, found {} filtered transforms...".format(tf_count, filtered_count))
        
        print("Total /tf messages processed: {}".format(tf_count))
        print("Total filtered transforms (map/odom/base_footprint): {}".format(filtered_count))

        # Write output file
        if filtered_transforms:
            print("Writing {} transforms to: {}".format(len(filtered_transforms), output_file))
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w') as f:
                if output_format.lower() == 'json':
                    # Write as JSON array
                    json.dump(filtered_transforms, f, indent=2, sort_keys=False)
                else:
                    # Write as YAML documents separated by ---
                    for i, transform in enumerate(filtered_transforms):
                        if i > 0:
                            f.write("---\n")
                        yaml.dump(transform, f, default_flow_style=False, sort_keys=False)
            
            print("Successfully wrote {} filtered transforms to {}".format(len(filtered_transforms), output_file))
        else:
            print("No transforms with frame_id 'map', 'odom', or 'base_footprint' found in the bag file.")

        return True
        
    except Exception as e:
        print("Error processing bag file: {}".format(e))
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and filter transforms from /tf topic in ROS bag files"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input .bag file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=['yaml', 'json'],
        default='yaml',
        help="Output format (default: yaml)"
    )
    
    return parser.parse_args()


def main():
    """Main function to extract and filter TF transforms."""
    args = parse_arguments()
    
    input_file = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)
    output_format = args.format
    
    # Validate input file
    if not os.path.exists(input_file):
        print("Error: Input file '{}' does not exist.".format(input_file))
        sys.exit(1)
    
    if not input_file.endswith('.bag'):
        print("Error: Input file '{}' is not a .bag file.".format(input_file))
        sys.exit(1)
    
    print("=" * 60)
    print("TF Transform Filter")
    print("=" * 60)
    print("Input bag file: {}".format(input_file))
    print("Output file: {}".format(output_file))
    print("Output format: {}".format(output_format.upper()))
    print("Filter: frame_id in ['map', 'odom', 'base_footprint']")
    print("=" * 60)
    
    # Extract and filter transforms
    success = extract_tf_transforms(input_file, output_file, output_format)
    
    if success:
        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Processing failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
