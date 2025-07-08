#!/usr/bin/env python
"""
ROS Bag File Sorter
This script scans all .bag files in a specified directory for the /raw_bodies topic
and moves files containing this topic to a specified output folder.
"""

import os
import shutil
import subprocess
import sys
import glob
import argparse


def check_bag_for_topic(bag_file, topic_name="/raw_bodies"):
    """
    Check if a ROS bag file contains a specific topic.
    
    Args:
        bag_file (str): Path to the bag file
        topic_name (str): Name of the topic to search for
        
    Returns:
        bool: True if topic is found, False otherwise
    """
    try:
        # Use rosbag info command to get topics in the bag file
        proc = subprocess.Popen(
            ["rosbag", "info", "--yaml", bag_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate()
        
        if proc.returncode == 0:
            # Check if the topic is mentioned in the output
            return topic_name in stdout
        else:
            print("Error checking {}: {}".format(bag_file, stderr))
            return False
            
    except OSError:
        print("Error: rosbag command not found. Make sure ROS is installed and sourced.")
        sys.exit(1)
    except Exception as e:
        print("Error checking {}: {}".format(bag_file, e))
        return False


def move_to_good_folder(bag_file, output_folder):
    """
    Move a bag file to the output folder.
    
    Args:
        bag_file (str): Path to the bag file to move
        output_folder (str): Path to the output folder
    """
    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Move the file
        destination = os.path.join(output_folder, os.path.basename(bag_file))
        shutil.move(bag_file, destination)
        print("Moved {} to {}".format(bag_file, destination))
        
    except Exception as e:
        print("Error moving {}: {}".format(bag_file, e))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sort ROS bag files based on the presence of /raw_bodies topic"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=".",
        help="Input directory containing .bag files (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="good",
        help="Output directory for files containing /raw_bodies topic (default: 'good')"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="/raw_bodies",
        help="Topic name to search for (default: '/raw_bodies')"
    )
    
    return parser.parse_args()


def main():
    """Main function to process all bag files in the specified directory."""
    args = parse_arguments()
    
    input_dir = os.path.abspath(args.input)
    output_folder = os.path.abspath(args.output)
    topic_name = args.topic
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print("Error: Input directory '{}' does not exist.".format(input_dir))
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print("Error: '{}' is not a directory.".format(input_dir))
        sys.exit(1)
    
    print("Input directory: {}".format(input_dir))
    print("Output directory: {}".format(output_folder))
    print("Looking for topic: {}".format(topic_name))
    print()
    
    # Find all .bag files in the input directory
    bag_pattern = os.path.join(input_dir, "*.bag")
    bag_files = glob.glob(bag_pattern)
    
    if not bag_files:
        print("No .bag files found in the input directory: {}".format(input_dir))
        return
    
    print("Found {} bag files to check...".format(len(bag_files)))
    
    moved_count = 0
    
    for bag_file in bag_files:
        print("Checking {}...".format(os.path.basename(bag_file)))
        
        if check_bag_for_topic(bag_file, topic_name):
            print("Found {} topic in {}".format(topic_name, os.path.basename(bag_file)))
            move_to_good_folder(bag_file, output_folder)
            moved_count += 1
        else:
            print("No {} topic found in {}".format(topic_name, os.path.basename(bag_file)))
    
    print("\nProcessing complete!")
    print("Moved {} files to the output folder: {}".format(moved_count, output_folder))
    
    # Show remaining files
    remaining_files = glob.glob(bag_pattern)
    if remaining_files:
        print("Remaining files in input directory: {}".format(len(remaining_files)))
        for f in remaining_files:
            print("  - {}".format(os.path.basename(f)))


if __name__ == "__main__":
    main()