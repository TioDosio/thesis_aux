#!/usr/bin/env python
"""
ROS Bag Topic Message Analyzer
This script reads all .bag files in a directory and provides detailed analysis
of /image_detections and /raw_bodies topics, including:
- Total number of messages
- Number of empty vs non-empty messages  
- Total number of detections/poses across all messages
- Maximum detections/poses found in a single message

This helps debug frame count differences between ego and local format outputs.
"""

import os
import glob
import sys
import argparse

# Try to import rosbag, provide helpful error message if not available
try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    print("Error: rosbag module not found. Please install it with:")
    print("pip install rospkg rosbag")
    print("or ensure ROS is properly installed and sourced.")
    sys.exit(1)


def count_topic_messages(bag_file, topics_to_count):
    """
    Count messages in specified topics from a ROS bag file and analyze their content.
    
    Args:
        bag_file (str): Path to the .bag file
        topics_to_count (list): List of topic names to count
        
    Returns:
        dict: Dictionary with topic names as keys and analysis data as values
    """
    topic_analysis = {}
    
    for topic in topics_to_count:
        topic_analysis[topic] = {
            'total_messages': 0,
            'empty_messages': 0,
            'messages_with_data': 0,
            'total_detections': 0,
            'max_detections_per_message': 0
        }
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            # Get bag info to check available topics
            info = bag.get_type_and_topic_info()
            available_topics = list(info[1].keys())
            
            # Analyze messages for each requested topic
            for topic in topics_to_count:
                if topic in available_topics:
                    for _, msg, _ in bag.read_messages(topics=[topic]):
                        topic_analysis[topic]['total_messages'] += 1
                        
                        if topic == '/image_detections':
                            # Count persons in image_detections
                            num_persons = len(msg.persons) if hasattr(msg, 'persons') else 0
                            if num_persons == 0:
                                topic_analysis[topic]['empty_messages'] += 1
                            else:
                                topic_analysis[topic]['messages_with_data'] += 1
                            topic_analysis[topic]['total_detections'] += num_persons
                            topic_analysis[topic]['max_detections_per_message'] = max(
                                topic_analysis[topic]['max_detections_per_message'], num_persons
                            )
                            
                        elif topic == '/raw_bodies':
                            # Count poses in raw_bodies
                            num_poses = len(msg.poses) if hasattr(msg, 'poses') else 0
                            if num_poses == 0:
                                topic_analysis[topic]['empty_messages'] += 1
                            else:
                                topic_analysis[topic]['messages_with_data'] += 1
                            topic_analysis[topic]['total_detections'] += num_poses
                            topic_analysis[topic]['max_detections_per_message'] = max(
                                topic_analysis[topic]['max_detections_per_message'], num_poses
                            )
                    
        return topic_analysis
        
    except Exception as e:
        print("Error reading bag file {}: {}".format(bag_file, e))
        return topic_analysis


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze messages in /image_detections and /raw_bodies topics for all .bag files in a directory"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=".",
        help="Input directory containing .bag files (default: current directory)"
    )
    
    return parser.parse_args()


def process_all_bag_files(input_directory):
    """
    Process all .bag files in the specified directory and count topic messages.
    
    Args:
        input_directory (str): Path to directory containing .bag files
    """
    # Validate input directory
    if not os.path.exists(input_directory):
        print("Error: Directory '{}' does not exist.".format(input_directory))
        return
    
    if not os.path.isdir(input_directory):
        print("Error: '{}' is not a directory.".format(input_directory))
        return
    
    # Find all .bag files in the specified directory
    bag_pattern = os.path.join(input_directory, "*.bag")
    bag_files = glob.glob(bag_pattern)
    
    if not bag_files:
        print("No .bag files found in directory: {}".format(input_directory))
        return
    
    print("Found {} .bag files to process in: {}\n".format(len(bag_files), input_directory))
    
    # Topics to count
    topics_to_count = ['/image_detections', '/raw_bodies']
    
    # Process each bag file
    for bag_file in sorted(bag_files):
        bag_name = os.path.basename(bag_file)
        
        print("file: {}".format(bag_name))
        
        # Count messages in each topic
        topic_analysis = count_topic_messages(bag_file, topics_to_count)
        
        # Print detailed analysis for this file
        for topic in topics_to_count:
            analysis = topic_analysis[topic]
            print("  {}: {} total messages".format(topic, analysis['total_messages']))
            print("    - Empty messages: {}".format(analysis['empty_messages']))
            print("    - Messages with data: {}".format(analysis['messages_with_data']))
            print("    - Total detections/poses: {}".format(analysis['total_detections']))
            print("    - Max detections per message: {}".format(analysis['max_detections_per_message']))
        
        print("-" * 50)


def main():
    """Main function to analyze topic messages in all bag files."""
    args = parse_arguments()
    
    input_directory = os.path.abspath(args.input)
    
    print("ROS Bag Topic Message Analyzer")
    print("=" * 50)
    print("Analyzing messages in /image_detections and /raw_bodies topics")
    print("Input directory: {}".format(input_directory))
    print("=" * 50)
    
    process_all_bag_files(input_directory)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()