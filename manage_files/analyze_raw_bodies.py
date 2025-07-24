#!/usr/bin/env python
"""
Script to analyze ROS bag files and count messages with more than 1 person in /raw_bodies topic.
Compatible with Python 2.7 and ROS 1 Melodic.
"""

from __future__ import print_function
import rosbag
import os
import glob
import shutil
from collections import defaultdict
import sys

def analyze_rosbag(bag_path):
    """
    Analyze a single rosbag file and count messages with more than 1 person in /raw_bodies topic.
    
    Args:
        bag_path (str): Path to the bag file
        
    Returns:
        tuple: (total_messages, multi_person_messages)
    """
    try:
        bag = rosbag.Bag(bag_path, 'r')
        total_messages = 0
        multi_person_messages = 0
        
        # Check if /raw_bodies topic exists in the bag
        topics = bag.get_type_and_topic_info()[1].keys()
        if '/raw_bodies' not in topics:
            print("Warning: /raw_bodies topic not found in {}".format(os.path.basename(bag_path)))
            bag.close()
            return 0, 0
        
        # Read messages from /raw_bodies topic
        for topic, msg, t in bag.read_messages(topics=['/raw_bodies']):
            total_messages += 1
            
            # Count the number of poses (persons) in the message
            num_poses = len(msg.poses)
            
            if num_poses > 1:
                multi_person_messages += 1
        
        bag.close()
        return total_messages, multi_person_messages
        
    except Exception as e:
        print("Error processing {}: {}".format(bag_path, str(e)))
        return 0, 0

def main():
    """Main function to process all bag files in the directory."""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create ~/bags directory if it doesn't exist
    home_dir = os.path.expanduser("~")
    bags_dir = os.path.join(home_dir, "bags")
    if not os.path.exists(bags_dir):
        os.makedirs(bags_dir)
        print("Created directory: {}".format(bags_dir))
    
    # Find all .bag files in the current directory (excluding bad_ones subdirectory)
    bag_pattern = os.path.join(current_dir, "*.bag")
    bag_files = glob.glob(bag_pattern)
    
    if not bag_files:
        print("No .bag files found in the current directory")
        return
    
    # Sort bag files for consistent output
    bag_files.sort()
    
    print("=" * 80)
    print("ROS Bag Analysis: Counting messages with more than 1 person in /raw_bodies topic")
    print("=" * 80)
    print()
    
    # Statistics tracking
    total_bags = 0
    total_messages_all_bags = 0
    total_multi_person_all_bags = 0
    results = []
    bags_to_copy = []  # List to store bags with 0 multi-person messages
    
    # Process each bag file
    for bag_path in bag_files:
        bag_name = os.path.basename(bag_path)
        print("Processing: {}".format(bag_name))
        
        total_messages, multi_person_messages = analyze_rosbag(bag_path)
        
        if total_messages > 0:
            percentage = (multi_person_messages / total_messages) * 100
            print("  Total /raw_bodies messages: {}".format(total_messages))
            print("  Messages with >1 person: {}".format(multi_person_messages))
            print("  Percentage: {:.2f}%".format(percentage))
        else:
            percentage = 0.0
            print("  No /raw_bodies messages found")
        
        print()
        
        # Store results
        results.append({
            'filename': bag_name,
            'total_messages': total_messages,
            'multi_person_messages': multi_person_messages,
            'percentage': percentage
        })
        
        # Check if this bag should be copied (0 multi-person messages but has messages)
        if total_messages > 0 and multi_person_messages == 0:
            bags_to_copy.append(bag_path)
        
        total_bags += 1
        total_messages_all_bags += total_messages
        total_multi_person_all_bags += multi_person_messages
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Total bag files processed: {}".format(total_bags))
    print("Total /raw_bodies messages across all bags: {}".format(total_messages_all_bags))
    print("Total messages with >1 person across all bags: {}".format(total_multi_person_all_bags))
    
    if total_messages_all_bags > 0:
        overall_percentage = (total_multi_person_all_bags / total_messages_all_bags) * 100
        print("Overall percentage of multi-person messages: {:.2f}%".format(overall_percentage))
    else:
        print("No messages found across all bags")
    
    print()
    
    # Print detailed table
    print("Detailed Results:")
    print("-" * 80)
    print("{:<50} {:<8} {:<10} {:<8}".format('Filename', 'Total', '>1 Person', '%'))
    print("-" * 80)
    
    for result in results:
        print("{:<50} {:<8} {:<10} {:<8.2f}".format(result['filename'], result['total_messages'], result['multi_person_messages'], result['percentage']))
    
    print("-" * 80)
    
    # Copy bags with 0 multi-person messages to ~/bags
    if bags_to_copy:
        print()
        print("=" * 80)
        print("COPYING BAGS WITH 0 MULTI-PERSON MESSAGES TO ~/bags")
        print("=" * 80)
        
        for bag_path in bags_to_copy:
            bag_name = os.path.basename(bag_path)
            dest_path = os.path.join(bags_dir, bag_name)
            
            try:
                shutil.copy2(bag_path, dest_path)
                print("Copied: {} -> {}".format(bag_name, dest_path))
            except Exception as e:
                print("Error copying {}: {}".format(bag_name, str(e)))
        
        print()
        print("Total bags copied: {}".format(len(bags_to_copy)))
    else:
        print()
        print("No bags with 0 multi-person messages found to copy.")
    
    print()
    
    # Also analyze bad_ones directory if it exists
    bad_ones_dir = os.path.join(current_dir, "bad_ones")
    if os.path.exists(bad_ones_dir):
        print()
        print("=" * 80)
        print("ANALYZING BAD_ONES DIRECTORY")
        print("=" * 80)
        
        bad_bag_pattern = os.path.join(bad_ones_dir, "*.bag")
        bad_bag_files = glob.glob(bad_bag_pattern)
        bad_bag_files.sort()
        
        for bag_path in bad_bag_files:
            bag_name = os.path.basename(bag_path)
            print("Processing: bad_ones/{}".format(bag_name))
            
            total_messages, multi_person_messages = analyze_rosbag(bag_path)
            
            if total_messages > 0:
                percentage = (multi_person_messages / total_messages) * 100
                print("  Total /raw_bodies messages: {}".format(total_messages))
                print("  Messages with >1 person: {}".format(multi_person_messages))
                print("  Percentage: {:.2f}%".format(percentage))
            else:
                print("  No /raw_bodies messages found")
            print()

if __name__ == "__main__":
    main()
