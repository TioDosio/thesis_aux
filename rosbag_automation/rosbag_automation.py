#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS 1 Melodic Rosbag Automation Script (Python 2.7 Compatible)
Processes rosbag files by playing them back with additional ROS nodes
and recording all topics into new rosbag files.

"""

import os
import sys
import subprocess
import time
import signal
import argparse
import glob
import logging
from datetime import datetime


class RosbagAutomation:
    """Automates rosbag processing with ROS nodes"""

    def __init__(self, input_folder,output_folder, catkin_ws,
                 launch_package="vizzy", launch_file="human_detection.launch",
                 startup_delay=3.0, recording_delay=2.0):
        """
        Initialize the automation system

        Args:
            input_folder (str): Path to folder containing input rosbag files
            output_folder (str): Path to folder for output rosbag files
            catkin_ws (str): Path to catkin workspace
            launch_package (str): ROS package containing launch file
            launch_file (str): Launch file name
            startup_delay (float): Seconds to wait after launching node
            recording_delay (float): Seconds to wait after starting recording
        """
        self.input_folder = os.path.expanduser(input_folder)
        self.output_folder = os.path.expanduser(output_folder)
        self.catkin_ws = os.path.expanduser(catkin_ws)
        self.launch_package = launch_package
        self.launch_file = launch_file
        self.startup_delay = startup_delay
        self.recording_delay = recording_delay

        # Process tracking
        self.processes = []
        self.roscore_process = None
        self.node_process = None
        self.record_process = None
        self.play_process = None

        # Setup logging
        self.setup_logging()

        # Validate paths
        self.validate_paths()

    def setup_logging(self):
        """Configure logging for the automation"""
        log_file = os.path.join(self.output_folder, 'rosbag_automation.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_paths(self):
        """Validate that required paths exist"""
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError("Input folder does not exist: {}".format(self.input_folder))

        if not os.path.exists(self.catkin_ws):
            raise FileNotFoundError("Catkin workspace does not exist: {}".format(self.catkin_ws))

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            self.logger.info("Created output folder: {}".format(self.output_folder))

        # Check launch file path
        launch_path = os.path.join(
            self.catkin_ws, 
            "src", 
            self.launch_package, 
            "human_detection_3d", 
            "launch", 
            self.launch_file
        )
        if not os.path.exists(launch_path):
            raise FileNotFoundError("Launch file does not exist: {}".format(launch_path))

    def signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.logger.info("Received signal {}. Cleaning up...".format(signum))
        self.cleanup_processes()
        sys.exit(0)

    def cleanup_processes(self):
        """Clean up all running processes"""
        processes_to_cleanup = [
            (self.play_process, "rosbag play"),
            (self.record_process, "rosbag record"),
            (self.node_process, "roslaunch"),
            (self.roscore_process, "roscore")
        ]

        for process, name in processes_to_cleanup:
            if process and process.poll() is None:
                self.logger.info("Terminating {}...".format(name))
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Force killing {}...".format(name))
                    process.kill()
                except Exception as e:
                    self.logger.error("Error terminating {}: {}".format(name, e))

    def start_roscore(self):
        """Start roscore if not already running"""
        try:
            # Check if roscore is already running
            with open(os.devnull, 'w') as devnull:
                result = subprocess.call(
                    ["rostopic", "list"], 
                    stdout=devnull, 
                    stderr=devnull
                )
            if result == 0:
                self.logger.info("roscore is already running")
                return True
        except Exception:
            pass

        # Start roscore
        self.logger.info("Starting roscore...")
        with open(os.devnull, 'w') as devnull:
            self.roscore_process = subprocess.Popen(
                ["roscore"],
                stdout=devnull,
                stderr=devnull
            )

        # Wait for roscore to start
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                with open(os.devnull, 'w') as devnull:
                    result = subprocess.call(
                        ["rostopic", "list"],
                        stdout=devnull,
                        stderr=devnull
                    )
                if result == 0:
                    self.logger.info("roscore started successfully")
                    return True
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError("Failed to start roscore after {} attempts".format(max_attempts))

    def set_sim_time(self):
        """Set use_sim_time parameter to true"""
        self.logger.info("Setting use_sim_time to true...")
        try:
            subprocess.check_call(["rosparam", "set", "use_sim_time", "true"])
            self.logger.info("use_sim_time set to true")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed to set use_sim_time: {}".format(e))

    def launch_node(self):
        """Launch the ROS node"""
        launch_dir = os.path.join(
            self.catkin_ws,
            "src",
            self.launch_package,
            "human_detection_3d",
            "launch"
        )

        self.logger.info("Launching node from directory: {}".format(launch_dir))

        # Change to launch directory and run roslaunch
        self.node_process = subprocess.Popen(
            ["roslaunch", self.launch_file],
            cwd=launch_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for node to initialize
        self.logger.info("Waiting {:.1f} seconds for node to initialize...".format(self.startup_delay))
        time.sleep(self.startup_delay)

        # Check if node is still running
        if self.node_process.poll() is not None:
            stdout, stderr = self.node_process.communicate()
            raise RuntimeError("Node failed to start. Error: {}".format(stderr))

        self.logger.info("Node launched successfully")

    def get_active_topics(self):
        """Get list of currently active topics"""
        try:
            result = subprocess.check_output(["rostopic", "list"])
            topics = result.decode().strip().split('\n')
            self.logger.info("Found {} active topics".format(len(topics)))
            return topics
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to get topic list: {}".format(e))
            return []

    def start_recording(self, output_bag_path):
        """Start recording all topics"""
        self.logger.info("Starting recording to: {}".format(output_bag_path))

        # Use rosbag record -a to record all topics
        self.record_process = subprocess.Popen(
            ["rosbag", "record", "-a", "-O", output_bag_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for recording to start
        time.sleep(self.recording_delay)

        # Check if recording started successfully
        if self.record_process.poll() is not None:
            stdout, stderr = self.record_process.communicate()
            raise RuntimeError("Recording failed to start. Error: {}".format(stderr))

        self.logger.info("Recording started successfully")

    def play_rosbag(self, input_bag_path):
        """Play the input rosbag with --clock flag"""
        self.logger.info("Playing rosbag: {}".format(input_bag_path))

        # Start rosbag play with --clock flag
        self.play_process = subprocess.Popen(
            ["rosbag", "play", input_bag_path, "--clock"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for playback to complete
        stdout, stderr = self.play_process.communicate()

        if self.play_process.returncode != 0:
            raise RuntimeError("Rosbag play failed. Error: {}".format(stderr))

        self.logger.info("Rosbag playback completed")

    def stop_recording(self):
        """Stop the recording process"""
        if self.record_process and self.record_process.poll() is None:
            self.logger.info("Stopping recording...")
            self.record_process.send_signal(signal.SIGINT)

            # Wait for graceful shutdown
            try:
                self.record_process.wait(timeout=10)
                self.logger.info("Recording stopped successfully")
            except subprocess.TimeoutExpired:
                self.logger.warning("Recording did not stop gracefully, force killing...")
                self.record_process.kill()

    def process_single_rosbag(self, input_bag_path):
        """Process a single rosbag file"""
        bag_name = os.path.splitext(os.path.basename(input_bag_path))[0]
        output_bag_path = os.path.join(self.output_folder, "{}_new.bag".format(bag_name))

        self.logger.info("Processing: {} -> {}".format(input_bag_path, output_bag_path))

        try:
            # Start recording
            self.start_recording(output_bag_path)

            # Play the rosbag
            self.play_rosbag(input_bag_path)

            # Stop recording
            self.stop_recording()

            # Verify output file was created
            if os.path.exists(output_bag_path + ".bag"):
                final_path = output_bag_path + ".bag"
                self.logger.info("Successfully created: {}".format(final_path))
                return True
            else:
                self.logger.error("Output file not found: {}".format(output_bag_path))
                return False

        except Exception as e:
            self.logger.error("Error processing {}: {}".format(input_bag_path, e))
            return False

    def run(self):
        """Run the automation for all rosbag files"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Start roscore
            self.start_roscore()

            # Set simulation time
            self.set_sim_time()

            # Launch node
            self.launch_node()

            # Get list of rosbag files
            bag_pattern = os.path.join(self.input_folder, "*.bag")
            bag_files = glob.glob(bag_pattern)

            if not bag_files:
                self.logger.warning("No .bag files found in {}".format(self.input_folder))
                return

            self.logger.info("Found {} rosbag files to process".format(len(bag_files)))

            # Process each rosbag file
            successful = 0
            failed = 0

            for bag_file in bag_files:
                if self.process_single_rosbag(bag_file):
                    successful += 1
                else:
                    failed += 1

            self.logger.info("Processing complete. Successful: {}, Failed: {}".format(successful, failed))

        except Exception as e:
            self.logger.error("Automation failed: {}".format(e))
            raise
        finally:
            self.cleanup_processes()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Automate rosbag processing with ROS nodes')
    parser.add_argument('--input_folder', required=True, help='Path to input rosbag folder')
    parser.add_argument('--output_folder', required=True, help='Path to output rosbag folder')
    parser.add_argument('--catkin_ws', required=True, help='Path to catkin workspace')
    parser.add_argument('--launch_package', default='vizzy', help='ROS package name')
    parser.add_argument('--launch_file', default='human_detection.launch', help='Launch file name')
    parser.add_argument('--startup_delay', type=float, default=3.0, help='Node startup delay in seconds')
    parser.add_argument('--recording_delay', type=float, default=2.0, help='Recording startup delay in seconds')

    args = parser.parse_args()

    # Create and run automation
    automation = RosbagAutomation(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        catkin_ws=args.catkin_ws,
        launch_package=args.launch_package,
        launch_file=args.launch_file,
        startup_delay=args.startup_delay,
        recording_delay=args.recording_delay
    )

    automation.run()


if __name__ == "__main__":
    main()
