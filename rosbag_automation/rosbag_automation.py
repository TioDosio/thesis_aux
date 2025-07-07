#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS 1 Melodic Rosbag Automation Script
=====================================

This script automates the process of:
1. Starting a ROS node (human_detection.launch)
2. Recording a new rosbag with all topics
3. Playing back the original rosbag with sync
4. Processing multiple rosbags in batch

Author: Auto-generated for ROS 1 Melodic
"""

import subprocess
import os
import time
import signal
import glob
import sys
import argparse
import logging
from datetime import datetime

class RosbagAutomation:
    """Main automation class for processing rosbags with additional ROS nodes"""

    def __init__(self, input_folder, output_folder, catkin_ws_path):
        self.input_folder = os.path.abspath(input_folder)
        self.output_folder = os.path.abspath(output_folder)
        self.catkin_ws_path = os.path.abspath(catkin_ws_path)
        self.launch_path = os.path.join(self.catkin_ws_path, "src", "vizzy", "human_detection_3d", "launch")

        # Process tracking
        self.processes = {}
        self.active_processes = []

        # Setup logging
        self.setup_logging()

        # Create output directory
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Validate paths
        self.validate_setup()

    def setup_logging(self):
        """Setup logging for the automation process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('rosbag_automation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_setup(self):
        """Validate that all required paths and files exist"""
        if not os.path.exists(self.input_folder):
            raise Exception("Input folder does not exist: {}".format(self.input_folder))

        if not os.path.exists(self.catkin_ws_path):
            raise Exception("Catkin workspace does not exist: {}".format(self.catkin_ws_path))

        if not os.path.exists(self.launch_path):
            raise Exception("Launch file path does not exist: {}".format(self.launch_path))

        launch_file = os.path.join(self.launch_path, "human_detection.launch")
        if not os.path.exists(launch_file):
            raise Exception("Launch file does not exist: {}".format(launch_file))

        self.logger.info("All paths validated successfully")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received signal {}, shutting down gracefully...".format(signum))
        self.cleanup_all_processes()
        sys.exit(0)

    def cleanup_all_processes(self):
        """Clean up all running processes"""
        self.logger.info("Cleaning up all processes...")

        for name, process in self.processes.items():
            if process and process.poll() is None:
                try:
                    self.logger.info("Terminating {} (PID: {})".format(name, process.pid))
                    # Send SIGINT first for graceful shutdown
                    process.send_signal(signal.SIGINT)
                    # Wait for process with manual timeout handling
                    for _ in range(50):  # 5 seconds with 0.1s intervals
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    else:
                        # If SIGINT doesn't work, use SIGTERM
                        self.logger.warning("SIGINT timeout for {}, sending SIGTERM".format(name))
                        process.terminate()
                        for _ in range(30):  # 3 seconds with 0.1s intervals
                            if process.poll() is not None:
                                break
                            time.sleep(0.1)
                        else:
                            # Last resort: SIGKILL
                            self.logger.warning("SIGTERM timeout for {}, sending SIGKILL".format(name))
                            process.kill()
                            process.wait()
                except Exception as e:
                    self.logger.error("Error terminating {}: {}".format(name, e))

        self.processes.clear()
        self.logger.info("All processes cleaned up")

    def run_command(self, cmd, name, cwd=None, env=None, shell=True):
        """Run a command and track the process"""
        try:
            # Set up environment
            if env is None:
                env = os.environ.copy()

            self.logger.info("Starting {}: {}".format(name, cmd))

            process = subprocess.Popen(
                cmd,
                shell=shell,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            self.processes[name] = process
            return process

        except Exception as e:
            self.logger.error("Failed to start {}: {}".format(name, e))
            return None

    def wait_for_topics(self, timeout=30):
        """Wait for ROS topics to be available"""
        self.logger.info("Waiting for ROS topics to be available...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                process = subprocess.Popen(
                    ["rostopic", "list"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()

                if process.returncode == 0 and stdout.strip():
                    topics = stdout.strip().split('\n')
                    if len(topics) > 1:  # More than just /rosout
                        self.logger.info("Found {} topics".format(len(topics)))
                        return True

            except Exception as e:
                self.logger.debug("Waiting for topics: {}".format(e))

            time.sleep(1)

        self.logger.warning("Timeout waiting for topics")
        return False

    def start_roscore(self):
        """Start roscore if not already running"""
        try:
            # Check if roscore is already running
            process = subprocess.Popen(
                ["rostopic", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.logger.info("ROS master already running")
                return True

        except:
            pass

        self.logger.info("Starting roscore...")
        process = self.run_command("roscore", "roscore")

        if process:
            # Wait for roscore to be ready
            time.sleep(3)
            return True

        return False

    def start_human_detection_node(self):
        """Start the human detection launch file"""
        self.logger.info("Starting human detection node...")

        # Change to launch directory and run roslaunch
        cmd = "cd {} && roslaunch human_detection.launch".format(self.launch_path)
        process = self.run_command(cmd, "human_detection", cwd=self.launch_path)

        if process:
            # Wait for node to initialize
            self.logger.info("Waiting for human detection node to initialize...")
            time.sleep(3)  # User specified 2-3 seconds
            return True

        return False

    def get_all_topics(self):
        """Get list of all currently published topics"""
        try:
            process = subprocess.Popen(
                ["rostopic", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                topics = [t.strip() for t in stdout.strip().split('\n') if t.strip()]
                self.logger.info("Found {} topics: {}".format(len(topics), topics))
                return topics

        except Exception as e:
            self.logger.error("Failed to get topics: {}".format(e))

        return []

    def process_single_rosbag(self, input_bag_path):
        """Process a single rosbag file"""
        input_bag = os.path.abspath(input_bag_path)

        if not os.path.exists(input_bag):
            self.logger.error("Input bag does not exist: {}".format(input_bag))
            return False

        # Create output filename with "new" suffix
        input_bag_basename = os.path.basename(input_bag)
        output_bag_name = os.path.splitext(input_bag_basename)[0] + "_new.bag"
        output_bag_path = os.path.join(self.output_folder, output_bag_name)

        self.logger.info("Processing: {} -> {}".format(input_bag, output_bag_path))

        try:
            # Step 1: Set simulation time parameter
            self.logger.info("Setting use_sim_time parameter...")
            process = subprocess.Popen(
                ["rosparam", "set", "use_sim_time", "true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                self.logger.error("Failed to set use_sim_time parameter")
                return False

            # Step 2: Start recording new rosbag with all topics
            self.logger.info("Starting rosbag recording...")
            record_cmd = "rosbag record -a -O {}".format(output_bag_path)
            record_process = self.run_command(record_cmd, "rosbag_record")

            if not record_process:
                self.logger.error("Failed to start rosbag recording")
                return False

            # Wait a moment for recording to start
            time.sleep(2)

            # Step 3: Play original rosbag with clock
            self.logger.info("Playing original rosbag: {}".format(input_bag))
            play_cmd = "rosbag play {} --clock".format(input_bag)
            play_process = self.run_command(play_cmd, "rosbag_play")

            if not play_process:
                self.logger.error("Failed to start rosbag playback")
                self.cleanup_all_processes()
                return False

            # Step 4: Wait for playback to complete
            self.logger.info("Waiting for rosbag playback to complete...")
            try:
                play_process.wait()
                self.logger.info("Rosbag playback completed")
            except Exception as e:
                self.logger.error("Error during playback: {}".format(e))
                return False

            # Step 5: Stop recording
            self.logger.info("Stopping rosbag recording...")
            if record_process and record_process.poll() is None:
                record_process.send_signal(signal.SIGINT)
                # Wait for process with manual timeout handling
                for _ in range(100):  # 10 seconds with 0.1s intervals
                    if record_process.poll() is not None:
                        break
                    time.sleep(0.1)
                else:
                    record_process.terminate()
                    for _ in range(50):  # 5 seconds with 0.1s intervals
                        if record_process.poll() is not None:
                            break
                        time.sleep(0.1)
                    else:
                        record_process.kill()
                        record_process.wait()

            # Clean up processes
            if "rosbag_play" in self.processes:
                del self.processes["rosbag_play"]
            if "rosbag_record" in self.processes:
                del self.processes["rosbag_record"]

            # Verify output file was created
            if os.path.exists(output_bag_path):
                self.logger.info("Successfully created: {}".format(output_bag_path))
                return True
            else:
                self.logger.error("Output bag file was not created: {}".format(output_bag_path))
                return False

        except Exception as e:
            self.logger.error("Error processing {}: {}".format(input_bag, e))
            self.cleanup_all_processes()
            return False

    def process_all_rosbags(self):
        """Process all rosbag files in the input folder"""
        # Find all .bag files in input folder
        bag_pattern = os.path.join(self.input_folder, "*.bag")
        bag_files = glob.glob(bag_pattern)

        if not bag_files:
            self.logger.warning("No .bag files found in {}".format(self.input_folder))
            return

        self.logger.info("Found {} bag files to process".format(len(bag_files)))

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Start roscore
            if not self.start_roscore():
                self.logger.error("Failed to start roscore")
                return

            # Start human detection node
            if not self.start_human_detection_node():
                self.logger.error("Failed to start human detection node")
                return

            # Wait for topics to be available
            if not self.wait_for_topics():
                self.logger.error("Topics not available")
                return

            successful = 0
            failed = 0

            # Process each bag file
            for i, bag_file in enumerate(bag_files, 1):
                bag_name = os.path.basename(bag_file)
                self.logger.info("Processing bag {}/{}: {}".format(i, len(bag_files), bag_name))

                if self.process_single_rosbag(bag_file):
                    successful += 1
                else:
                    failed += 1

                # Small delay between processing
                if i < len(bag_files):
                    time.sleep(1)

            self.logger.info("Processing complete. Successful: {}, Failed: {}".format(successful, failed))

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error("Unexpected error during processing: {}".format(e))
        finally:
            self.cleanup_all_processes()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Automate ROS bag processing with human detection node"
    )

    parser.add_argument(
        "--input_folder",
        required=True,
        help="Path to folder containing input rosbag files"
    )

    parser.add_argument(
        "--output_folder",
        required=True,
        help="Path to folder for output rosbag files"
    )

    parser.add_argument(
        "--catkin_ws",
        default="~/catkin_ws",
        help="Path to catkin workspace (default: ~/catkin_ws)"
    )

    args = parser.parse_args()

    # Expand user paths
    input_folder = os.path.expanduser(args.input_folder)
    output_folder = os.path.expanduser(args.output_folder)
    catkin_ws = os.path.expanduser(args.catkin_ws)

    try:
        automation = RosbagAutomation(input_folder, output_folder, catkin_ws)
        automation.process_all_rosbags()
    except Exception as e:
        print("Error: {}".format(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
