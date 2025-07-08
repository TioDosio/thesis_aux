import os
import sys
import subprocess
import time
import signal
import glob
import logging
from datetime import datetime


class RosbagAutomation:
    """Automates rosbag processing with ROS nodes"""

    def __init__(self, input_folder, output_folder, catkin_ws,
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
        self.roscore_process = None
        self.node_process = None
        self.record_process = None
        self.play_process = None

        # Setup logging first
        self.setup_logging()

        # Validate paths
        self.validate_paths()

        self.logger.info("RosbagAutomation initialized successfully")

    def get_processing_status(self):
        """Get the current processing status of files in input/output folders"""
        # Get all input bag files
        bag_pattern = os.path.join(self.input_folder, "*.bag")
        input_files = glob.glob(bag_pattern)
        input_files.sort()
        
        # Count processed files (those that exist in output folder)
        processed_count = 0
        for input_file in input_files:
            bag_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_folder, "{}_new.bag".format(bag_name))
            if os.path.exists(output_file):
                processed_count += 1
        
        return processed_count, len(input_files), input_files

    def display_processing_status(self):
        """Display current processing status"""
        processed, total, files = self.get_processing_status()
        
        if total == 0:
            self.logger.info("No .bag files found in input folder")
            return
        
        self.logger.info("=" * 50)
        self.logger.info("PROCESSING STATUS: {}/{} files completed ({:.1f}%)".format(
            processed, total, (processed * 100.0) / total
        ))
        self.logger.info("=" * 50)
        
        # Show detailed status for each file
        for i, input_file in enumerate(files, 1):
            bag_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_folder, "{}_new.bag".format(bag_name))
            
            if os.path.exists(output_file):
                status = "✓ DONE"
                # Get file size for additional info
                try:
                    size_mb = os.path.getsize(output_file) / (1024.0 * 1024.0)
                    status += " ({:.1f} MB)".format(size_mb)
                except:
                    pass
            else:
                status = "⏳ PENDING"
            
            self.logger.info("{:2d}. {} - {}".format(i, bag_name, status))
        
        self.logger.info("=" * 50)

    def setup_logging(self):
        """Configure logging for the automation"""
        # Create output folder if it doesn't exist (for log file)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        log_file = os.path.join(self.output_folder, 'rosbag_automation.log')
        
        # Configure logging with both file and console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_paths(self):
        """Validate that required paths exist"""
        # Check input folder
        if not os.path.exists(self.input_folder):
            raise RuntimeError("Input folder does not exist: {}".format(self.input_folder))

        # Check catkin workspace
        if not os.path.exists(self.catkin_ws):
            raise RuntimeError("Catkin workspace does not exist: {}".format(self.catkin_ws))

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

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
            raise RuntimeError("Launch file does not exist: {}".format(launch_path))

    def signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.logger.info("Received signal {}. Cleaning up...".format(signum))
        self.cleanup_processes()
        sys.exit(0)

    def _wait_for_process_termination(self, process, timeout=5):
        """Wait for process termination with timeout (Python 2.7 compatible)"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                return True
            time.sleep(0.1)
        return False

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
                    if not self._wait_for_process_termination(process, 5):
                        self.logger.warning("Force killing {}...".format(name))
                        process.kill()
                        self._wait_for_process_termination(process, 2)
                except Exception as e:
                    self.logger.error("Error terminating {}: {}".format(name, e))

    def start_roscore(self):
        """Start roscore if not already running"""
        # Check if roscore is already running
        try:
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
            # Python 2.7 compatible string handling
            if isinstance(result, bytes):
                result = result.decode('utf-8')
            topics = result.strip().split('\n')
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

            # Wait for graceful shutdown using helper method
            if self._wait_for_process_termination(self.record_process, 10):
                self.logger.info("Recording stopped successfully")
            else:
                self.logger.warning("Recording did not stop gracefully, force killing...")
                self.record_process.kill()
                self._wait_for_process_termination(self.record_process, 2)

    def process_single_rosbag(self, input_bag_path, current_index, total_files):
        """Process a single rosbag file with progress tracking"""
        bag_name = os.path.splitext(os.path.basename(input_bag_path))[0]
        output_bag_path = os.path.join(self.output_folder, "{}_new".format(bag_name))

        self.logger.info("=" * 60)
        self.logger.info("PROCESSING [{}/{}]: {}".format(current_index, total_files, bag_name))
        self.logger.info("Input:  {}".format(input_bag_path))
        self.logger.info("Output: {}.bag".format(output_bag_path))
        self.logger.info("=" * 60)

        try:
            # Start recording
            self.start_recording(output_bag_path)

            # Play the rosbag
            self.play_rosbag(input_bag_path)

            # Stop recording
            self.stop_recording()

            # Verify output file was created (rosbag record adds .bag extension)
            expected_output = output_bag_path + ".bag"
            if os.path.exists(expected_output):
                # Get file size for logging
                try:
                    size_mb = os.path.getsize(expected_output) / (1024.0 * 1024.0)
                    self.logger.info("✓ SUCCESS [{}/{}]: {} ({:.1f} MB)".format(
                        current_index, total_files, bag_name, size_mb
                    ))
                except:
                    self.logger.info("✓ SUCCESS [{}/{}]: {}".format(
                        current_index, total_files, bag_name
                    ))
                return True
            else:
                self.logger.error("✗ FAILED [{}/{}]: Output file not found - {}".format(
                    current_index, total_files, bag_name
                ))
                return False

        except Exception as e:
            self.logger.error("✗ FAILED [{}/{}]: {} - Error: {}".format(
                current_index, total_files, bag_name, e
            ))
            return False

    def run(self):
        """Run the automation for all rosbag files"""
        self.logger.info("Starting rosbag automation...")
        
        # Display initial processing status
        self.display_processing_status()
        
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
            
            # Sort files for consistent processing order
            bag_files.sort()

            if not bag_files:
                self.logger.warning("No .bag files found in {}".format(self.input_folder))
                return

            self.logger.info("Found {} rosbag files to process".format(len(bag_files)))

            # Filter out already processed files (optional - comment out if you want to reprocess)
            unprocessed_files = []
            for bag_file in bag_files:
                bag_name = os.path.splitext(os.path.basename(bag_file))[0]
                output_file = os.path.join(self.output_folder, "{}_new.bag".format(bag_name))
                if not os.path.exists(output_file):
                    unprocessed_files.append(bag_file)
                else:
                    self.logger.info("Skipping already processed: {}".format(bag_name))

            if not unprocessed_files:
                self.logger.info("All files have been processed already!")
                return

            self.logger.info("Processing {} unprocessed files...".format(len(unprocessed_files)))

            # Process each unprocessed rosbag file
            successful = 0
            failed = 0
            total_files = len(bag_files)

            for i, bag_file in enumerate(unprocessed_files, 1):
                # Calculate current index in the full list
                current_index = bag_files.index(bag_file) + 1
                
                if self.process_single_rosbag(bag_file, current_index, total_files):
                    successful += 1
                else:
                    failed += 1

            self.logger.info("Processing complete. Successful: {}, Failed: {}".format(successful, failed))
            
            # Display final status
            self.display_processing_status()

        except Exception as e:
            self.logger.error("Automation failed: {}".format(e))
            raise
        finally:
            self.cleanup_processes()


def main():
    # Check if folder argument is provided
    if len(sys.argv) != 2:
        print("Usage: python rosbag_automation.py <folder>")
        print("<folder> can be 'wild' or 'invited'")
        sys.exit(1)
    
    flag_folder_input = sys.argv[1]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input", flag_folder_input)
    output_folder = os.path.join(current_dir, "output", flag_folder_input)
    catkin_ws = "~/catkin_ws"
    launch_package = "vizzy"
    launch_file = "human_detection.launch"
    startup_delay = 3.0
    recording_delay = 2.0

    # Create and run automation
    automation = RosbagAutomation(
        input_folder=input_folder,
        output_folder=output_folder,
        catkin_ws=catkin_ws,
        launch_package=launch_package,
        launch_file=launch_file,
        startup_delay=startup_delay,
        recording_delay=recording_delay
    )

    automation.run()


if __name__ == "__main__":
    main()
