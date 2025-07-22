# Real-Time Trajectory Prediction System

## Overview

This system provides real-time human trajectory prediction using the MonoTransmotion neural network model. It processes ROS topics containing human pose detections and ego-vehicle transforms to predict future human trajectories and visualize them in RViz.

## Architecture

The system consists of two main components:

1. **RealTimeTrajectoryPredictor** - Main prediction engine and ROS interface
2. **RealTimeDataProcessor** - Data processing and model input preparation

## Key Features

- Real-time trajectory prediction from ROS topics
- Support for rosbag playback with timestamp synchronization
- Configurable coordinate transformation methods
- RViz visualization with multiple marker types
- Trajectory enhancement for low-frequency data
- Person filtering (currently configured for person 0 only)

## System Requirements

### Dependencies

- ROS (Robot Operating System)
- PyTorch
- Python packages: numpy, yaml, argparse, json, traceback
- ROS message types: geometry_msgs, visualization_msgs, sensor_msgs, tf2_msgs
- Custom message: human_awareness_msgs

### Model Requirements

- MonoTransmotion model checkpoint: `../checkpoints/traj_pred/best_traj_model.pth`
- Configuration files: `configs/traj_pred.yaml`, `configs/localization.yaml`
- Model expects specific parameters:
  - `seq_len`: 10 frames
  - `obs_len`: 4 observation frames
  - `pred_len`: 2 prediction frames

## Configuration Parameters

### Core Parameters

```python
self.seq_len = 10       # Total sequence length (FIXED: must match model)
self.interval = 15      # Frame sampling interval
self.obs_len = 4        # Observation frames (FIXED: must match model)
self.pred_len = 2       # Prediction frames (FIXED: must match model)
```

### Coordinate Transformation Methods

- `'full'` - Apply complete ego pose and coordinate frame transformations
- `'simple'` - Use raw model output with minimal processing
- `'scaled'` - Apply scale corrections only

### Enhancement Options

- `enable_frame_interpolation` - Smooth predictions at low frequency
- `enable_trajectory_smoothing` - Temporal filtering using prediction history

## ROS Interface

### Subscribed Topics

- `/tf` - Transform messages (map→odom, odom→base_footprint)
- `/image_detections` - PersonsList messages with human pose detections

### Published Topics

- `/predicted_trajectories` - PoseArray format
- `/predicted_trajectories_json` - JSON string format
- `/trajectory_visualization` - MarkerArray for RViz
- `/predicted_paths` - Path format for each person

## Function Documentation

### Core Functions

#### `__init__(self)`

**Purpose**: Initialize the trajectory predictor with configuration parameters, model setup, and ROS publishers/subscribers.

**Key Operations**:

- Configure timing parameters (seq_len, interval, obs_len, pred_len)
- Initialize data processor for handling incoming ROS data
- Setup neural network model and load checkpoints
- Create ROS publishers for trajectory output and visualization
- Setup subscribers for transform and detection data

#### `setup_model(self)`

**Purpose**: Load and configure the MonoTransmotion neural network model for trajectory prediction.

**Key Operations**:

- Load model configuration from YAML files
- Initialize evaluator with trajectory and localization models
- Set model to evaluation mode for inference
- Handle error cases when model files are missing

#### `tf_callback(self, msg)`

**Purpose**: Process transform messages to extract robot pose information needed for coordinate transformations.

**Key Operations**:

- Extract map→odom and odom→base_footprint transforms
- Combine transforms to compute map→base_footprint transformation
- Store latest transform data with timestamps
- Convert to format expected by data processor

#### `combine_transforms(self, map_odom_transform, odom_base_transform)`

**Purpose**: Mathematically combine two coordinate transforms to create a single map→base_footprint transform.

**Key Operations**:

- Extract translation and rotation components from both transforms
- Apply quaternion multiplication for rotation combination
- Apply translation composition: t_combined = t1 + q1.rotate(t2)
- Return combined transform in standard format

#### `image_detections_callback(self, msg)`

**Purpose**: Process incoming human pose detections and trigger trajectory prediction when sufficient data is available.

**Key Operations**:

- Extract person data from PersonsList messages
- Convert keypoints to required format
- Add person detections to data processor with timestamps
- Trigger trajectory prediction attempt for all active persons

#### `extract_keypoints_from_person(self, person)`

**Purpose**: Convert keypoints from ROS message format to the standardized format required by the model.

**Key Operations**:

- Map keypoint names from message format to model expectations
- Ensure correct keypoint ordering (17 keypoints in specific sequence)
- Handle missing keypoints with zero confidence values
- Return keypoints in [x, y, confidence] format

#### `estimate_position_from_keypoints(self, keypoints)`

**Purpose**: Estimate 3D world position from 2D keypoint detections using camera intrinsics.

**Key Operations**:

- Filter keypoints by confidence threshold
- Calculate center point from valid keypoints
- Estimate depth using bounding box height and camera parameters
- Transform from image coordinates to world coordinates
- Return position in robot coordinate frame

#### `try_predict_trajectories(self, message_timestamp=None)`

**Purpose**: Main prediction orchestrator that attempts trajectory prediction for all active persons.

**Key Operations**:

- Check if sufficient data is available for prediction
- Generate model input for each active person
- Run trajectory prediction using neural network
- Apply trajectory enhancements for low-frequency data
- Publish predictions in multiple formats

#### `predict_single_trajectory(self, model_input)`

**Purpose**: Execute neural network inference to predict a single person's trajectory.

**Key Operations**:

- Prepare keypoint data in format expected by model (batch, seq, 17, 2)
- Run localization model to get 3D positions
- Convert localization output to trajectory prediction format
- Run trajectory prediction model for future positions
- Apply coordinate transformations based on selected method

#### `transform_model_output_to_robot_frame(self, pred_array, model_input)`

**Purpose**: Transform raw model predictions from relative coordinates to global robot coordinate frame.

**Key Operations**:

- Extract ego pose information for coordinate transformation
- Apply scale and coordinate corrections to raw predictions
- Transform relative displacements to global coordinates using ego pose
- Use quaternion rotation to properly orient relative movements
- Return trajectory points in map coordinate frame

#### `publish_predictions(self, predictions, timestamp)`

**Purpose**: Publish trajectory predictions in multiple ROS message formats for different consumers.

**Key Operations**:

- Create PoseArray messages for programmatic access
- Generate JSON string format for debugging and logging
- Create Path messages for navigation planning
- Trigger visualization marker generation
- Publish to all relevant topics with proper timestamps

#### `publish_trajectory_visualization(self, predictions, timestamp)`

**Purpose**: Create comprehensive RViz visualization markers for predicted trajectories.

**Key Operations**:

- **Filter for Person 0**: Only visualize trajectories for person with ID 0
- Create line strip markers for trajectory paths
- Add sphere markers for individual prediction points
- Generate text labels for person identification
- Create ground plane projections for spatial reference
- Apply color gradients to show temporal progression

### Data Processing Functions

#### `get_raw_keypoints_list(self, keypoints)`

**Purpose**: Convert keypoints from dictionary format to flat list format required by model.

#### `compute_bbox_from_keypoints(self, keypoints)`

**Purpose**: Calculate 2D bounding box from keypoint positions for depth estimation.

#### `apply_scale_and_coordinate_corrections(self, x_rel, y_rel)`

**Purpose**: Apply scale factors and coordinate frame corrections to model output.

### Enhancement Functions

#### `enhance_trajectory_for_low_frequency(self, person_id, trajectory)`

**Purpose**: Apply trajectory enhancements to improve prediction quality for low-frequency input data.

#### `interpolate_trajectory(self, trajectory, factor=2)`

**Purpose**: Insert interpolated points between predicted trajectory points to increase effective frequency.

#### `smooth_trajectory_with_history(self, person_id, trajectory)`

**Purpose**: Apply temporal smoothing using weighted average of recent trajectory predictions.

### Utility Functions

#### `calculate_optimal_interval(self, your_rosbag_fps)`

**Purpose**: Calculate recommended sampling interval based on rosbag frequency to match training data timing.

#### `get_current_reference_frame(self)`

**Purpose**: Retrieve the current combined map→base_footprint transform for coordinate conversions.

## Usage

### Running the System

```bash
# Start the trajectory predictor
rosrun real_time_simulation real_time_trajectory_predictor.py
```

### With Rosbag Playback

```bash
# Play rosbag with clock
rosbag play your_bag.bag --clock

# In another terminal, start the predictor
rosrun real_time_simulation real_time_trajectory_predictor.py
```

### RViz Visualization

1. Start RViz: `rosrun rviz rviz`
2. Add MarkerArray display
3. Set topic to `/trajectory_visualization`
4. Set fixed frame to `map`

## Data Flow

1. **Input Processing**:

   - Transform data from `/tf` topic
   - Human detections from `/image_detections` topic

2. **Data Accumulation**:

   - Store sequences of poses and detections
   - Maintain temporal synchronization

3. **Model Input Preparation**:

   - Sample frames at configured intervals
   - Format data for neural network input

4. **Prediction**:

   - Run localization model for 3D positions
   - Run trajectory model for future predictions

5. **Output Processing**:

   - Transform predictions to robot coordinate frame
   - Apply enhancements and smoothing

6. **Visualization**:
   - Generate RViz markers
   - Publish in multiple message formats

## Coordinate Frames

- **Input**: Camera image coordinates (pixels)
- **Processing**: Model-specific coordinate frame
- **Output**: ROS map coordinate frame (ENU - East-North-Up)

## Timing Configuration

The system is optimized for rosbag data at 7.5Hz with an interval of 15:

- **Time window**: ~20 seconds (150 frames total)
- **Observation period**: 4 frames (last 8 seconds)
- **Prediction horizon**: 2 frames (next 4 seconds)

## Known Limitations

1. Currently configured to visualize only person 0
2. Requires specific model checkpoint files
3. Coordinate transformation may need tuning for different setups
4. Performance depends on keypoint detection quality

## Troubleshooting

### Common Issues

1. **Model not loading**: Check checkpoint paths and YAML configurations
2. **No predictions**: Verify sufficient observation frames are available
3. **Coordinate misalignment**: Adjust transformation method or scale factors
4. **Visualization not appearing**: Check RViz topic subscription and frame settings

### Debug Options

- Set `debug_coordinates = True` for coordinate transformation details
- Use different `coordinate_transform_method` values to test transformations
- Enable trajectory smoothing for noisy predictions

## File Structure

```
real_time_simulation/
├── real_time_trajectory_predictor.py    # Main prediction system
├── real_time_data_processor.py          # Data processing utilities
├── rviz_config.rviz                     # RViz configuration
└── README.md                            # This documentation
```

## Model Information

The system uses the MonoTransmotion model architecture:

- **Input**: Sequence of 2D keypoints and ego poses
- **Output**: 2D trajectory points in relative coordinates
- **Architecture**: Transformer-based sequence-to-sequence model
- **Training**: Uses temporal sequences with specific interval sampling

For more details on the model architecture and training process, refer to the MonoTransmotion paper and model documentation.
