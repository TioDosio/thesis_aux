# Complete Pipeline Script

## Overview

`complete_pipeline.py` is a comprehensive, self-contained script that processes ROS bag files through the entire pipeline without calling external scripts.

## What It Does

This single script handles everything:

1. **Finds all `.bag` files** in:

   - `rosbag_automation/output/invited/`
   - `rosbag_automation/output/wild/`

2. **Processes each bag file** to create:

   - `*_ego_coordinates.json` (from `/raw_bodies` and `/tf` topics)
   - `*_local_coordinates.json` (from `/image_detections` and `/raw_bodies` topics)

3. **Saves output files** to:

   - `preprocessing_files/invited/`
   - `preprocessing_files/wild/`

4. **Organizes files** into paired folders:
   - Each folder contains both ego and local coordinate files for the same recording

## Key Features

✅ **Self-contained** - No external script calls  
✅ **Detailed logging** - Timestamps and progress tracking  
✅ **Error handling** - Continues processing even if some files fail  
✅ **Automatic directory creation** - Creates needed folders  
✅ **Progress reporting** - Shows success/failure counts

## Usage

Simply run:

```bash
python complete_pipeline.py
```

## Prerequisites

- **Python 3.x**
- **ROS environment** with rosbag support
- **Required ROS packages**:
  - `rosbag`
  - `tf2_msgs`
  - `geometry_msgs`

## Input Structure Expected

```
rosbag_automation/output/
├── invited/
│   ├── 2022-04-27-16-03-00_greeting_exp_Heenan_new.bag
│   ├── 2022-04-29-11-42-05_greeting_exp_Heenan_new.bag
│   └── ... (more .bag files)
└── wild/
    ├── 2022-04-28-17-29-58_greeting_exp_Heenan_new.bag
    └── ... (more .bag files)
```

## Output Structure Generated

```
preprocessing_files/
├── invited/
│   ├── 2022-04-27-16-03-00_greeting_exp_Heenan_new/
│   │   ├── 2022-04-27-16-03-00_greeting_exp_Heenan_new_ego_coordinates.json
│   │   └── 2022-04-27-16-03-00_greeting_exp_Heenan_new_local_coordinates.json
│   └── 2022-04-29-11-42-05_greeting_exp_Heenan_new/
│       ├── 2022-04-29-11-42-05_greeting_exp_Heenan_new_ego_coordinates.json
│       └── 2022-04-29-11-42-05_greeting_exp_Heenan_new_local_coordinates.json
└── wild/
    └── ... (similar structure)
```

## What Each Format Contains

### Ego Coordinates Format

- Robot's position and orientation in the world
- Uses `/tf` topic for odometry transforms
- Uses `/raw_bodies` topic for position data
- Format: `{"frame": N, "coordinates": {"x": X, "y": Y, "z": Z, "q1": Q1, ...}}`

### Local Coordinates Format

- Human pose detection and tracking data
- Uses `/image_detections` topic for keypoints
- Uses `/raw_bodies` topic for 3D positions
- Format: `{"frame": N, "coordinates": [{"id": ID, "x": X, "y": Y, "z": Z, "bbox": [...], "keypoints": [...]}]}`

## Sample Output Log

```
[2025-07-14 10:30:15] INFO: COMPLETE PIPELINE PROCESSING - STARTING
[2025-07-14 10:30:15] INFO: Processing INVITED folder
[2025-07-14 10:30:15] INFO: Found 25 .bag files in invited
[2025-07-14 10:30:16] INFO: [1/25] Processing: 2022-04-27-16-03-00_greeting_exp_Heenan_new
[2025-07-14 10:30:45] INFO: ✓ Successfully processed: 2022-04-27-16-03-00_greeting_exp_Heenan_new
...
[2025-07-14 11:15:20] INFO: Organizing JSON files into paired folders
[2025-07-14 11:15:21] INFO: PIPELINE PROCESSING COMPLETE
[2025-07-14 11:15:21] INFO: Total successful: 47
[2025-07-14 11:15:21] INFO: Total failed: 3
```

## Error Handling

The script will:

- Continue processing even if individual files fail
- Show detailed error messages for debugging
- Provide a final summary of successes and failures
- Skip folders that don't exist
- Handle missing ROS dependencies gracefully

## Troubleshooting

1. **"rosbag module not available"**: Install ROS or rosbag package
2. **"No .bag files found"**: Check that input directories exist and contain .bag files
3. **"Topic not found in bag"**: Verify bag files contain required topics (`/tf`, `/raw_bodies`, `/image_detections`)
4. **Permission errors**: Ensure write access to output directories

This script replaces the need for multiple separate scripts and provides a complete end-to-end solution.
