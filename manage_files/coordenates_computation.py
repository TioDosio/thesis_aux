# this code parses a ROS tf file and computes the absolute 
# poses of frames in a robot's coordinate system, specifically 
# extracting the camera pose in the 'map' frame.


import re
import numpy as np
from collections import deque

# Quaternion operations
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return (w, x, y, z)

def quaternion_rotate_vector(q, v):
    v_quat = (0, v[0], v[1], v[2])
    q_conj = (q[0], -q[1], -q[2], -q[3])
    temp = quaternion_multiply(q, v_quat)
    result = quaternion_multiply(temp, q_conj)
    return (result[1], result[2], result[3])

# Parse TF file
def parse_tf_file(filename):
    connections = {}
    transformations = {}
    all_links = set()
    
    with open(filename, 'r') as f:
        content = f.read()
    
    pattern = r'frame_id: "(\w+)"\s+child_frame_id: "(\w+)"[^t]*translation:\s+x: ([\d\.\-e]+)\s+y: ([\d\.\-e]+)\s+z: ([\d\.\-e]+)\s+rotation:\s+x: ([\d\.\-e]+)\s+y: ([\d\.\-e]+)\s+z: ([\d\.\-e]+)\s+w: ([\d\.\-e]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        parent, child, tx, ty, tz, rx, ry, rz, rw = match
        translation = (float(tx), float(ty), float(tz))
        rotation = (float(rw), float(rx), float(ry), float(rz))
        
        # Build connections graph
        if parent not in connections:
            connections[parent] = []
        connections[parent].append(child)
        all_links.update([parent, child])
        
        # Store transformation (use first occurrence)
        if (parent, child) not in transformations:
            transformations[(parent, child)] = (translation, rotation)
    
    # Ensure all links appear as keys
    for link in all_links:
        if link not in connections:
            connections[link] = []
    
    return connections, transformations

# Compute absolute poses starting from root frame
def compute_absolute_poses(connections, transformations, root_frame='map'):
    poses = {}
    queue = deque([root_frame])
    
    # Initialize root frame
    poses[root_frame] = {
        'position': (0.0, 0.0, 0.0),
        'quaternion': (1.0, 0.0, 0.0, 0.0)  # Identity
    }
    
    while queue:
        parent = queue.popleft()
        if parent not in connections:
            continue
            
        for child in connections[parent]:
            # Skip if transformation not found
            if (parent, child) not in transformations:
                print(f"Missing transformation: {parent} -> {child}")
                queue.append(child)
                continue
                
            # Get relative transform
            t_rel, q_rel = transformations[(parent, child)]
            parent_pose = poses[parent]
            
            # Compute absolute orientation
            q_abs = quaternion_multiply(parent_pose['quaternion'], q_rel)
            
            # Compute absolute position
            rotated_t = quaternion_rotate_vector(parent_pose['quaternion'], t_rel)
            t_abs = (
                parent_pose['position'][0] + rotated_t[0],
                parent_pose['position'][1] + rotated_t[1],
                parent_pose['position'][2] + rotated_t[2]
            )
            
            # Store child pose
            poses[child] = {
                'position': t_abs,
                'quaternion': q_abs
            }
            queue.append(child)
    
    return poses

# Main execution
if __name__ == "__main__":
    # Process TF data
    connections, transformations = parse_tf_file('tf.txt')
    
    # Compute absolute poses from map frame
    absolute_poses = compute_absolute_poses(connections, transformations, root_frame='map')
    
    # Get camera pose (eyes_tilt_link)
    camera_frame = 'eyes_tilt_link'
    if camera_frame in absolute_poses:
        cam_pose = absolute_poses[camera_frame]
        print(f"Camera position in map frame (x, y, z):")
        print(f"  {cam_pose['position'][0]:.6f}, {cam_pose['position'][1]:.6f}, {cam_pose['position'][2]:.6f}")
        print("\nCamera orientation (quaternion w, x, y, z):")
        print(f"  {cam_pose['quaternion'][0]:.6f}, {cam_pose['quaternion'][1]:.6f}, "
              f"{cam_pose['quaternion'][2]:.6f}, {cam_pose['quaternion'][3]:.6f}")
    else:
        print(f"Camera frame '{camera_frame}' not found in computed poses")
        print("Available frames:", list(absolute_poses.keys()))
