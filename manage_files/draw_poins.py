# this code draws keypoints from a given array and labels 
# them with COCO keypoint names.

import matplotlib.pyplot as plt

# Keypoint names (COCO order)
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Your keypoints array
keypoints = [
    0.0, 0.0, 0.0, 152.42, 317.77, 0.52, 0.0, 0.0, 0.0, 161.18, 319.44, 0.68,
    0.0, 0.0, 0.0, 174.59, 336.68, 0.93, 151.02, 335.77, 0.81, 176.78, 355.41, 0.81,
    145.3, 355.94, 0.52, 157.93, 347.77, 0.73, 0.0, 0.0, 0.0, 166.13, 381.19, 0.83,
    148.84, 378.97, 0.77, 166.57, 410.32, 0.77, 145.25, 410.84, 0.73, 180.81, 431.65, 0.66,
    148.13, 438.93, 0.76
]

plt.figure(figsize=(8, 8))

for idx in range(0, len(keypoints), 3):
    x, y, conf = keypoints[idx], keypoints[idx+1], keypoints[idx+2]
    if x == 0 and y == 0 and conf == 0:
        continue
    kp_idx = idx // 3
    name = keypoint_names[kp_idx] if kp_idx < len(keypoint_names) else f"kp{kp_idx}"
    # Swap x and y
    plt.scatter(x, y, c='b')
    plt.text(x, y, f"{kp_idx} {name}", fontsize=9, ha='center', va='bottom')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Keypoints (x, y, confidence) with COCO Labels')
plt.gca().invert_yaxis()  # Optional: matches image coordinates
plt.grid(True)
plt.show()
