# this code reads a JSON file containing keypoints data, 
# sorts the keypoints by the number of non-zero values, 
# and writes the top 5 keypoints to a text file.

import json

input_filename = 'epfl_local_coordinates.json'  # Your input file
output_filename = 'top_5_keypoints.txt'

def count_non_zero(arr):
    return sum(1 for v in arr if v != 0)

with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError as e:  # Works for both Python 2 and 3
            print("Skipping line due to JSON error:", e)
            continue
        keypoints_arrays = []
        for obj in data.get('coordinates', []):
            keypoints = obj.get('keypoints', [])
            keypoints_arrays.append(keypoints)
        # Sort by number of nonzero values, descending
        sorted_keypoints = sorted(keypoints_arrays, key=count_non_zero, reverse=True)
        # Take top 5
        for kp in sorted_keypoints[:5]:
            outfile.write(str(kp) + '\n')
