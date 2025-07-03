import yaml
import json
from collections import OrderedDict

# Keypoint order as specified
CORRECT_ORDER = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
]

def parse_yaml_file(filename):
    """Yield each non-empty YAML document."""
    with open(filename, 'r') as f:
        for doc in yaml.safe_load_all(f):
            if doc is not None:
                yield doc

def extract_keypoints(person):
    """
    Return keypoints in the CORRECT_ORDER.
    Missing keypoints are [0.0, 0.0, 0.0].
    """
    part_map = {bp['part_id']: [float(bp['x']), float(bp['y']), float(bp['confidence'])]
                for bp in person.get('body_parts', [])}
    keypoints = []
    for part in CORRECT_ORDER:
        keypoints.extend(part_map.get(part, [0.0, 0.0, 0.0]))
    return keypoints

def compute_bbox(keypoints):
    """
    Compute bounding box [x_min, y_min, x_max, y_max] from valid keypoints.
    Only considers keypoints with confidence > 0.
    """
    xs, ys = [], []
    for i in range(0, len(keypoints), 3):
        x, y, c = keypoints[i], keypoints[i+1], keypoints[i+2]
        if c > 0:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max, y_max]

def main():
    input_file = 'image_detections.txt'
    output_file = 'keypoints_output.json'

    with open(output_file, 'w') as out_f:
        for doc in parse_yaml_file(input_file):
            persons = doc.get('persons', [])
            for person in persons:
                keypoints = extract_keypoints(person)
                bbox = compute_bbox(keypoints)
                # Output as requested
                output = OrderedDict([
                ("id", 1),
                ("x", 0.0),
                ("y", 0.0),
                ("z", 0.0),
                ("bbox", bbox),
                ("keypoints", keypoints)
                ])
                json_string = json.dumps(output, sort_keys=False)
                out_f.write(json_string + '\n')

if __name__ == '__main__':
    main()
