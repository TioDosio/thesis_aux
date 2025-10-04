import os
import subprocess
import json
import re

def parse_rosbag_info(info_str):
    # Parse the output of 'rosbag info' into a dictionary
    result = {}
    lines = info_str.splitlines()
    topics = []
    types = []
    section = None
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        # Top-level key: value pairs
        m = re.match(r'^(\w+):\s+(.*)$', line)
        if m and not line.startswith(' '):
            key, value = m.group(1), m.group(2)
            section = key.lower()
            if section == 'topics':
                result['topics'] = []
            elif section == 'types':
                result['types'] = []
            else:
                result[section] = value
            continue
        # Parse topics
        if section == 'topics' and line.startswith(' '):
            # Example: /bt_log 9 msgs : std_msgs/String
            topic_match = re.match(r'^\s*(\S+)\s+(\d+) msgs?\s+:\s+([\w/\[\]]+)(\s+\((.*)\))?$', line)
            if topic_match:
                topic_name = topic_match.group(1)
                msg_count = int(topic_match.group(2))
                msg_type = topic_match.group(3)
                extra = topic_match.group(5) if topic_match.group(5) else None
                result['topics'].append({
                    'name': topic_name,
                    'messages': msg_count,
                    'type': msg_type,
                    'extra': extra
                })
            continue
        # Parse types
        if section == 'types' and line.startswith(' '):
            # Example: sensor_msgs/CameraInfo [c9a58c1b0b154e0e6da7578cb991d214]
            type_match = re.match(r'^\s*([\w/]+)\s+\[([a-f0-9]+)\]$', line)
            if type_match:
                result['types'].append({
                    'type': type_match.group(1),
                    'md5': type_match.group(2)
                })
            continue
    return result

def generate_info_json(folder_path, output_json):
    info_dict = {}
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.bag'):
            bag_path = os.path.join(folder_path, fname)
            try:
                # Run 'rosbag info' and capture output (Python 2.7 compatible)
                p = subprocess.Popen(['rosbag', 'info', bag_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = p.communicate()
                info = ''
                if out:
                    info += out
                if err:
                    info += "[stderr] " + err
                parsed = parse_rosbag_info(info)
                info_dict[fname] = parsed
            except Exception as e:
                info_dict[fname] = {"error": "Error getting info: {}".format(e)}
    with open(output_json, 'w') as out_f:
        json.dump(info_dict, out_f, indent=2)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder in ["bad", "good"]:
        folder_path = os.path.join(base_dir, folder)
        output_json = os.path.join(base_dir, "{}_rosbag_info.json".format(folder))
        print("Processing {} -> {}".format(folder_path, output_json))
        generate_info_json(folder_path, output_json)
