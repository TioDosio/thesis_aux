# This code parses a ROS tf file and creates a dictionary of 
# parent-child relationships between frames.

from collections import defaultdict

def parse_tf_file(filename):
    connections = defaultdict(set)
    all_links = set()

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('frame_id:'):
                parent = line.split('"')[1]
                all_links.add(parent)
            if line.startswith('child_frame_id:'):
                child = line.split('"')[1]
                all_links.add(child)
                # Add connection
                connections[parent].add(child)

    # Ensure all links are keys, even those with no children
    for link in all_links:
        connections.setdefault(link, set())

    # Convert sets to sorted lists for readability (optional)
    connections = {k: sorted(list(v)) for k, v in connections.items()}
    return connections

# Usage
tf_dict = parse_tf_file('tf.txt')
for parent, children in tf_dict.items():
    print(f"{parent}: {children}")

