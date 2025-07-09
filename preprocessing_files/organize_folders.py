import os
import shutil
from collections import defaultdict

def organize_json_pairs(base_path):
    """
    Organize pairs of JSON files into folders based on their common name prefix.
    Files ending with '_ego_coordinates.json' and '_local_coordinates.json' that share the same prefix
    will be moved into a folder named after that prefix.
    """
    
    # Process both wild and invited folders
    for folder_name in ['wild', 'invited']:
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print("Folder {0} does not exist, skipping...".format(folder_path))
            continue
            
        print("\nProcessing folder: {0}".format(folder_name))
        
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        # Group files by their common prefix
        file_groups = defaultdict(list)
        
        for file in json_files:
            # Remove the specific suffixes to get the common prefix
            if file.endswith('_ego_coordinates.json'):
                prefix = file[:-len('_ego_coordinates.json')]
                file_groups[prefix].append(file)
            elif file.endswith('_local_coordinates.json'):
                prefix = file[:-len('_local_coordinates.json')]
                file_groups[prefix].append(file)
        
        # Create folders and move files
        for prefix, files in file_groups.items():
            if len(files) == 2:  # Only process if we have a complete pair
                # Create the new folder
                new_folder_path = os.path.join(folder_path, prefix)
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                
                print("  Created folder: {0}".format(prefix))
                
                # Move both files to the new folder
                for file in files:
                    src_path = os.path.join(folder_path, file)
                    dst_path = os.path.join(new_folder_path, file)
                    shutil.move(src_path, dst_path)
                    print("    Moved: {0}".format(file))
            else:
                print("  Warning: Found {0} files for prefix '{1}' (expected 2): {2}".format(len(files), prefix, files))

if __name__ == "__main__":
    # Get the current directory (preprocessing_files)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting to organize JSON file pairs...")
    organize_json_pairs(base_path)
    print("\nOrganization complete!")