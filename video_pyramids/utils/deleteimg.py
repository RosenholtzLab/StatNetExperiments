import os
import glob

# Specify the pattern to search for
# ** means to look in all subdirectories, * means any file
def remove_duplicates_from_file(filepath):
    # Read the lines from the file
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Use a set to keep track of seen lines and a list to keep them in order
    seen = set()
    unique_lines = [line for line in lines if line not in seen and not seen.add(line)]
    
    # Write the unique lines back to the file
    with open(filepath, 'w') as file:
        file.writelines(unique_lines)

# Use the function on your file
remove_duplicates_from_file('/home/gridsan/ckoevesdi/data/OT/dtd_torch/dtd/dtd/labels/train10.txt')
