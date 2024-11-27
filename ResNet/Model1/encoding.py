import os

# Path to the main folder with subfolders
main_folder_path = '/Users/michallejza/Desktop/Data/IMAGENET/tiny-imagenet-200/train'

# Path to the words.txt file
words_file_path = '/Users/michallejza/Desktop/Data/IMAGENET/tiny-imagenet-200/words.txt'

# Output file path for the folder-to-description mapping
output_file_path = 'mapping.txt'

# Load the folder name-to-description mapping from words.txt
folder_descriptions = {}
with open(words_file_path, 'r') as f:
    for line in f:
        folder_name, description = line.strip().split('\t')  # Assuming tab-separated
        folder_descriptions[folder_name] = description

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over each folder in the main directory
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path) and folder_name in folder_descriptions:
            # Write the folder name and description to the output file
            output_file.write(f"{folder_name}\t{folder_descriptions[folder_name]}\n")
        else:
            # If there's no description, you can also log it or leave it out
            output_file.write(f"{folder_name}\tNo description found\n")

# Path to the mapping.txt file
mapping_file_path = 'mapping.txt'

# Open the file for reading
with open(mapping_file_path, 'r') as file:
    lines = file.readlines()

# Process each line to keep only the first name
with open(mapping_file_path, 'w') as file:
    for line in lines:
        folder_code, names = line.strip().split('\t', 1)  # Split at the first tab
        # Keep only the first name before the first comma
        first_name = names.split(',')[0] if ',' in names else names
        # Write the updated line back to the file
        file.write(f"{folder_code}\t{first_name}\n")
