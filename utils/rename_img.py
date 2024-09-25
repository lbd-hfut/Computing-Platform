import os

# Get the current working directory
folder_path = os.getcwd()

# Get all files in the directory
files = sorted(os.listdir(folder_path))
index = 0
# Initialize counter
for i, file_name in enumerate(files):
    # Get the full path of the file
    old_file_path = os.path.join(folder_path, file_name)
    
    # Get the file extension
    _, ext = os.path.splitext(file_name)

    # Check if the file is an image of the specified formats
    if ext.lower() in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        # Rename based on the index
        if index == 0:
            new_file_name = f"r{index:05d}{ext}"  # First image starts with 'r'
        else:
            new_file_name = f"d{index:05d}{ext}"  # Subsequent images start with 'd'

        new_file_path = os.path.join(folder_path, new_file_name)
        index = index + 1
        # Rename the file
        os.rename(old_file_path, new_file_path)

print("Renaming complete!")
