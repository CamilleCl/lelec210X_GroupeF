import os

def rename_files(folder_path, start):
    # List all files in the folder
    files = os.listdir(folder_path)
    # Sort files based on their names
    files.sort()

    # Counter for numbering
    counter = start
    print(counter)
    # Previous prefix
    prev_prefix = None  

    # Iterate over each file
    for file_name in files:
        # Split the file name and extension
        name, ext = os.path.splitext(file_name)
        print(name)
        # Split the name at underscore and get the prefix
        prefix = name.split('_')[0]

        # If prefix changes, reset the counter
        if prefix != prev_prefix:
            counter = start

        # New file name pattern
        new_name = f'{prefix}_{counter:05d}.pickle'
        print(new_name)
        # Build the full old and new file paths
        old_path = os.path.join(folder_path, file_name)
        print(old_path)
        new_path = os.path.join(folder_path, new_name)
        print(new_path)
        # Rename the file
        os.rename(old_path, new_path)
        # Increment the counter
        counter += 1
        # Update previous prefix
        prev_prefix = prefix

# Example usage: provide the path to your folder
folder_path = "datasetCuisineGloe"
start = 650
rename_files(folder_path, start)

#pyth de 0 à 249
# chambre gloe de 250 à 449
# couloir gloe 450 à 649
# cuisine gloe 650 à 850