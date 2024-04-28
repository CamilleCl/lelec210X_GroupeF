# import os

# x = 7
# x /= 0
# for i in range(6):
#     try:
#         os.rename(f"handsaw_medium_{i:01d}.wav", f"handsaw_{i+6:03d}.wav")
#     except FileNotFoundError:
#         print(f"Fichier non trouvé : {i}")
#     except FileExistsError:
#         print(f"Le fichier {i} existe déjà.")

# print("Renommage terminé.")

import os

# Define the directory containing the files
directory = 'bigDatasetBIS/helico'

new_name = "birds"
i = 250

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Define the new filename
    new_filename = f'helicopter_{i:03d}.pickle'
    
    # Use os.rename() to rename the file
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

    i+=1

