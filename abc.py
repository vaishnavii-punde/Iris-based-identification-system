import os

root_folder = "dataset_processed"
empty_folders = []

for root, dirs, files in os.walk(root_folder):
    # Folder is empty if it has no files AND no subfolders
    if len(files) == 0 and len(dirs) == 0:
        empty_folders.append(root)

print("\n------ EMPTY FOLDERS ------")
for folder in empty_folders:
    print(folder)

print("\nTotal empty folders:", len(empty_folders))
