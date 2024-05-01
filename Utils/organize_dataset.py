import os
import shutil

def organize_data(directory):
  i = 0
  for filename in os.listdir(directory):
    new_filename, ext = os.path.splitext(filename.replace(" ", ""))

    print(new_filename + " => " + ext)

    folder_path = os.path.join(directory, new_filename)

    try:
      os.makedirs(folder_path, exist_ok=True)
    except OSError as error:
      print(f"Error creating folder {folder_path}: {error}")

    source = os.path.join(directory, filename)
    destination = os.path.join(folder_path, f"{new_filename}({i}){ext}")

    try:
      shutil.move(source, destination)
      i += 1
    except OSError as error:
      print(f"Error moving file {filename}: {error}")
    


directory = "Dataset\ImagesQuery"
organize_data(directory)

print("Files organized successfully!")
