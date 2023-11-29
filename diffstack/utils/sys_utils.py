import os


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files.")
