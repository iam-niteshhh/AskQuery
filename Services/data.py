"""
    In this file only data related things are there

"""
import os
import zipfile


def unzip_the_file(file_path="../Data", file_name="bank+marketing.zip"):
    """
        The code walks through the path and unzips the provided file.
        It also checks if there are any nested zip files inside and extracts them as well.
    """

    # Validate file path
    if not file_path:
        raise FileNotFoundError("Please provide a file path")

    # Validate file name
    if not file_name:
        raise FileNotFoundError("Please provide a file name")

    # Check if the provided path is a valid directory
    if not os.path.isdir(file_path):
        raise NotADirectoryError("Please provide a valid path")

    # Construct the full path to the zip file
    file_full_path = os.path.join(file_path, file_name)

    # Check if the zip file exists
    if not os.path.isfile(file_full_path):
        raise FileNotFoundError(f"The file {file_name} does not exist at {file_path}")

    try:
        # Open the zip file
        with zipfile.ZipFile(file_full_path, mode="r") as zip_ref:
            is_nested_zips = False

            # Iterate over all files in the zip archive
            for file in zip_ref.namelist():
                # If a nested zip file is found
                if file.endswith(".zip"):
                    nested_file_path = os.path.join(file_path, file)
                    with zipfile.ZipFile(nested_file_path, mode="r") as nested_zip:
                        print(f"Found nested zip: {file}")
                        nested_zip.extractall(path=file_path)
                        is_nested_zips = True

            # If no nested zips are found, just extract all the files from the main zip
            if not is_nested_zips:
                zip_ref.extractall(path=file_path)
                print(f"Extracted all files from {file_name} to {file_path}")

            return True

    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"{file_name} is not a valid zip file.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


print(unzip_the_file())