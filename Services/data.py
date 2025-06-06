"""
    In this file only data related things are there

"""
import os
import zipfile
import pandas as pd

class DataService:
    def __init__(self, file_path, zip_file_name, csv_file_name):
        self.file_path = file_path
        self.zip_file_name = zip_file_name
        self.csv_file_name = csv_file_name
        self._processed_zips = set()  # Keep track of extracted zips

    def unzip_all(self):
        zip_path = os.path.abspath(os.path.join(self.file_path, self.zip_file_name))
        # print(f"Unzipping {zip_path}")
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        extract_to = os.path.dirname(zip_path)
        self._extract_and_find_nested(zip_path, extract_to)

    def _extract_and_find_nested(self, zip_path, extract_to):
        # Avoid reprocessing the same zip
        if zip_path in self._processed_zips:
            return
        self._processed_zips.add(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            # print(f"Extracted: {zip_path}")

        # Optionally delete the zip after extracting to prevent reprocessing
        # os.remove(zip_path)

        for root, _, files in os.walk(extract_to):
            for file in files:
                if file.endswith(".zip"):
                    nested_zip_path = os.path.abspath(os.path.join(root, file))
                    if nested_zip_path not in self._processed_zips:
                        # print(f"Found nested zip: {nested_zip_path}")
                        self._extract_and_find_nested(nested_zip_path, root)

    def load_dataset(self):
        """
            *********
                This code load the data set as a data frame
            *********

        :param
             file_path
             file_name
        :return
            DataFrame
        """

        if not self.file_path:
            raise FileNotFoundError("Please provide a file path")

        if not self.csv_file_name:
            raise FileNotFoundError("Please provide a file name")

        if not os.path.isdir(self.file_path):
            raise NotADirectoryError("Please provide a valid path")

        if not self.csv_file_name.endswith(".csv"):
            raise FileNotFoundError("Please Provide a CSV file name")

        # Construct the full path to the zip file
        file_full_path = os.path.normpath(os.path.join(self.file_path, self.csv_file_name))

        # Check if the zip file exists
        if not os.path.isfile(file_full_path):
            raise FileNotFoundError(f"The file {self.csv_file_name} does not exist at {self.file_path}")

        dataset = pd.read_csv(file_full_path,delimiter=';')

        return dataset
