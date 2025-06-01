"""
    THe file prepare the data and other stuff to go with
"""
import os
from Services.data import DataService

class DataHandler:
    def __init__(self, file_path, zip_file_name,csv_file_name, data_service):
        self.file_path = file_path
        self.zip_file_name = zip_file_name
        self.csv_file_name = csv_file_name
        self.data_service = DataService(
            file_path=self.file_path,
            zip_file_name=self.zip_file_name,
            csv_file_name=self.csv_file_name
        )

    def execute(self):

        # checks if the file is already present in the path
        csv_file_full_path = os.path.join(self.file_path, self.csv_file_name)

        if not os.path.isfile(csv_file_full_path):
            #unzip the files
            self.data_service.unzip_all()

            # Check if the zip file exists
            if not os.path.isfile(csv_file_full_path):
                raise FileNotFoundError(f"The file {self.csv_file_name} does not exist at {self.file_path}")

        data_set =  self.data_service.load_dataset()

        # get the data set tops
        print("This is the top Rows")
        data_set.head()

        print("This is the Data info")
        #get data set info
        data_set.info()



