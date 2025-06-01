from Usecases.data import DataHandler
from Services.data import DataService


if __name__ == "__main__":
    file_path = "./Data"
    zip_file_name = "bank+marketing.zip"
    csv_file_name = "bank.csv"
    data_service = DataService(
        file_path=file_path,
        zip_file_name=zip_file_name,
        csv_file_name=csv_file_name
    )
    data_handler = DataHandler(
        file_path=file_path,
        zip_file_name=zip_file_name,
        csv_file_name=csv_file_name,
        data_service=data_service
    )
    data_handler.execute()
