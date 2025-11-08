import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass #used to define class variable directly without init
class DataIngestionConfig:
     train_data_path: str= os.path.join('artifact','train.csv')
     test_data_path: str= os.path.join('artifact','test.csv')
     raw_data_path: str= os.path.join('artifact','data.csv')

#use dataclass if you only want to define variables
# inside a class. and not define any functions.

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
       logging.info("Entered the data ingestion method:")
       try:
           
           file_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebook", "data", "stud.csv")
           file_path = os.path.abspath(file_path)
           df = pd.read_csv(file_path)
           logging.info("Read the dataset as dataframe")

           #make the directory exist 
           os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
           df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
           
           logging.info("Train test data initiated")

           train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
           train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
           test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

           logging.info("Ingestion of the data is completed")

           return(
               self.ingestion_config.train_data_path,
               self.ingestion_config.test_data_path
           )

       except Exception as e:
           raise CustomException(e,sys)
       
if __name__ == "__main__":
    print(os.getcwd())
    obj = DataIngestion()
    obj.initiate_data_ingestion()       