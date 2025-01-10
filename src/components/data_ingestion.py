## importing data
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
  raw_data_path: str=os.path.join('artifacts', "data.csv")
  train_data_path: str=os.path.join('artifacts', "train.csv")
  test_data_path: str=os.path.join('artifacts', "test.csv")


class DataIngestion:
  def __init__(self):
    self.data_ingestion_config = DataIngestionConfig()
  
  def initiate_data_ingestion(self):
    logging.info("Initiated Data Ingestion")
    try:
      df=pd.read_csv('./notebook/data/stud.csv')
      logging.info('Read the dataset as dataframe')
      os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

      df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

      logging.info("Train test split initiated")
      train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

      train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

      test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

      logging.info("Ingestion of the data iss completed")

      return(
          self.data_ingestion_config.train_data_path,
          self.data_ingestion_config.test_data_path
      )
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__=='__main__':
  ## Step1: Inititate Data ingestion
  obj = DataIngestion()
  train_data_path,test_data_path = obj.initiate_data_ingestion()

  ## Step2: Inititate Data tranformer
  dataTransformation = DataTransformation()
  train_arr,test_arr,_ = dataTransformation.initiate_data_transformer(train_data_path,test_data_path)

  modeltrainer=ModelTrainer()
  print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
