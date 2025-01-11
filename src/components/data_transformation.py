import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()

  def get_data_transformer_object(self):
    '''
      This function is responsible for data transformation
    '''

    try:
      ## Step1: Differentiate columns between numericals and categoricals columns
      numerical_columns = ["writing_score", "reading_score"]
      categorical_columns = [
          "gender",
          "race_ethnicity",
          "parental_level_of_education",
          "lunch",
          "test_preparation_course",
      ]


      numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), ## fill the na values using median strategy
        ("standard_scaler", StandardScaler())
      ])

      categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), ## fill the na values using mode strategy
        ("one_hot_encoding", OneHotEncoder(drop='first')),
        ("standard_scaler", StandardScaler(with_mean=False))
      ])

      preprocessor= ColumnTransformer([
        ("numerical_pipeline", numerical_pipeline, numerical_columns),
        ("categorical_pipeline", categorical_pipeline, categorical_columns)
      ])

      return preprocessor

    except Exception as e:
      raise CustomException(e, sys)
    
  def initiate_data_transformer(self, train_path, test_path):
    try:
      ## Step1: import CV
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)

      logging.info("Read train and test data completed")

      logging.info("Obtaining preprocessing object")

      ## Step2: get transformer object
      preprocessor = self.get_data_transformer_object()

      ## Step3: Divide dependent and independent Feature
      target_column_name="math_score"

      input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
      target_feature_train_df=train_df[target_column_name]

      input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
      target_feature_test_df=test_df[target_column_name]

      logging.info(
        f"Applying preprocessing object on training dataframe and testing dataframe."
      )

      input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessor.transform(input_feature_test_df)

      train_arr = np.c_[
          input_feature_train_arr, np.array(target_feature_train_df)
      ]
      test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

      logging.info(f"Saved preprocessing object.")

      save_object(
          file_path=self.data_transformation_config.preprocessor_obj_file_path,
          obj=preprocessor
      )

      return (
          train_arr,
          test_arr,
          self.data_transformation_config.preprocessor_obj_file_path,
      )

    except Exception as e:
        raise CustomException(e,sys)
    


  