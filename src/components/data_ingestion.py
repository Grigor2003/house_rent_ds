import os
import sys
from src.logger import logging
from src.exception import HousingException


import pandas as pd
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")



class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Running data ingestion component")
        try:
            df = pd.read_csv("Data/rent.csv")
            logging.info(f"Original dataset size: {len(df)}")
            
            # Remove max rent outlier
            max_rent = df['Rent'].max()
            index_max_rent = df[df['Rent'] == max_rent].index
            df = df.drop(index_max_rent)
            
            # Remove outliers using 5-95 percentiles
            # lower_bound = df['Rent'].quantile(0.2)
            # upper_bound = df['Rent'].quantile(0.8)
            
            # df = df[(df['Rent'] >= lower_bound) & (df['Rent'] <= upper_bound)]
            # logging.info(f"After 5-95 percentile filter (Rent {lower_bound:.0f} - {upper_bound:.0f}): {len(df)} rows")
            
            # Drop unnecessary columns
            df.drop('Area Locality', axis= 1, inplace = True)
            df.drop('Point of Contact', axis= 1, inplace = True)
            # df.drop('Bathroom', axis= 1, inplace = True)
            df.drop('Area Type', axis= 1, inplace = True)
            # df.drop('BHK', axis= 1, inplace = True)

            logging.info("Read dataset successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)
            logging.info("Raw data saved")


            train, test = train_test_split(df, test_size= 0.2, random_state= 42)
            train.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise HousingException(e, sys)