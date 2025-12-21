import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

from src.exception import HousingException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")



class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test array")
            X_train, y_train, X_test, Y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            models = {
                'linear_regression': LinearRegression(),  # fit_intercept=True by default
            }


            params={
                "linear_regression": {}
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, Y_test, models, params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]


            logging.info("Model trained successfully")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, 
                obj = best_model,
            )

            predicted = best_model.predict(X_test)
            r_squared = r2_score(Y_test, predicted)
            return r_squared


        except Exception as e:
            raise HousingException(e, sys)