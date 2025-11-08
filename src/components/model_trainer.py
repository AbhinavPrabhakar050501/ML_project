import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,load_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],  # All rows, all columns except last
                train_array[:,-1],   # All rows, only last column
                test_array[:,:-1],   # All rows, all columns except last
                test_array[:,-1],    # All rows, only last column
            )
            models = {
            
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose = False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),

            }

            model_report = evaluate_models(X_train=X_train,y_train = y_train, X_test= X_test, y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print(best_model_name)

            best_model = models[best_model_name]

            preprocessor = load_object(preprocessor_path)   # load saved preprocessor
            final_pipeline = Pipeline([("preprocessor", preprocessor),
                                        ("model", best_model)])    

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                            obj=final_pipeline)
            logging.info("Trained model saved")

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)

            print(r2)

        except Exception as e:
            raise CustomException(e,sys)
                        
            