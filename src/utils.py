
import dill
import sys
from src.exception import CustomException
import os
import logging
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            obj = dill.load(f)
        logging.info(f"Loaded object from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test, y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            # print(report)
            # {'Random Forest': 0.8528374168042437, 'Decision Tree': 0.7573543674159113, 
            #  'CatBoost Regressor': 0.8523560006768236, 'AdaBoost Regressor': 0.8531170911463195, 'Gradient Boosting': 0.8723159501494331,
            #   'Linear Regression': 0.8804353318506781, 'KNN Regressor': 0.47561174068704326, 
            #   'XGBoost Regressor': 0.8230898008444014}
        return report  

    except Exception as e:
        raise CustomException(e,sys)            