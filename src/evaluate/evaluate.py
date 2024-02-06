# evaluate.py
import pathlib
import sys
import yaml
import joblib

import os
from itertools import product


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri('http://localhost:5000') 

def evaluate_model(X_test, y_test, model):
    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate accuracy (modify this based on your actual evaluation metric)
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='micro')

    recall = recall_score(y_test, y_pred,average='micro')

    return accuracy, recall, accuracy

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    # parameters
    params_file = home_dir.as_posix() + sys.argv[2]
    params = yaml.safe_load(open(params_file))

    # data path
    data_path = home_dir.as_posix() + sys.argv[1]

    # model path
    model_path = home_dir.as_posix() + sys.argv[3]

    # load test data
    test_df = pd.read_csv(data_path + '/test_iris.csv')
    X_test = test_df.drop(columns=['encoded_labels', 'Species'], axis=1)
    y_test = test_df['encoded_labels']


    # MLflow experiment name
    experiment_name= "default_ml_2"

    # Evaluate models based on configurations from params.yaml
    for model_config in params['models']:
        model_type = model_config['model_type']
        hyperparameters_list = model_config['hyperparameters']

        for hyperparameters in product(*hyperparameters_list.values()):
            # Load the trained model
            model_folder = '_'.join(str(val) for val in hyperparameters)
            model_file_path = os.path.join(model_path, f"{model_type}/{model_folder}/model.joblib")
                
            model = joblib.load(model_file_path)

            # Check if a run is active
            if mlflow.active_run():
                mlflow.end_run()


            # Log the evaluation metrics using MLflow
            with mlflow.start_run():

                mlflow.log_param("model_name", model_type)

                # Evaluate the model
                accuracy,precision,recall = evaluate_model(X_test, y_test, model)

                mlflow.log_params(dict(zip(hyperparameters_list.keys(), hyperparameters)))
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                # mlflow.log_artifact(model_file_path, "models")  # Log the model artifact


if __name__ == '__main__':
    main()
