# evaluate.py
import pathlib
import sys
import yaml
import joblib
import logging

import os
from itertools import product


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


import mlflow


def setup_logging():
    # Configure logging to write to a file and also print to the console
    log_file_path = pathlib.Path(__file__).parent.as_posix() + sys.argv[4]  # Specify the path to your log file
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Set the MLflow tracking URI
mlflow.set_tracking_uri('http://localhost:5000') 



def evaluate_model(X, y, model, metrics, cv_values):
    results = {}
    for cv in cv_values:
        cv_scores = {
            metric['metric_type']: cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=metric.get('scorer', None) or metric.get('type', 'accuracy')
            ).mean()
            for metric in metrics
        }
        results[cv] = cv_scores
        logging.info(f"Cross-validation results - CV: {cv}, Scores: {cv_scores}")
    return results


def main():

    setup_logging()

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
                
            # Check if the model file exists
            model = joblib.load(model_file_path)

            # Check if a run is active
            if mlflow.active_run():
                mlflow.end_run()


            # Log the evaluation metrics using MLflow
            with mlflow.start_run():
                mlflow.log_param("model_name", model_type)

                # Log hyperparameters
                mlflow.log_params(dict(zip(hyperparameters_list.keys(), hyperparameters)))

                metrics = params.get('evaluation', {}).get('metrics', [{'metric_type': 'accuracy'}])
                cv_values = params.get('evaluation', {}).get('cv', [5])

                scores = evaluate_model(X_test, y_test, model, metrics=metrics, cv_values=cv_values)

                for cv, cv_scores in scores.items():
                    for metric_type, score in cv_scores.items():
                        mlflow.log_metric(f"{metric_type}_cv_{cv}", score)



if __name__ == '__main__':
    main()
