# train_model.py
import pathlib
import sys
import yaml
import joblib
import logging

import os
from itertools import product


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def setup_logging():
    # Configure logging to write to a file and also print to the console
    log_file_path = pathlib.Path(__file__).parent.as_posix() + sys.argv[5]  # Specify the path to your log file
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def train_model(X_train, y_train, model_type, hyperparameters, output_path):
    if model_type == 'logistic_regression':
        model = LogisticRegression(**hyperparameters)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**hyperparameters)
    elif model_type == 'svm':
        model = SVC(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    logging.info(f"Training {model_type} model with hyperparameters: {hyperparameters}")

    # Save the model using joblib
    model_folder = '_'.join(str(val) for val in hyperparameters.values())
    os.makedirs(os.path.join(output_path, model_folder), exist_ok=True)
    joblib.dump(model, os.path.join(output_path, model_folder, 'model.joblib'))


def main():

    setup_logging()

    curr_dir= pathlib.Path(__file__)
    home_dir= curr_dir.parent.parent.parent

    # parameters
    params_file= home_dir.as_posix()+ sys.argv[2]
    params= yaml.safe_load(open(params_file))


    # print(params_file)

    # data path
    data_path = home_dir.as_posix()+ sys.argv[1]

    # model path
    model_path = home_dir.as_posix()+ sys.argv[3]
    pathlib.Path(model_path).mkdir(parents=True,exist_ok=True)

    train_df = pd.read_csv(data_path+'/train_iris.csv')
    X_train= train_df.drop(columns=['encoded_labels','Species'], axis=1)
    y_train= train_df['encoded_labels']

    # Train models based on configurations from params.yaml
    for model_config in params['models']:
        model_type = model_config['model_type']
        hyperparameters_list = model_config['hyperparameters']

        for hyperparameters in product(*hyperparameters_list.values()):
            train_model(X_train, y_train, model_type, dict(zip(hyperparameters_list.keys(), hyperparameters)), f"models/{model_type}/")



if __name__ == '__main__':
    main()