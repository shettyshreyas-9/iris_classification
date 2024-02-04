# import pathlib
# import yaml
# import sys

# import mlflow
# from itertools import product
# from mlflow.sklearn

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC


# def main():

#     mlflow.set_tracking_uri("http://127.0.0.1:5000/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&datasetsFilter=W10%3D&modelVersionFilter=All%20Runs&selectedColumns=attributes.%60Source%60,attributes.%60Models%60&compareRunCharts=")  # Set your MLflow tracking URI


#     curr_dir= pathlib.Path(__file__)
#     home_dir= curr_dir.parent.parent.parent

#     # parameters
#     params_file= home_dir.as_posix()+ sys.argv[2]
#     params= yaml.safe_load(open(params_file))['models']

#     # print(params_file)

#     # data path
#     data_path = home_dir.as_posix()+ sys.argv[1]

#     # model path
#     model_path = home_dir.as_posix()+ sys.argv[3]
#     pathlib.Path(model_path).mkdir(parents=True,exist_ok=True)

#     train_df = pd.read_csv(data_path+'/train_iris.csv')
#     X= train_df.drop(columns=['encoded_labels','Species'], axis=1)
#     y= train_df['encoded_labels']



# ###############

#     # Train models based on configurations
#     for model_config in params:
#         model_type = model_config['model_type']
#         hyperparameters_list = model_config['hyperparameters']

#         for i, hyperparameters in enumerate(hyperparameters_list):
#             # Train model
#             def train_model(X_train, y_train, model_type, hyperparameters):
#                 with mlflow.start_run():
#                     # Train the specified model type
#                     if model_type == 'logistic_regression':
#                         model = LogisticRegression(**hyperparameters)
#                     elif model_type == 'random_forest':
#                         model = RandomForestClassifier(**hyperparameters)
#                     elif model_type == 'svm':
#                         model = SVC(**hyperparameters)
#                     else:
#                         raise ValueError(f"Unsupported model type: {model_type}")
        


#                     model.fit(X_train, y_train)

#                     # Log hyperparameters
#                     mlflow.log_params(hyperparameters)

#                     # Log metrics (you need to replace 'your_metric' with the actual metric)
#                     # mlflow.log_metric('your_metric', model.score(X_train, y_train))

#                     # Save the model
#                     model_save_loc = home_dir.as_posix() +sys.argv[3] + f"/{model_type}_{i}"
#                     # model_save_loc= 'C:\shreyas\ML\campusx\MLOps\week_3\iris\models'
#                     mlflow.sklearn.save_model(model, model_save_loc)


#             train_model(X,y,model_type,hyperparameters)

    


# ##########################

#     # # train model function

#     # for model_name,model_params in params.items():    # read params file

#     #     def train_model(X_train,y_train,model_name,model_params):      # create function for training
#     #         # for prm in model_params.items():           # loop through the list of params

#     #         with mlflow.start_run():       # start mlflow run

#     #             # Train the specific model with specific param combination
#     #             if model_name== 'logistic_regression':
#     #                 model = LogisticRegression(**model_params)
#     #             # elif model_name == 'random_forest':
#     #             #     model = RandomForestClassifier(params)
#     #             # elif model_name == 'svm':
#     #             #     model = SVC(**params)
#     #             else:
#     #                 raise ValueError(f"Unsupported model type: {model_name}")
                
#     #             model.fit(X_train, y_train)
            

#     #             # Log parameters
#     #             mlflow.log_params(model_params)

#     #             # Save the model
#     #             mlflow.sklearn.save_model(model, f"{model_name}_model")

#     #     train_model(X,y,model_name,model_params)





# if __name__ == '__main__':
#     mlflow.set_tracking_uri("http://127.0.0.1:5000/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&datasetsFilter=W10%3D&modelVersionFilter=All%20Runs&selectedColumns=attributes.%60Source%60,attributes.%60Models%60&compareRunCharts=")  # Set your MLflow tracking URI
#     # mlflow.set_experiment("your_experiment_name")  # Set your experiment name
#     main()
