# code/your_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import yaml

def load_data():
    df = pd.read_csv("data/iris.csv")
    # df = sklearn.load_data('iris.csv')
    return df

def preprocess_data(df):
    # Your data preprocessing steps here
    return df

def tune_hyperparameters(model_type, X_train, y_train, params):
    if model_type == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Your hyperparameter tuning logic here
    # For example, use grid search, random search, or any other approach

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def train_model(X_train, y_train, X_test, y_test, model_type, params):
    with mlflow.start_run():
        # Train the specified model type
        model = tune_hyperparameters(model_type, X_train, y_train, params)
        
        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save the model
        mlflow.sklearn.save_model(model, f"{model_type}_model")

def main():
    # Load parameters from params.yaml
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    # Load data
    data = load_data()

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data into features and target
    X = preprocessed_data.drop('species', axis=1)
    y = preprocessed_data['species']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models based on configurations in params.yaml
    for model_type, model_params in params['models'].items():
        train_model(X_train, y_train, X_test, y_test, model_type, model_params)

if __name__ == "__main__":
    main()
