from data import load_data, preprocess_data
from features import engineer_features
from model import train_model
from evaluate import evaluate_model
import pandas as pd
import numpy as np

def main():
    filepath_ml_data = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterMLpublic.csv'
    ml_data = load_data(filepath_ml_data)

    if ml_data is None:
        print("Data loading failed. Exiting program.")
        return

    print("DataFrame columns:", ml_data.columns)

    ml_data = preprocess_data(ml_data)
    ml_data = engineer_features(ml_data)

    target_column = 'result' 
    if target_column not in ml_data.columns:
        print(f"Target column '{target_column}' not found.")
        return

    X = ml_data.drop([target_column], axis=1)
    Y = ml_data[target_column]

    model, X_train, X_test, y_train, y_test = train_model(X, Y)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()