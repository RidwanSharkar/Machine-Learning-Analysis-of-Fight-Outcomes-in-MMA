from data import load_data, preprocess_data
from features import engineer_features
from model import train_model, predict
from evaluate import evaluate_model
import pandas as pd

def main():
    # Define file paths for the datasets
    filepath_data = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterdataframe.csv'
    filepath_ml_data = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterMLpublic.csv'

    # Load datasets
    data = load_data(filepath_data)
    ml_data = load_data(filepath_ml_data)

    # Check if data is loaded
    if data is None or ml_data is None:
        print("Data loading failed. Exiting program.")
        return

    # Preprocess both datasets
    data = preprocess_data(data)
    ml_data = preprocess_data(ml_data)

    # Merge datasets on a common key, assuming 'fighter' is the common key
    combined_data = pd.merge(data, ml_data, on='fighter', how='inner')

    # Feature engineering
    combined_data = engineer_features(combined_data)

    # Prepare data for model training
    X = combined_data.drop(['result'], axis=1)
    y = combined_data['result']

    # Train the model
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()