import pandas as pd
import numpy as np

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully from", filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return None

def preprocess_data(df):
    # Convert data types that should be numeric but are objects
    numeric_columns = ['age', 'reach', 'height', 'total_comp_time', 'knockdowns', 'sub_attempts', 'reversals', 'control',
                       'takedowns_landed', 'takedowns_attempts', 'sig_strikes_landed', 'sig_strikes_attempts', 
                       'total_strikes_landed', 'total_strikes_attempts', 'head_strikes_landed', 'head_strikes_attempts', 
                       'body_strikes_landed', 'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 
                       'distance_strikes_landed', 'distance_strikes_attempts', 'clinch_strikes_landed', 'clinch_strikes_attempts', 
                       'ground_strikes_landed', 'ground_strikes_attempts']

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.fillna(numeric_df.mean(), inplace=True)

    for column in numeric_df.columns:
        df[column] = numeric_df[column]

    non_numeric_df = df.select_dtypes(exclude=[np.number])
    for column in non_numeric_df.columns:
        df[column] = df[column].fillna('Unknown')

    return df
