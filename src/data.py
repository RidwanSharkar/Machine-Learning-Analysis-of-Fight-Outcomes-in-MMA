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
    
def time_to_seconds(time_str):
    if pd.isnull(time_str):
        return np.nan
    mins, secs = map(int, time_str.split(':'))
    return 60 * mins + secs

def preprocess_data(df):
    if 'fighter' in df.columns:
        df['fighter'] = df['fighter'].str.lower()
    if 'time' in df.columns:
        df['time'] = df['time'].apply(time_to_seconds)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['days_since'] = (df['date'] - pd.Timestamp('2010-01-01')).dt.days
        df.drop(['date'], axis=1, inplace=True)
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (pd.Timestamp('now') - df['dob']).dt.days / 365.25
        df.drop(['dob'], axis=1, inplace=True)
    
    # Convert useful attributes 
    if 'stance' in df.columns:
        df['stance'] = df['stance'].map({'Orthodox': 0, 'Southpaw': 1})
    if 'method' in df.columns:
        df['method'] = df['method'].map({'KO/TKO': 0, 'SUB': 1, 'U-DEC': 2, 'S-DEC' : 3, 'DRAW' : 4, 'DQ' : 5 })

    numeric_columns = [
        'age', 'reach', 'height', 'total_comp_time', 'knockdowns', 'sub_attempts',
        'reversals', 'control', 'takedowns_landed', 'takedowns_attempts',
        'sig_strikes_landed', 'sig_strikes_attempts', 'total_strikes_landed',
        'total_strikes_attempts', 'head_strikes_landed', 'head_strikes_attempts',
        'body_strikes_landed', 'body_strikes_attempts', 'leg_strikes_landed',
        'leg_strikes_attempts', 'distance_strikes_landed', 'distance_strikes_attempts',
        'clinch_strikes_landed', 'clinch_strikes_attempts', 'ground_strikes_landed',
        'ground_strikes_attempts']
    
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    if df['result'].dtype == 'float64' or df['result'].dtype == 'int64':
        df['result'] = df['result'].astype('int').astype('category')  # Adjust

    return df

#===============================================================================================================#

def calculate_fighter_averages(df, fighter_name):
    fighter_data = df[df['fighter'] == fighter_name]

    if fighter_data.empty:
        print(f"No data available for fighter: {fighter_name}")
        return None

    # Columns to average
    cols_to_average = [
        'knockdowns', 'sub_attempts', 'reversals', 'control', 
        'takedowns_landed', 'takedowns_attempts', 'sig_strikes_landed', 
        'sig_strikes_attempts', 'total_strikes_landed', 'total_strikes_attempts', 
        'head_strikes_landed', 'head_strikes_attempts', 'body_strikes_landed', 
        'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 
        'distance_strikes_landed', 'distance_strikes_attempts', 'clinch_strikes_landed', 
        'clinch_strikes_attempts', 'ground_strikes_landed', 'ground_strikes_attempts'
    ]
    averages = fighter_data[cols_to_average].mean().to_dict()

    # Columns that retain their values
    averages['stance'] = fighter_data.iloc[-1]['stance']
    averages['reach'] = fighter_data.iloc[-1]['reach']
    averages['age'] = fighter_data.iloc[-1]['age']
    averages['height'] = fighter_data.iloc[-1]['height']
    averages['days_since_last_comp'] = fighter_data.iloc[-1]['days_since']
    averages['total_comp_time'] = fighter_data['total_comp_time'].sum()  # Sum, not average
    averages['num_fights'] = len(fighter_data)
    averages['KO_losses'] = fighter_data['KO_losses'].sum()  # Adjust if 'result' is encoded differently

    return averages


def predict_fight_result(fighter1_stats, fighter2_stats, model):
    combined_stats = pd.DataFrame([fighter1_stats, fighter2_stats])
    return model.predict(combined_stats)


def calculate_fighter_comparisons(fighter1_stats, fighter2_stats):
    # Test
    fighter_comparisons = {}

    for feature in fighter1_stats:
        if feature in fighter2_stats:
            # Subtraction***?
            if feature in ['reach', 'age', 'height', 'total_comp_time']:  # Example features
                fighter_comparisons[f"{feature}_diff"] = fighter1_stats[feature] - fighter2_stats[feature]
            # Categorical
            if feature in ['stance', 'method']:
                fighter_comparisons[f"{feature}_diff"] = 0 if fighter1_stats[feature] == fighter2_stats[feature] else 1

    return fighter_comparisons