import pandas as pd
import numpy as np

#======================================================================================================================

numeric_features = [
    'days_since_last_comp', 
    'sub_attempts',           
    'sub_landed',             
    'reversals',               
    'control',               
    'takedowns_landed',        
    'takedowns_attempts',     
    'sig_strikes_landed',      
    'sig_strikes_attempts',    
    'total_strikes_landed',   
    'total_strikes_attempts',  
    'head_strikes_landed',     
    'head_strikes_attempts',
    'body_strikes_landed',
    'body_strikes_attempts',
    'leg_strikes_landed',
    'leg_strikes_attempts',
    'distance_strikes_landed',
    'distance_strikes_attempts',
    'clinch_strikes_landed',
    'clinch_strikes_attempts',
    'ground_strikes_landed',
    'ground_strikes_attempts',
    'comp_time',
    'title_fight_loss',
    'title_fight_losses',
    'num_fights',
    #'win_streak',
    #'lose_streak',
    #'win_loss_ratio',
    'KO_losses',
    'sub_absorbed',
    'takedowns_absorbed',
    'sig_strikes_absorbed',
    'total_strikes_absorbed',
    'head_strikes_absorbed',
    'body_strikes_absorbed',
    'leg_strikes_absorbed',
    'distance_strikes_absorbed',
    'clinch_strikes_absorbed',
    'ground_strikes_absorbed',
    'total_sub_absorbed',
    'total_takedowns_absorbed',
    'total_sig_strikes_absorbed',
    'total_total_strikes_absorbed',
    'total_head_strikes_absorbed',
    'total_body_strikes_absorbed',
    'total_leg_strikes_absorbed',
    'total_distance_strikes_absorbed',
    'total_clinch_strikes_absorbed',
    'total_ground_strikes_absorbed']

categorical_features = ['fighter', 'opponent', 'stance', 'division', 'method', 'referee', 'time_format', 'fighter_url', 'opponent_url']
#=======================================================================================================================================

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully from", filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return None

def preprocess_data(df):
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

#=======================================================================================================================================

def print_fight_stats(df, fighter):
    filtered_df = df[df['fighter'].str.lower() == fighter.lower()]
    if filtered_df.empty:
        print("No records found for fighter:", fighter)
        return

    print(f"Individual Fight Stats for {fighter.title()}")
    for index, row in filtered_df.iterrows():
        print(f"Fight {index + 1}:")
        print(f"Opponent: {row['opponent'].title()}")
        for col in numeric_features:
            print(f"{col}: {row[col]}")
        print("-------")

#=======================================================================================================================================

def main():
    filepath_ml_data = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterMLpublic.csv'
    full_data = load_data(filepath_ml_data)
    if full_data is None:
        return
    
    full_data_preprocessed = preprocess_data(full_data)

    while True:
        fighter_name = input("Enter the fighter name (or type 'end' to exit): ").strip().lower()
        if fighter_name == 'end':
            break

        print_fight_stats(full_data_preprocessed, fighter_name)
        print()

if __name__ == '__main__':
    main()
