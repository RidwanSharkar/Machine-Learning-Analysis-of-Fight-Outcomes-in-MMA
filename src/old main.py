from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#======================================================================================================================

important_features = [
    'days_since_last_comp_differential',    
    'sub_attempts_differential',            
    'sub_landed_differential',            
    'reversals_differential',               
    'control_differential',                
    'takedowns_landed_differential',       
    'takedowns_attempts_differential',     
    'sig_strikes_landed_differential',     
    'sig_strikes_attempts_differential',    
    'total_strikes_landed_differential',    
    'total_strikes_attempts_differential',  
    'head_strikes_landed_differential',     
    'head_strikes_attempts_differential',
    'body_strikes_landed_differential',
    'body_strikes_attempts_differential',
    'leg_strikes_landed_differential',
    'leg_strikes_attempts_differential',
    'distance_strikes_landed_differential',
    'distance_strikes_attempts_differential',
    'clinch_strikes_landed_differential',
    'clinch_strikes_attempts_differential',
    'ground_strikes_landed_differential',
    'ground_strikes_attempts_differential',
    'comp_time_differential',
    'title_fight_loss_differential',
    'title_fight_losses_differential',
    'num_fights_differential',
    #'win_streak_differential',
    #'lose_streak_differential',
    #'win_loss_ratio_differential',
    'KO_losses_differential',
    'sub_absorbed_differential',
    'takedowns_absorbed_differential',
    'sig_strikes_absorbed_differential',
    'total_strikes_absorbed_differential',
    'head_strikes_absorbed_differential',
    'body_strikes_absorbed_differential',
    'leg_strikes_absorbed_differential',
    'distance_strikes_absorbed_differential',
    'clinch_strikes_absorbed_differential',
    'ground_strikes_absorbed_differential',
    'total_sub_absorbed_differential',
    'total_takedowns_absorbed_differential',
    'total_sig_strikes_absorbed_differential',
    'total_total_strikes_absorbed_differential',
    'total_head_strikes_absorbed_differential',
    'total_body_strikes_absorbed_differential',
    'total_leg_strikes_absorbed_differential',
    'total_distance_strikes_absorbed_differential',
    'total_clinch_strikes_absorbed_differential',
    'total_ground_strikes_absorbed_differential',
    'total_comp_time_differential',
    'stamina_differential',
    'odds_differential',
    'sub_acc_differential',
    'takedowns_acc_differential',
    'sig_strikes_acc_differential',
    'total_strikes_acc_differential',
    'head_strikes_acc_differential',
    'body_strikes_acc_differential',
    'leg_strikes_acc_differential',
    'distance_strikes_acc_differential',
    'clinch_strikes_acc_differential',
    'ground_strikes_acc_differential',
    'sub_def_differential',
    'takedowns_def_differential',
    'sig_strikes_def_differential',
    'total_strikes_def_differential',
    'head_strikes_def_differential',
    'body_strikes_def_differential',
    'leg_strikes_def_differential',
    'distance_strikes_def_differential',
    'clinch_strikes_def_differential',
    'ground_strikes_def_differential',
    'sub_attempts_per_min_differential',
    'sub_landed_per_min_differential',
    'takedowns_landed_per_min_differential',
    'takedowns_attempts_per_min_differential',
    'sig_strikes_landed_per_min_differential',
    'sig_strikes_attempts_per_min_differential',
    'total_strikes_landed_per_min_differential',
    'total_strikes_attempts_per_min_differential',
    'head_strikes_landed_per_min_differential',
    'head_strikes_attempts_per_min_differential',
    'body_strikes_landed_per_min_differential',
    'body_strikes_attempts_per_min_differential',
    'leg_strikes_landed_per_min_differential',
    'leg_strikes_attempts_per_min_differential',
    'distance_strikes_landed_per_min_differential',
    'distance_strikes_attempts_per_min_differential',
    'clinch_strikes_landed_per_min_differential',
    'clinch_strikes_attempts_per_min_differential',
    'ground_strikes_landed_per_min_differential',
    'ground_strikes_attempts_per_min_differential']

raw_features = [
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
    'total_ground_strikes_absorbed',
    'total_comp_time',
    'stamina',
    'odds',
    'sub_acc',
    'takedowns_acc',
    'sig_strikes_acc',
    'total_strikes_acc',
    'head_strikes_acc',
    'body_strikes_acc',
    'leg_strikes_acc',
    'distance_strikes_acc',
    'clinch_strikes_acc',
    'ground_strikes_acc',
    'sub_def',
    'takedowns_def',
    'sig_strikes_def',
    'total_strikes_def',
    'head_strikes_def',
    'body_strikes_def',
    'leg_strikes_def',
    'distance_strikes_def',
    'clinch_strikes_def',
    'ground_strikes_def',
    'sub_attempts_per_min',
    'sub_landed_per_min',
    'takedowns_landed_per_min',
    'takedowns_attempts_per_min',
    'sig_strikes_landed_per_min',
    'sig_strikes_attempts_per_min',
    'total_strikes_landed_per_min',
    'total_strikes_attempts_per_min',
    'head_strikes_landed_per_min',
    'head_strikes_attempts_per_min',
    'body_strikes_landed_per_min',
    'body_strikes_attempts_per_min',
    'leg_strikes_landed_per_min',
    'leg_strikes_attempts_per_min',
    'distance_strikes_landed_per_min',
    'distance_strikes_attempts_per_min',
    'clinch_strikes_landed_per_min',
    'clinch_strikes_attempts_per_min',
    'ground_strikes_landed_per_min',
    'ground_strikes_attempts_per_min']

#======================================================================================================================

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

#======================================================================================================================

def display_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Attributes' Computed Weights: ")
    for index, (feature, importance) in enumerate(sorted_features[:40], start=1):
        formatted_feature = ' '.join(word.capitalize() for word in feature.split('_'))
        print(f"{index}) {formatted_feature}: {importance:.3f}")

#======================================================================================================================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

#======================================================================================================================

def preprocess_data(df):
    df['fighter'] = df['fighter'].str.lower()
    df['opponent'] = df['opponent'].str.lower()
    if 'date' in df.columns:
        df.drop('date', axis=1, inplace=True)
    
    all_features = set(raw_features + important_features)
    for col in all_features:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

#======================================================================================================================

def calculate_average_stats(df, fighter):
    stats = df[df['fighter'] == fighter][raw_features].mean()
    return stats

def get_symmetric_features(df, fighter1, fighter2):
    stats1 = calculate_average_stats(df, fighter1)
    stats2 = calculate_average_stats(df, fighter2)
    
    differential_stats = stats1 - stats2
    differential_stats = differential_stats.to_frame().T  # Transpose to make it a single row DataFrame
    differential_stats.columns = [col + '_differential' for col in differential_stats.columns]  # Rename columns
    
    return differential_stats.iloc[0]  # Returns Series

def predict_fight_outcome(model, data, fighter1, fighter2):
    diff1 = get_symmetric_features(data, fighter1, fighter2)
    diff2 = get_symmetric_features(data, fighter2, fighter1)

    prob1 = model.predict_proba(pd.DataFrame([diff1]))[0]
    prob2 = model.predict_proba(pd.DataFrame([diff2]))[0]
    

    final_prob = (prob1 + prob2[::-1]) / 2

    return final_prob
    
#======================================================================================================================

def main():
    filepath_ml_data = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterMLpublic.csv'
    full_data = load_data(filepath_ml_data)
    if full_data is None:
        print("Data loading failed. Exiting program.")
        return
    
    #PREPROCESS
    print("Preprocessing data...")
    full_data_preprocessed = preprocess_data(full_data)
    if 'fighter' not in full_data_preprocessed.columns:
        print("Error: 'fighter' column not found after preprocessing.")
    print("Data is clean. Proceeding with model training...")
    full_data_reduced = full_data_preprocessed[important_features + ['result']]

    #SPLIT
    train_data, test_data = train_test_split(full_data_reduced, test_size=0.10, random_state=42)
    X_train = train_data.drop(['result'], axis=1)
    Y_train = train_data['result']

    #TRAIN
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    display_feature_importance(model, X_train.columns)

    #INPUT PREDICTION
    while True:
        fighter1_name = input("Enter first fighter name: ").strip().lower()
        if fighter1_name == 'end':
            break
        fighter2_name = input("Enter second fighter name: ").strip().lower()
        if fighter2_name == 'end':
            break

        fighter_differential = get_symmetric_features(full_data_preprocessed, fighter1_name, fighter2_name)
        print("Differential Stats for {} vs {}:".format(fighter1_name, fighter2_name), fighter_differential)
        print()
        print(fighter_differential.to_string())

        fight_input = pd.DataFrame([fighter_differential]).reindex(columns=X_train.columns, fill_value=0)
        probability = model.predict_proba(fight_input)[0]
        print("Probability of outcomes (Fighter 1 wins, Fighter 2 wins):", probability)
        predicted_winner = "Fighter 1 wins" if probability[0] > 0.5 else "Fighter 2 wins"
        confidence = max(probability)
        print(f"Predicted fight result: {predicted_winner} with {confidence*100:.2f}% confidence")
        print("========================================================================")

if __name__ == '__main__':
    main()