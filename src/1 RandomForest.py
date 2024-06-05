from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#======================================================================================================================

important_features = [
'precomp_age_differential',
'precomp_height_differential',
'precomp_days_since_last_comp_differential',
'precomp_reach_differential',
'precomp_avg_head_strikes_absorbed_differential',
'precomp_total_comp_time_differential',
'precomp_avg_takedowns_def_differential',
'precomp_avg_distance_strikes_def_differential',
'precomp_avg_sig_strikes_absorbed_differential',
'precomp_avg_control_differential',
'precomp_avg_total_strikes_absorbed_differential',
'precomp_avg_takedowns_attempts_per_min_differential',
'precomp_win_loss_ratio_differential',
'precomp_avg_distance_strikes_absorbed_differential',
'precomp_avg_body_strikes_def_differential',
'precomp_avg_stamina_differential',
'precomp_avg_ground_strikes_def_differential',
'precomp_total_strikes_def_differential',
'precomp_avg_ground_strikes_attempts_per_min_differential',
'precomp_avg_clinch_strikes_def_differential',
'precomp_avg_clinch_strikes_attempts_per_min_differential',
'precomp_avg_head_strikes_attempts_differential',
'precomp_avg_head_strikes_landed_per_min_differential',
'precomp_avg_total_strikes_def_differential',
'precomp_avg_leg_strikes_def_differential',
'precomp_distance_strikes_def_differential',
'precomp_avg_ground_strikes_landed_per_min_differential',
'precomp_avg_clinch_strikes_landed_per_min_differential',
'precomp_head_strikes_def_differential',
'precomp_avg_sig_strikes_def_differential',
'precomp_sig_strikes_def_differential',
'precomp_avg_leg_strikes_attempts_per_min_differential',
'precomp_avg_head_strikes_def_differential',
'precomp_control_differential',
'precomp_avg_total_comp_time_differential',
'precomp_avg_total_distance_strikes_absorbed_differential',
'precomp_avg_ground_strikes_absorbed_differential',
'precomp_avg_body_strikes_attempts_per_min_differential',
'precomp_head_strikes_landed_per_min_differential',
'precomp_clinch_strikes_attempts_per_min_differential',
'precomp_avg_distance_strikes_attempts_per_min_differential',
'precomp_avg_clinch_strikes_absorbed_differential',
'precomp_num_fights_differential',
'precomp_avg_head_strikes_attempts_per_min_differential',
'precomp_avg_body_strikes_landed_per_min_differential',
'precomp_avg_total_body_strikes_absorbed_differential',
'precomp_avg_leg_strikes_landed_per_min_differential',
'precomp_body_strikes_def_differential',
'precomp_avg_takedowns_landed_per_min_differential',
'precomp_total_distance_strikes_absorbed_differential',
'precomp_avg_body_strikes_absorbed_differential',
'precomp_avg_total_clinch_strikes_absorbed_differential',
'precomp_avg_total_strikes_attempts_per_min_differential',
'precomp_avg_leg_strikes_absorbed_differential',
'precomp_avg_num_fights_differential',
'precomp_body_strikes_landed_per_min_differential',
'precomp_avg_distance_strikes_landed_per_min_differential',
'precomp_body_strikes_attempts_per_min_differential',
'precomp_distance_strikes_attempts_per_min_differential',
'precomp_avg_total_head_strikes_absorbed_differential',
'precomp_head_strikes_attempts_per_min_differential',
'precomp_total_head_strikes_absorbed_differential',
'precomp_avg_total_strikes_landed_per_min_differential',
'precomp_head_strikes_absorbed_differential',
'precomp_total_clinch_strikes_absorbed_differential',
'precomp_avg_sub_attempts_per_min_differential',
'precomp_total_sig_strikes_absorbed_differential',
'precomp_total_leg_strikes_absorbed_differential',
'precomp_avg_win_loss_ratio_differential',
'precomp_sig_strikes_absorbed_differential',
'precomp_avg_total_leg_strikes_absorbed_differential',
'precomp_avg_sig_strikes_landed_per_min_differential',
'precomp_avg_sub_def_differential',
'precomp_leg_strikes_attempts_per_min_differential',
'precomp_avg_total_sig_strikes_absorbed_differential',
'precomp_total_total_strikes_absorbed_differential',
'precomp_total_strikes_landed_per_min_differential',
'precomp_distance_strikes_absorbed_differential',
'precomp_stamina_differential',
'precomp_avg_total_total_strikes_absorbed_differential',
'precomp_avg_total_takedowns_absorbed_differential',
'precomp_total_strikes_absorbed_differential',
'precomp_avg_takedowns_absorbed_differential',
'precomp_avg_sig_strikes_attempts_per_min_differential',
'precomp_total_strikes_attempts_per_min_differential',
'precomp_total_body_strikes_absorbed_differential',
'precomp_avg_win_streak_differential',
'precomp_clinch_strikes_landed_per_min_differential',
'precomp_ground_strikes_def_differential',
'precomp_avg_total_ground_strikes_absorbed_differential',
'precomp_leg_strikes_landed_per_min_differential',
'precomp_total_ground_strikes_absorbed_differential',
'precomp_takedowns_def_differential',
'precomp_clinch_strikes_def_differential',
'precomp_sig_strikes_landed_per_min_differential',
'precomp_sig_strikes_attempts_per_min_differential',
'precomp_distance_strikes_landed_per_min_differential',
'precomp_takedowns_attempts_per_min_differential',
'precomp_ground_strikes_attempts_per_min_differential',
'precomp_body_strikes_absorbed_differential',
'precomp_avg_KO_losses_differential',
'precomp_leg_strikes_absorbed_differential',
'precomp_leg_strikes_def_differential',
'precomp_ground_strikes_landed_per_min_differential',
'precomp_total_takedowns_absorbed_differential',
'precomp_avg_lose_streak_differential',
'precomp_clinch_strikes_absorbed_differential',
'precomp_takedowns_landed_per_min_differential',
'precomp_ground_strikes_absorbed_differential',
'precomp_avg_sub_landed_per_min_differential',
'precomp_avg_total_sub_absorbed_differential',
'precomp_avg_knockdowns_differential',
'precomp_avg_reversals_differential',
'precomp_win_streak_differential',
'precomp_avg_sub_absorbed_differential',
'precomp_sub_attempts_per_min_differential',
'precomp_lose_streak_differential',
'precomp_takedowns_absorbed_differential',
'precomp_KO_losses_differential',
'precomp_sub_def_differential',
'precomp_total_sub_absorbed_differential',
'precomp_sub_landed_per_min_differential',
'precomp_knockdowns_differential',
'precomp_reversals_differential',
'precomp_sub_absorbed_differential',
'precomp_comp_time_differential',
'precomp_avg_comp_time_differential',
#--------------ACCURACY---------------
'precomp_avg_ground_strikes_acc_differential',
'precomp_avg_leg_strikes_acc_differential',
'precomp_avg_takedowns_acc_differential',
'precomp_head_strikes_acc_differential',
'precomp_avg_head_strikes_acc_differential',
'precomp_avg_sig_strikes_acc_differential',
'precomp_avg_body_strikes_acc_differential',
'precomp_body_strikes_acc_differential',
'precomp_avg_clinch_strikes_acc_differential',
'precomp_total_strikes_acc_differential',
'precomp_avg_distance_strikes_acc_differential',
'precomp_avg_total_strikes_acc_differential',
'precomp_distance_strikes_acc_differential',
'precomp_ground_strikes_acc_differential',
'precomp_sig_strikes_acc_differential',
'precomp_leg_strikes_acc_differential',
'precomp_clinch_strikes_acc_differential',
'precomp_avg_sub_acc_differential',
'precomp_takedowns_acc_differential',
'precomp_sub_acc_differential',
#---------------LANDED----------------
'precomp_avg_head_strikes_landed_differential',
'precomp_avg_sig_strikes_landed_differential',
'precomp_avg_total_strikes_landed_differential',
'precomp_avg_distance_strikes_landed_differential',
'precomp_avg_takedowns_landed_differential',
'precomp_avg_body_strikes_landed_differential',
'precomp_avg_leg_strikes_landed_differential',
'precomp_head_strikes_landed_differential',
'precomp_avg_clinch_strikes_landed_differential',
'precomp_avg_ground_strikes_landed_differential',
'precomp_sig_strikes_landed_differential',
'precomp_distance_strikes_landed_differential',
'precomp_total_strikes_landed_differential',
'precomp_body_strikes_landed_differential',
'precomp_leg_strikes_landed_differential',
'precomp_clinch_strikes_landed_differential',
'precomp_ground_strikes_landed_differential',
'precomp_takedowns_landed_differential',
'precomp_avg_sub_landed_differential',
'precomp_sub_landed_differential',
]

#======================================================================================================================

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully from", filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return None

#======================================================================================================================

def display_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Attributes' Computed Weights: ")
    for index, (feature, importance) in enumerate(sorted_features[:250], start=1):
        formatted_feature = ' '.join(word.capitalize() for word in feature.split('_'))
        print(f"{index}) {formatted_feature}: {importance:.3f}")

#======================================================================================================================

def preprocess_data(df):
    df['fighter'] = df['fighter'].str.lower()
    df['opponent'] = df['opponent'].str.lower()
    if 'date' in df.columns:
        df.drop('date', axis=1, inplace=True)

    for col in important_features:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
    
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

if __name__ == '__main__':
    main()