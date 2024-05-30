from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data import load_data, preprocess_data, calculate_fighter_averages
from data import load_data, preprocess_data
from features import engineer_features
from data import calculate_fighter_comparisons
import shap

def display_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Top 25 Computed Weights: ")
    for feature, importance in sorted_features[:25]:
        print(f"{feature}: {importance:.2f}")

def explain_with_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.initjs() 
    return shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_train.iloc[0])

#=======================================================================================================#

def main():
    filepath_ml_data = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterMLpublic.csv'
    full_data = load_data(filepath_ml_data)

    if full_data is None:
        print("Data loading failed. Exiting program.")
        return
    
    #PREPROCESS
    full_data_preprocessed = preprocess_data(full_data)
    full_data_preprocessed = engineer_features(full_data_preprocessed)
    if 'fighter' not in full_data_preprocessed.columns:
        print("Error: 'fighter' column not found after preprocessing.")

    # ATTRIBUTE SELECTION
    important_features = [
    'stance', 'reach', 'age', 'height', 'days_since_last_comp', 'total_comp_time', 
    'num_fights', 'KO_losses', 'knockdowns', 'sub_attempts', 'reversals', 
    'control', 'takedowns_landed', 'takedowns_attempts', 'sig_strikes_landed', 
    'sig_strikes_attempts', 'total_strikes_landed', 'total_strikes_attempts', 
    'head_strikes_landed', 'head_strikes_attempts', 'body_strikes_landed', 
    'body_strikes_attempts', 'leg_strikes_landed', 'leg_strikes_attempts', 
    'distance_strikes_landed', 'distance_strikes_attempts', 'clinch_strikes_landed', 
    'clinch_strikes_attempts', 'ground_strikes_landed', 'ground_strikes_attempts']  
    full_data_reduced = full_data_preprocessed[important_features + ['result']]  # Include target variable

    # SPLIT DATA
    train_data, test_data = train_test_split(full_data_reduced, test_size=0.01, random_state=42)
    X_train = train_data.drop(['result'], axis=1)
    Y_train = train_data['result']

    # TRAIN MODEL
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    display_feature_importance(model, X_train.columns)
    print("=============================================================================================")
    #shap_plot = explain_with_shap(model, X_train)
    #shap_plot.show()
    #===================================================================================================#

    while True:
        fighter1_name = input("Enter first fighter name (or type 'end' to exit): ").strip().lower()
        if fighter1_name == 'end':  # Check if user wants to end the session
            break

        fighter2_name = input("Enter second fighter name (or type 'end' to exit): ").strip().lower()
        if fighter2_name == 'end':  # Check if user wants to end the session
            break

        # CALCULATE CAREER AVERAGES
        fighter1_stats = calculate_fighter_averages(full_data_preprocessed, fighter1_name)
        fighter2_stats = calculate_fighter_averages(full_data_preprocessed, fighter2_name)

        if not fighter1_stats or not fighter2_stats:
            print("Fighter data not available for one or both fighters.")
            return
        
        #==============================================================================#

        fighter_comparisons = calculate_fighter_comparisons(fighter1_stats, fighter2_stats)

        fight_input = pd.DataFrame([fighter_comparisons], columns=X_train.columns)
        probabilities = model.predict_proba(fight_input)
        print("Probability of outcomes (Fighter 1 wins, Fighter 2 wins):", probabilities[0])
        predicted_winner = "Fighter 1 wins" if probabilities[0][0] > 0.5 else "Fighter 2 wins"
        confidence = max(probabilities[0])
        print(f"Predicted fight result: {predicted_winner} with {confidence*100:.2f}% confidence")
        print("========================================================================")

        #==============================================================================#
        fight_input = pd.DataFrame([fighter1_stats, fighter2_stats], columns=X_train.columns)
        probabilities = model.predict_proba(fight_input)
        print("Probability of outcomes (Fighter 1 wins, Fighter 2 wins):", probabilities[0])
        predicted_winner = "Fighter 1 wins" if probabilities[0][0] > 0.5 else "Fighter 2 wins"
        confidence = max(probabilities[0])
        print(f"Predicted fight result: {predicted_winner} with {confidence*100:.2f}% confidence")
        print("========================================================================")

if __name__ == '__main__':
    main()