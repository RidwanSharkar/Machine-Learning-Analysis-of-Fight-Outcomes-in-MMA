**Predictive Analysis of Fight Outcomes in Mixed Martial Arts Using Machine Learning**

OVERVIEW:

This project attempts to predict the outcomes of MMA fights based on a diverse set of attributes sourced from historical fight data, consisting of fighter statistics, historical performance, physical attributes, and fighting styles to train a predictive model based on Random Forest. The goal is to see if an accurate prediction of a novel fight can be determined from a vast set of fighter attributes. 

====================================================

IMPLEMENTATION 1:

Data set taken from (https://www.kaggle.com/datasets/danmcinerney/mma-differentials-and-elo {masterMLpublic.csv}) contains comprehensive historical fight data scraped from the official ufcstats.com. For each fight (row), attributes (columns) such as 'age', 'reach', 'height', 'significant strikes landed', 'knockdowns', 'submission attempts', 'reversals', 'control time', 'takedowns attempted', and 'time since last competition' are recorded. Differential calculations are performed on these attributes between the fighter and their opponent, revealing more valuable & relevant metrics, which are then suitable to be fed through various Machine Learning algorithms:

====================================================

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
...+40

====================================================

The following Machine Learning Models are then used to assign weights to the above attributes FOR predicting the outcome of a fight:

1) Random Forest Classifier
2) Gradient Boostin Classifier
3) Support Vector Machine
4) Logistic Regression
5) K-Neighbors Classifier
6) Neural MLP Classifier
7) Decision Tree Classifier

Results and discrepencies between how these these ML models assigned their weights are described in: 
"Attribute Weights for Predicting Fight Outcome by Machine Learning Model.pdf"

====================================================

IMPLEMENTATION 2:

This program then takes 2 fighter input names, and averages & standardizes both of their career stats for each of the relevant attributes that the ML model is learning from. The availablity & obtainability of these relevant fighter stats leads to how the attributes are chosen in the previous implementation, allowing for a flexible and symmetric way to compare 2 fighters. 
Each Machine learning model is then able to take this novel input of 2 fighters and attempts to predict a winner in a hypothetical fight between them. 



