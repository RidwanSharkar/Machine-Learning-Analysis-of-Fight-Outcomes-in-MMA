**Predictive Analysis of Fight Outcomes in Mixed Martial Arts Using Machine Learning**
________________________________________________________________________________________________________________

OVERVIEW:

This project attempts to predict the outcomes of MMA fights based on a diverse set of attributes sourced from historical fight data, consisting of fighter statistics, historical performance, physical attributes, and fighting styles to train multiple ML models. The goal is to see if an accurate prediction of a novel fight can be determined from a vast set of fighter attributes. 

________________________________________________________________________________________________________________

[IMPLEMENTATION 1]

ASSIGNING ATTRIBUTE WEIGHTS DETERMINANT TO FIGHT OUTCOME

Data set taken from (https://www.kaggle.com/datasets/danmcinerney/mma-differentials-and-elo {masterMLpublic.csv}) contains comprehensive historical fight data scraped from the official ufcstats.com. For each fight (row), attributes (columns) such as 'age', 'reach', 'height', 'significant strikes landed', 'knockdowns', 'submission attempts', 'reversals', 'control time', 'takedowns attempted', and 'time since last competition' are recorded. Differential calculations are performed on these attributes between the fighter and their opponent, revealing more valuable & relevant metrics, which are then suitable to be fed through various Machine Learning algorithms:

________________________________________________________________________________________________________________

'age_differential',
'height_differential',
'days_since_last_comp_differential',
'reach_differential',
'avg_head_strikes_absorbed_differential',
'total_comp_time_differential',
'avg_takedowns_def_differential',
'avg_distance_strikes_def_differential',
'avg_sig_strikes_absorbed_differential',
'avg_control_differential',
'avg_total_strikes_absorbed_differential',
'avg_takedowns_attempts_per_min_differential',
'win_loss_ratio_differential',
'avg_distance_strikes_absorbed_differential',
'avg_body_strikes_def_differential',
'avg_stamina_differential',
'avg_ground_strikes_def_differential',
'total_strikes_def_differential',
'avg_ground_strikes_attempts_per_min_differential',
'avg_clinch_strikes_def_differential',
'avg_clinch_strikes_attempts_per_min_differential',
'avg_head_strikes_attempts_differential',
'avg_head_strikes_landed_per_min_differential',
'avg_total_strikes_def_differential',
'avg_leg_strikes_def_differential',
'distance_strikes_def_differential',
'avg_ground_strikes_landed_per_min_differential',
'avg_clinch_strikes_landed_per_min_differential',
'head_strikes_def_differential',
'avg_sig_strikes_def_differential',
'sig_strikes_def_differential',
'avg_leg_strikes_attempts_per_min_differential',
'avg_head_strikes_def_differential',
'control_differential',
'avg_total_comp_time_differential',
'avg_total_distance_strikes_absorbed_differential',
'avg_ground_strikes_absorbed_differential',
'avg_body_strikes_attempts_per_min_differential',
'head_strikes_landed_per_min_differential',
'clinch_strikes_attempts_per_min_differential',
'avg_distance_strikes_attempts_per_min_differential',
'avg_clinch_strikes_absorbed_differential',
'num_fights_differential',
'avg_head_strikes_attempts_per_min_differential',
'avg_body_strikes_landed_per_min_differential',
'avg_total_body_strikes_absorbed_differential',
'avg_leg_strikes_landed_per_min_differential',
'body_strikes_def_differential',
'avg_takedowns_landed_per_min_differential',
'total_distance_strikes_absorbed_differential',
'avg_body_strikes_absorbed_differential',
'avg_total_clinch_strikes_absorbed_differential',
'avg_total_strikes_attempts_per_min_differential',
'avg_leg_strikes_absorbed_differential',
'avg_num_fights_differential',
'body_strikes_landed_per_min_differential',
'avg_distance_strikes_landed_per_min_differential',
'body_strikes_attempts_per_min_differential',
'distance_strikes_attempts_per_min_differential',
'avg_total_head_strikes_absorbed_differential',
'head_strikes_attempts_per_min_differential',
'total_head_strikes_absorbed_differential',
'avg_total_strikes_landed_per_min_differential',
'head_strikes_absorbed_differential',
'total_clinch_strikes_absorbed_differential',
'avg_sub_attempts_per_min_differential',
'total_sig_strikes_absorbed_differential',
... +100

________________________________________________________________________________________________________________

The following Machine Learning Models are then used to assign weights to the above attributes WRT predicting the outcome of a fight ('result' column):

1) Random Forest Classifier
2) Gradient Boosting Classifier
3) Support Vector Machine
4) Logistic Regression
5) K-Neighbors Classifier
6) Neural MLP Classifier
7) Decision Tree Classifier

Results and discrepencies between how these these ML models assigned their weights are described in: 

"Attribute Weights for Predicting Fight Outcome by Machine Learning Model.pdf" attached

________________________________________________________________________________________________________________

[IMPLEMENTATION 2]

PREDICTING A NOVEL HYPOTHETICAL FIGHT BETWEEN 2 FIGHTERS

This program then takes 2 fighter input names, and averages & standardizes both of their career stats for each of the relevant attributes that the ML model is learning from. The availablity & symmetry of these relevant fighter stats are crucial for how the attributes were originally chosen, permitting a flexible way to compare 2 fighters. 
Each model is then able to take this 2-fighter input and attempts to predict and display the winner of their hypothetical fight, along with the level of certainty with its prediction.  


![image](https://github.com/RidwanSharkar/Predictive-Analysis-of-MMA-Fights/assets/158855066/72453e23-83fc-49ce-9ce7-d251904e86c2)


