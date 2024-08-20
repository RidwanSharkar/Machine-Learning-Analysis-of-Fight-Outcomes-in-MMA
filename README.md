**Predictive Analysis of Fight Outcomes in Mixed Martial Arts Using Machine Learning**
________________________________________________________________________________________________________________

**OVERVIEW:**

This project attempts to predict the outcomes of MMA fights based on a diverse set of attributes sourced from historical fight data, consisting of fighter statistics, historical performance, physical attributes, and fighting styles to train multiple ML models. The goal is to see if an accurate prediction of a novel fight can be determined from a vast set of fighter attributes. 

________________________________________________________________________________________________________________

**[IMPLEMENTATION 1]**

ASSIGNING ATTRIBUTE WEIGHTS DETERMINANT TO FIGHT OUTCOME

Data set taken from (https://www.kaggle.com/datasets/danmcinerney/mma-differentials-and-elo {masterMLpublic.csv}) contains comprehensive historical fight data scraped from the official ufcstats.com. For each fight (row), the result & attributes (columns) such as 'age', 'reach', 'height', 'significant strikes landed', 'knockdowns', 'submission attempts' are recorded for both fighters. 

Differential calculations are performed on these attributes between the fighter and their opponent, which are then suitable to be fed through various Machine Learning algorithms (Random Forest, Gradient Boosting Classifier, Support Vector Machine, K-Neighbors Classifier) to assign weights to the above attributes WRT predicting the outcome of a fight ('result' column). Feature Importances and discrepencies between the ML models are described in "Attribute Weights for Predicting Fight Outcome by Machine Learning Model.pdf"

________________________________________________________________________________________________________________

• 'age_differential'<br>
• 'height_differential'<br>
• 'days_since_last_comp_differential'<br>
• 'reach_differential'<br>
• 'avg_head_strikes_absorbed_differential'<br>
• 'total_comp_time_differential'<br>
• 'avg_takedowns_def_differential'<br>
• 'avg_distance_strikes_def_differential'<br>
• 'avg_sig_strikes_absorbed_differential'<br>
• 'avg_control_differential'<br>
• 'avg_total_strikes_absorbed_differential'<br>
• 'avg_takedowns_attempts_per_min_differential'<br>
• 'win_loss_ratio_differential'<br>
• 'avg_distance_strikes_absorbed_differential'<br>
• 'avg_body_strikes_def_differential'<br>
• 'avg_stamina_differential'<br>
• 'avg_ground_strikes_def_differential'<br>
• 'total_strikes_def_differential'<br>
• 'avg_ground_strikes_attempts_per_min_differential'<br>
• 'avg_clinch_strikes_def_differential'<br>
• 'avg_clinch_strikes_attempts_per_min_differential'<br>
• 'avg_head_strikes_attempts_differential'<br>
• 'avg_head_strikes_landed_per_min_differential'<br>
• 'avg_total_strikes_def_differential'<br>
• 'avg_leg_strikes_def_differential'<br>
• 'distance_strikes_def_differential'<br>
• 'avg_ground_strikes_landed_per_min_differential'<br>
• 'avg_clinch_strikes_landed_per_min_differential'<br>
• 'head_strikes_def_differential'<br>
• 'avg_sig_strikes_def_differential'<br>
• 'sig_strikes_def_differential'<br>
• 'avg_leg_strikes_attempts_per_min_differential'<br>
• 'avg_head_strikes_def_differential'<br>
• 'control_differential'<br>
• 'avg_total_comp_time_differential'<br>
• 'avg_total_distance_strikes_absorbed_differential'<br>
• 'avg_ground_strikes_absorbed_differential'<br>
• 'avg_body_strikes_attempts_per_min_differential'<br>
• 'head_strikes_landed_per_min_differential'<br>
• 'clinch_strikes_attempts_per_min_differential'<br>
• 'avg_distance_strikes_attempts_per_min_differential'<br>
• 'avg_clinch_strikes_absorbed_differential'<br>
• 'num_fights_differential'<br>
• 'avg_head_strikes_attempts_per_min_differential'<br>
• 'avg_body_strikes_landed_per_min_differential'<br>
• 'avg_total_body_strikes_absorbed_differential'<br>
• 'avg_leg_strikes_landed_per_min_differential'<br>
• 'body_strikes_def_differential'<br>
• 'avg_takedowns_landed_per_min_differential'<br>
• 'avg_body_strikes_absorbed_differential'<br>
• 'avg_total_clinch_strikes_absorbed_differential'<br>
• 'avg_total_strikes_attempts_per_min_differential'<br>
• 'avg_leg_strikes_absorbed_differential'<br>
• 'avg_num_fights_differential'<br>
• 'body_strikes_landed_per_min_differential'<br>
• 'avg_distance_strikes_landed_per_min_differential'<br>
• 'body_strikes_attempts_per_min_differential'<br>
• 'distance_strikes_attempts_per_min_differential'<br>
• 'avg_total_head_strikes_absorbed_differential'<br>
• 'head_strikes_attempts_per_min_differential'<br>
• 'head_strikes_absorbed_differential',<br>
• 'avg_sub_attempts_per_min_differential',<br>
• ... +100

________________________________________________________________________________________________________________

**[IMPLEMENTATION 2]**

PREDICTING A NOVEL FIGHT BETWEEN 2 MMA ATHLETHES

This program then takes 2 fighter input names, and averages & standardizes both of their career stats for each of the relevant attributes that the ML model is learning from. The availablity & symmetry of these relevant fighter stats are crucial for how the attributes were originally chosen, permitting a flexible way to compare 2 fighters. 
Each model is then able to take this 2-fighter input and attempts to predict and display the winner of their hypothetical fight, along with the level of certainty with its prediction.  


![image](https://github.com/user-attachments/assets/52cefd71-831f-4503-a6da-a8ef694c9040)



