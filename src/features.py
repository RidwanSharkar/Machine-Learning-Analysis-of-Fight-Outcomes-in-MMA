from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def engineer_features(df):
    scaler = StandardScaler()

    if 'dob' in df.columns and 'fight_date' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'])
        df['fight_date'] = pd.to_datetime(df['fight_date'])
        df['age_at_fight'] = (df['fight_date'] - df['dob']).dt.days / 365.25
        df.drop(['dob', 'fight_date'], axis=1, inplace=True)

    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    df = df.select_dtypes(include=[np.number])

    return df