from sklearn.preprocessing import StandardScaler
import pandas as pd

def engineer_features(df):
    # Debugger might delete
    if 'fighter' in df.columns:
        fighter_series = df['fighter'].copy()

    # Process Categorical Variables
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'fighter']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Normalize
    scaler = StandardScaler()
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if 'fighter' not in df.columns:
        df.insert(0, 'fighter', fighter_series)

    return df