from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """Create new features and scale numeric features."""
    # Assuming existing functions like add_age_at_fight() and calculate_experience() are already defined and imported
    df = add_age_at_fight(df)
    df = calculate_experience(df)

    # Scale numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df