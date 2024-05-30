import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def find_fighter_stats(df, fighter_name):
    #Search
    fighter_data = df[df['fighter'].str.lower() == fighter_name.lower()]
    days_since_col = 'days_since_last_comp' if 'days_since_last_comp' in df.columns else None #debug*

    # Sum Columns
    stats = {
        
        'Fighter: ': fighter_name,
        'Record: ': fighter_data['method'].value_counts().to_dict(),
        'List of Opponents:': list(fighter_data['opponent'].unique()),

        'Division: ': list(fighter_data['division'].unique()),
        'Stance: ': fighter_data.iloc[0]['stance'] if not fighter_data.empty else 'Unknown',
        'Age: ': fighter_data['age'].max(),
        'Height: ': fighter_data.iloc[0]['height'] if not fighter_data.empty else 'Unknown',
        'Reach: ': fighter_data.iloc[0]['reach'] if not fighter_data.empty else 'Unknown',

        'Knockdowns: ': fighter_data['knockdowns'].sum(),
        'Submission Attempts: ': fighter_data['sub_attempts'].sum(),
        'Reversals: ': fighter_data['reversals'].sum(),
        'Control Time: ': fighter_data['control'].sum(),
        'KO Losses: ': (fighter_data['result'] == 'KO').sum(),  #HMM
        'Total Number of Fights: ': fighter_data.shape[0],
        'Total Competition Time: ': fighter_data['total_comp_time'].sum(),
        'Days Since Last Competition: ': fighter_data.iloc[-1][days_since_col] if not fighter_data.empty and days_since_col else 'Unknown',  # last row in data

        'Takedowns Landed: ': fighter_data['takedowns_landed'].sum(),
        'Takedowns Attempted': fighter_data['takedowns_attempts'].sum(),

        'Significant Strikes Landed: ': fighter_data['sig_strikes_landed'].sum(),
        'Significant Strikes Attempted: ': fighter_data['sig_strikes_attempts'].sum(),

        'Head Strikes Landed: ': fighter_data['head_strikes_landed'].sum(),
        'Head Strikes Attempted: ': fighter_data['head_strikes_attempts'].sum(),

        'Body Strikes Landed: ': fighter_data['body_strikes_landed'].sum(),
        'Body Strikes Attempted: ': fighter_data['body_strikes_attempts'].sum(),

        'Leg Strikes Landed: ': fighter_data['leg_strikes_landed'].sum(),
        'Leg Strikes Attempted: ': fighter_data['leg_strikes_attempts'].sum(),

        'Distance Strikes Landed: ': fighter_data['distance_strikes_landed'].sum(),
        'Distance Strikes Attempted: ': fighter_data['distance_strikes_attempts'].sum(),

        'Clinch Strikes Landed: ': fighter_data['clinch_strikes_landed'].sum(),
        'Clinch Strikes Attempted: ': fighter_data['clinch_strikes_attempts'].sum(),

        'Ground Strikes Landed: ': fighter_data['ground_strikes_landed'].sum(),
        'Ground Strikes Attempted: ': fighter_data['ground_strikes_attempts'].sum(),

        'Total Strikes Landed: ': fighter_data['total_strikes_landed'].sum(),
        'Total Strikes Attempted: ': fighter_data['total_strikes_attempts'].sum()

        #'lose_streak': max(fighter_data['lose_streak']) if not fighter_data.empty else 0,
        #'win_streak': max(fighter_data['win_streak']) if not fighter_data.empty else 0,
        #'win_loss_ratio': (fighter_data['result'] == 'win').sum() / fighter_data['result'].count() if fighter_data['result'].count() > 0 else 0,
    }
    
    return stats

def main():
    filepath = 'C:\\Users\\Lenovo\\Desktop\\MMA-Predictive-Analysis\\data\\masterMLpublic.csv'
    df = load_data(filepath)
    fighter_name = input("Enter Fighter Name: ")
    stats = find_fighter_stats(df, fighter_name)
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()