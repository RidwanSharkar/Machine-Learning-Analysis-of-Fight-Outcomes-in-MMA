import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('C:\\Users\\Ridwa\\OneDrive\\Documents\\CODE\\MMA\\MMA-Predictive-Analysis\\data\\masterdataframe.csv')

# Basic preprocessing
data.fillna(data.mean(), inplace=True)  # Replace NaN with mean for numerical columns

# Feature engineering
data['age_at_fight'] = pd.to_datetime(data['date']).dt.year - pd.to_datetime(data['dob']).dt.year
data['experience'] = data['total_comp_time']  # or some other proxy

# Encode categorical variables
data = pd.get_dummies(data, columns=['stance', 'division'])

# Normalize features
scaler = StandardScaler()
features = ['age_at_fight', 'experience', 'reach', 'height']  # Add other features
data[features] = scaler.fit_transform(data[features])

# Split data
X = data.drop(['result'], axis=1)
y = data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


