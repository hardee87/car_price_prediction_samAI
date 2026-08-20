import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Load dataset
df = pd.read_csv('cardata.csv')

# Feature Engineering
df['Car_Age'] = datetime.now().year - df['Year']
df_processed = df.drop(columns=['Car_Name', 'Year'])
df_encoded = pd.get_dummies(df_processed, drop_first=True)

# Split Features & Target
X = df_encoded.drop(columns=['Selling_Price'])
y = df_encoded['Selling_Price']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model and Columns
joblib.dump(model, 'car_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("Model trained and saved successfully as 'car_model.pkl'!")