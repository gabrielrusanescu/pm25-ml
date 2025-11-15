import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
air_data = pd.read_csv('bucharest_air_quality.csv')
meteo_data = pd.read_csv('bucharest_weather.csv')

# merge
data = pd.merge(air_data, meteo_data, on='datetime')

# handle missing values
data.fillna(method='ffill', inplace=True)

# features and target
features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
X = data[features]
y = data['PM2.5'] # target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print('Data preprocessing complete. Ready for model training.')