# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required Python libraries.

2. Load the weather station dataset.

3. Convert the time column into datetime format.

4. Extract month, day, and hour features from the time column.

5. Handle missing values in the dataset.

6. Select humidity, pressure, wind speed, month, day, and hour as input features.

7. Select temperature, PM2.5, and energy values as target variables.

8. Split the dataset into training and testing datasets.

9. Create the Random Forest Regression model.

10. Train the model using training data.

11. Predict the output using testing data.

12. Evaluate the model using R² Score, MAE, and RMSE.

13. Perform 5-fold cross-validation to verify model stability.

14. Display the prediction accuracy and final results. 

## Program:
```
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: SANTHOSH S
RegisterNumber: 212224100052
```
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('weather-station-eee-block_2024_07_13.csv')

# Display first five rows
print("First Five Rows of Dataset")
print(df.head())

# Convert time column into datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract date and time features
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Handle missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Select input features
X = df[['hum', 'pressure', 'wind_speed', 'month', 'day', 'hour']]

# Select target variables
Y = df[['tem', 'pm2_5', 'tsr']]

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, Y_train)

# Predict output values
Y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Display evaluation metrics
print("\nModel Evaluation Metrics")
print("R2 Score :", r2)
print("MAE :", mae)
print("RMSE :", rmse)

# Perform cross-validation
cv_scores = cross_val_score(
    model,
    X,
    Y,
    cv=5,
    scoring='r2'
)

# Display cross-validation results
print("\nCross Validation Scores")
print(cv_scores)

print("\nAverage Cross Validation Score")
print(cv_scores.mean())
```

## Output:
```
First Five Rows of Dataset
                      time   hum   tem    co2  illumination  pressure  pm2_5  \
0  2023-05-02T16:51:03.665  99.9  30.7  282.0       32450.0     999.9    3.0   
1  2023-05-02T16:31:03.670  95.5  30.5  308.0       21770.0     999.9    3.0   
2  2023-05-02T16:11:03.648  97.0  31.1  303.0       85460.0     999.6    3.0   
3  2023-05-02T15:51:03.655  96.8  31.3  283.0       32180.0    1000.2    4.0   
4  2023-05-02T15:31:03.623  91.2  32.3  267.0       57080.0     999.9    4.0   

   pm10 wind_direction  wind_direction_angle  wind_speed  wind_speed_level  \
0   3.0            NNW                 327.1         0.7               1.0   
1   3.0            ESE                 117.2         1.2               1.0   
2   3.0            NNW                 326.4         1.2               1.0   
3   5.0            SSW                 192.8         1.1               1.0   
4   4.0             NW                 308.4         0.7               1.0   

     tsr  bat  
0  146.0  NaN  
1  111.8  NaN  
2  407.1  NaN  
3  186.6  NaN  
4  351.3  NaN  
/tmp/ipykernel_694/1817645486.py:26: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[col].fillna(df[col].mean(), inplace=True)

Model Evaluation Metrics
R2 Score : 0.9411894187975166
MAE : 15.505216735205629
RMSE : 41.82340227344076

Cross Validation Scores
[-1.33219604 -0.43620669  0.63576983  0.3907114   0.64820146]

Average Cross Validation Score
-0.018744007574348374
```

## Result:
