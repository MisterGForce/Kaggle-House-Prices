# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:03:26 2024

Using random forest regression to enter Kaggle House Prices competition.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Based on Kaggle Intro to Machine Learning Course

@author: Michael Goforth
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#load data
data_path = 'C:/Users/Michael/Data Science/House Prices'
home_data = pd.read_csv(data_path + '\\train.csv')

y = home_data.SalePrice

print(home_data.columns)

# features chosen as starting point
start_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
                  'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[start_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Random Forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
base_mae = mean_absolute_error(val_y, rf_val_predictions)

print('Estimated MAE with base features: {:,.0f}'.format(base_mae))