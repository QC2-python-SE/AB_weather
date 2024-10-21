# -*- coding: utf-8 -*-
"""
Weather Prediction Script
=========================

This script includes the following steps:
- Reading a dataset
- Training a machine learning model
- Plotting data
- Making predictions using user-provided information

Dependencies:
- numpy
- pandas
- matplotlib
- scikit-learn

Author: Bruno Camino & Abbie Bray
Date: YYYY-MM-DD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
import copy

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

def read_dataset(file_path: str) -> pd.DataFrame:

    """
    Reads a dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Cleans the data by:
     1. keeping only the columns containing the features we want to use 
        to build the model and the target property as the last column 
        (e.g., precipitation);
     2. removing NaN values and duplicates.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    df = df[features+[target]] #only keep the features and target

    df = df.dropna()  # Remove rows with NaN values
    df = df.drop_duplicates()  # Remove duplicate rows
    df = df.reset_index(drop=True)  # Reset index
    
    return df

def train_model(df: pd.DataFrame):
    """
    Trains a RandomForestRegressor on the provided dataset.
 
    Args:
        df (pd.DataFrame): The input dataset.
 
    Returns:
        RandomForestRegressor: The trained model.
        list: The feature columns.
        np.ndarray: The target column.
    """
    
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.2f}")
    
    return model, features.columns, target
 
def plot_data(df: pd.DataFrame):
    """
    Plots the dataset.
 
    Args:
        df (pd.DataFrame): The input dataset.
    """
    pd.plotting.scatter_matrix(df, figsize=(12, 8), diagonal='kde')
    plt.show()
 
def make_prediction(model, feature_columns: list):
    """
    Makes a prediction using user-provided information.
 
    Args:
        model: The trained model.
        feature_columns (list): The feature columns.
    """
    print("Please provide the following information:")
    user_input = []
    for feature in feature_columns:
        value = float(input(f"{feature}: "))
        user_input.append(value)
    
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    print(f"The prediction is: {'Rain' if prediction[0] == 1 else 'No Rain'}")
 
 
def select_data_in_date_interval(dataframe, start_date, end_date, target_column, date_format='%Y%m%d'):
    df = copy.deepcopy(dataframe)
 
    # Assume the first column is the date column
    date_column = df.columns[0]
 
    # Only keep the data and target column
    df = df[[date_column, target_column]]
 
    # Remove NaN rows
    df = df.dropna()
 
    # Ensure the date_column is in pandas Timestamp format
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
 
    # Convert start_date and end_date to pandas Timestamp
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
 
    # Filter the DataFrame for the specified date range
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
 
    # Return the filtered dates and target column
    return filtered_df[date_column], filtered_df[target_column]
 
# Placeholder for Average
def calculate_average(filtered_values):
    if len(filtered_values) == 0:
        return None  # Return None or handle empty data case as needed
    
    total_sum = sum(filtered_values)
    count = len(filtered_values)
    
    average = total_sum / count
    return average
 
# Placeholder for Standard Deviation
 
# Placeholder for Range
 
# Placeholder for Standard Median

# a small change locally (wasn't meant to push sorry Abbie)
def median(x: list):

    # sort the list into ascending order
    x.sort()

    if len(x) % 2 == 0:
        return (x[len(x)//2-1] + x[len(x) // 2]) / 2

    if len(x) % 2 == 1:
        return x[len(x)//2]

# Placeholder for Variance
 
 
def main():
    """
    Main function to execute the script.
    """
 
    
    # Read dataset
    dataset = read_dataset('london_weather.csv')
 
    dataset = clean_data(dataset, ['mean_temp','pressure','wind_speed'],
                                                 'precipitation')
    
    # Train model
    model, feature_columns, target = train_model(dataset)
    
    # Plot data
    # plot_data(dataset)
    
    # Make prediction
 
    make_prediction(model, feature_columns)
 
 
if __name__ == "__main__":
    main()