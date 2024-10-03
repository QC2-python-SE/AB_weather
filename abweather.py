# -*- coding: utf-8 -*-
"""
Weather Prediction Script
=========================

This script demonstrates good practices in programming, including:
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
    # I add this line here
    """
    Reads a dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)

def test_function(arguments):
    """THIS IS A TEST FUNCTION
    """
    return arguments

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


def select_data_in_date_interval_old(dataframe, start_date, end_date, target_column, date_format = '%Y%m%d'):
   
    df = copy.deepcopy(dataframe)
    
    # Assume the first column is the date column
    date_column = df.columns[0]
    
    # Only keep the data and target column
    df = df[[date_column,target_column]]
    
    #Remove NaN rows
    df = df.dropna()

    # Ensure the date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], format=date_format).dt.date

    # Now your start_date and end_date should also be in datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter the DataFrame for the specified date range
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

    # Calculate and return the average of the target column
    return filtered_df[date_column], filtered_df[target_column]

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

def calculate_average(filtered_values):
    if len(filtered_values) == 0:
        return None  # Return None or handle empty data case as needed
    
    total_sum = sum(filtered_values)
    count = len(filtered_values)
    
    average = total_sum / count
    return average


def main():
    """
    Main function to execute the script.
    """
    # Mock dataset creation for demonstration purposes
    data = {
        'temperature': np.random.uniform(15, 35, 1000),
        'humidity': np.random.uniform(40, 90, 1000),
        'wind_speed': np.random.uniform(0, 20, 1000),
        'precipitation': np.random.randint(0, 2, 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV for demonstration purposes
    df.to_csv('weather_data.csv', index=False)
    
    # Read dataset
    dataset = read_dataset('weather_data.csv')

    dataset = clean_data(dataset, ['temperature','humidity','wind_speed'],
                                                 'precipitation')
    
    # Train model
    model, feature_columns, target = train_model(dataset)
    
    # Plot data
    # plot_data(dataset)
    
    # Make prediction

    make_prediction(model, feature_columns)

if __name__ == "__main__":
    main()

# Let's create conflicts
# Group 1 best
# Group 2 is not best
# Group 3 is not best
# The teachers are the best