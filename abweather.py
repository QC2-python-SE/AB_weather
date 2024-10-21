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
from numbers import Number
from typing import List

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
def calculate_standard_deviation(filtered_values):
    """
    This function calculates the standard deviation

    Args:
        filtered_values (list[int|float]) : a list of values

    Returns:
        int|float : standard deviation of a list
    """
    if len(filtered_values) == 0:
        return None  # Return None or handle empty data case as needed
    
    average = calculate_average(filtered_values)
    sum_squared_differences = sum([(value - average) ** 2 for value in filtered_values]) 
    variance = sum_squared_differences / len(filtered_values)
    standard_deviation = variance ** 0.5
    return standard_deviation
 
# Placeholder for Range
def calculate_range(filtered_values: list):
    '''
    
    Take values in a numrical list, calculate the range between extremal values.

    Args:
        filtered_values (list): A list of numbers.

    Returns:
        float: Range of the list.
        
    Raises:
        ValueError: If list is empty
        TypeError: If the list contains elements that are not float or int
    '''
    
    if not filtered_values:  # checks if list is empty
        raise ValueError("Arrr, ye can't find treasure in an empty chest, and ye can't find range in an empty list!")
    
    if (all(not isinstance(item, (int, float)) for item in filtered_values)):  # checks if the list is float or int
        raise TypeError("Ah, the range of letters is boundless, but the range of charactors is meaningless.")
    
    r_range = max(filtered_values) - min(filtered_values)

    return r_range
 
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
def calculate_variance(data: List[Number]) -> Number:
    """
    Calculate variance of data.
    
    Parameters
    ----------
    data : List[Number]
        List of numbers to calculate variance of.
    
    Returns
    -------
    Number
        Variance of data.
    """
    if not isinstance(data, List):
        # raise TypeError if data is not a list
        raise TypeError(f"data should be a list of numbers, not {type(data).__name__}")
    if not len(data) > 0:
        # raise ValueError if data is empty
        raise ValueError("data should contain more than zero points")
    mean = sum(data)/len(data)
    return sum((x - mean)**2 for x in data)/len(data)
 
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