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