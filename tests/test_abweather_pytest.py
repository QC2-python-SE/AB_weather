import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from abweather import read_dataset, clean_data, train_model, calculate_average

@pytest.fixture
def mock_data():
    """
    Creates a mock dataset for testing.
    """
    data = {
        'temperature': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(40, 90, 100),
        'wind_speed': np.random.uniform(0, 20, 100),
        'precipitation': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    return df

def test_read_dataset(mock_data, tmpdir):
    # Save the DataFrame to a temporary CSV file
    temp_file = tmpdir.join("test_weather_data.csv")
    mock_data.to_csv(temp_file, index=False)

    # Read the dataset
    result = read_dataset(temp_file)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == mock_data.shape  # Check if the dimensions match

def test_clean_data(mock_data):
    # Test the data cleaning process
    cleaned_df = clean_data(mock_data, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
    
    assert 'temperature' in cleaned_df.columns
    assert 'humidity' in cleaned_df.columns
    assert cleaned_df.isnull().sum().sum() == 0  # Ensure no NaN values
    assert cleaned_df.duplicated().sum() == 0  # Ensure no duplicates

def test_train_model(mock_data):
    # Clean data before training
    cleaned_df = clean_data(mock_data, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
    
    # Train the model
    model, features, target = train_model(cleaned_df)
    
    assert isinstance(model, RandomForestRegressor)  # Check if the model is a RandomForestRegressor
    assert len(features) > 0  # Ensure features were returned
    assert (cleaned_df.iloc[:, -1].values == target.values).all()  # Ensure target matches the last column

def test_calculate_average():
    # Test calculate_average function
    values = [1, 2, 3, 4, 5]
    result = calculate_average(values)
    assert result == 3.0

    # Test for empty input
    empty_values = []
    result_empty = calculate_average(empty_values)
    assert result_empty is None
