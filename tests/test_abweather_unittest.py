import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from abweather import read_dataset, clean_data, train_model, calculate_average, calculate_variance, calculate_range

class TestWeatherPrediction(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset for testing
        self.data = {
            'temperature': np.random.uniform(15, 35, 100),
            'humidity': np.random.uniform(40, 90, 100),
            'wind_speed': np.random.uniform(0, 20, 100),
            'precipitation': np.random.randint(0, 2, 100)
        }
        self.df = pd.DataFrame(self.data)

        # Save the DataFrame to a CSV file for testing
        self.test_file = 'test_weather_data.csv'
        self.df.to_csv(self.test_file, index=False)

    def test_read_dataset(self):
        result = read_dataset(self.test_file)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)  # Check if the dimensions match

    def test_clean_data(self):
        # Test the data cleaning process
        cleaned_df = clean_data(self.df, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
        self.assertIn('temperature', cleaned_df.columns)
        self.assertIn('humidity', cleaned_df.columns)
        self.assertNotIn('NaN', cleaned_df)  # Ensure no NaN values remain
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)  # Ensure no NaN values

    def test_train_model(self):
        # Test the training of the model
        cleaned_df = clean_data(self.df, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
        model, features, target = train_model(cleaned_df)
        self.assertIsInstance(model, RandomForestRegressor)  # Check if the model is a RandomForestRegressor
        self.assertGreater(len(features), 0)  # Check if features were returned
        self.assertEqual(cleaned_df.iloc[:, -1].values.all(), target.values.all())  # Target should be the last column

    def test_calculate_average(self):
        # Test the calculate_average function
        values = [1, 2, 3, 4, 5]
        result = calculate_average(values)
        self.assertEqual(result, 3.0)

        empty_values = []
        result_empty = calculate_average(empty_values)
        self.assertIsNone(result_empty)

    def test_calculate_variance(self):
        self.assertAlmostEqual(calculate_variance([-1.0,1.0]),1)
    
    def test_range(self):
        """
        Function that tests whether the range gives the expected output when inputting valid lists, or the expected exceptions.
        """

        my_list = [1, 2, 3, 4]
        self.assertEqual(calculate_range(my_list), 3)

        # testing ValueError raised for empty lists
        self.assertRaises(ValueError, lambda: calculate_range([]))
        # testing TypeError raised for wrong types of lists
        self.assertRaises(TypeError, lambda: calculate_range(["Denmark", "California"]))

    def tearDown(self):
        # Clean up the test CSV file after each test
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main()
