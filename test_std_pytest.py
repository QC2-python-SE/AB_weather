import pytest
from abweather import calculate_standard_deviation
# Assuming calculate_standard_deviation and calculate_average functions are already defined

def test_standard_deviation():
    values = [2, 4, 4, 4, 5, 5, 7, 9]  
    result = calculate_standard_deviation(values)
    assert result == 2
