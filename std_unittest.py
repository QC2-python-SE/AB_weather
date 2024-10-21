import unittest
from abweather import calculate_standard_deviation

class TestStandardDeviation(unittest.TestCase):
    def test_calculate_standard_deviation(self):
        values = [2, 4, 4, 4, 5, 5, 7, 9] 
        result = calculate_standard_deviation(values)
        self.assertEqual(result, 2)


if __name__ == '__main__':
    unittest.main()