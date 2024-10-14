

from numbers import Number
from typing import List
# annoying merge

def calculate_variance(data: List[Number]) -> Number:
    if not isinstance(data, List):
        raise TypeError("Should be a list of numbers")
    if not len(data) > 0:
        raise ValueError("Should be more than zero points.")
    mean = sum(data)/len(data)
    return sum((x - mean)**2 for x in data)/len(data)