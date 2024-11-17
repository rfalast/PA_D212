import unittest
import pandas as pd

from pandas import DataFrame
from model.BaseModel import BaseModel


# test cases for BaseUtil
class test_BaseModel(unittest.TestCase):

    # constants
    ARGUMENT_VALID = "valid"
    ARGUMENT_NOT_VALID = None
    DF_MOCK_UP = pd.DataFrame({"foo_id": [1, 2, 3, 4, 5]})

    # test not_none() method
    def test_not_none(self):
        # create an instance of BaseUtil
        base = BaseModel()

        # validate that None argument returns false
        self.assertEqual(base.not_none(None), BaseModel.NONE_VALUE)

    # test is_valid() method on negative paths
    def test_is_valid_negative(self):
        # create an instance of BaseModel
        base = BaseModel()

        # run assertions

        # make sure we handle None, None
        self.assertFalse(base.is_valid(self.ARGUMENT_NOT_VALID, None))
        # make sure we handle String as a single argument
        self.assertFalse(base.is_valid(self.ARGUMENT_VALID))
        # make sure we handle String, None
        self.assertFalse(base.is_valid(self.ARGUMENT_VALID, None))

    # happy paths for is_valid() method
    def test_is_valid(self):
        # create an instance of BaseModel
        base = BaseModel()

        # make sure we handle String
        self.assertTrue(base.is_valid(self.ARGUMENT_VALID, str))

        # make sure we handle int as a single argument
        self.assertTrue(base.is_valid(5, int))

        # make sure we handle float as a single argument
        self.assertTrue(base.is_valid(5.515, float))

        # make sure we handle a DataFrame
        self.assertTrue(base.is_valid(self.DF_MOCK_UP, DataFrame))