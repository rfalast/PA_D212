import unittest

import numpy
import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_float_dtype
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.Project_Assessment import Project_Assessment
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models.KNN_Model import KNN_Model
from model.constants.BasicConstants import TIMEZONE_DICTIONARY, TIMEZONE_COLUMN, D_212_CHURN, ANALYZE_DATASET_FULL, \
    MT_KNN_CLASSIFICATION
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Detector import Detector
from model.constants.DatasetConstants import BOOL_VALUE_KEY, FLOAT64_COLUMN_KEY, UNIQUE_COLUMN_LIST_KEY
from model.converters.Converter import Converter
from util.CSV_loader import CSV_Loader
from util.ExcelManager import ExcelManager


# test class for Converter
class test_Converter(unittest.TestCase):
    # constants
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    VALID_NORM_PATH = "../../../resources/Output/test_normalized.xlsx"
    VALID_BOOLEAN_COLUMN = "Churn"
    INVALID_BOOLEAN_COLUMN = "Foo"
    VALID_NAN_COLUMN = "Techie"
    INVALID_NAN_COLUMN = "Foo"
    VALID_FLOAT_WITH_NAN_1 = "Income"
    VALID_FLOAT_NO_NAN_1 = "Outage_sec_perweek"
    VALID_FLOAT_NO_NAN_2 = "Tenure"
    VALID_INT_NO_NAN_1 = "Population"
    VALID_INT_NO_NAN_2 = "Contacts"
    VALID_INT_NAN_1 = "Age"
    VALID_INT_NAN_2 = "Children"
    UN_NAMED_COLUMN = "Unnamed: 0"

    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"

    VALID_FIELD_DICT = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                        "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                        "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    VALID_COLUMN_DROP_LIST_1 = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction',
                                'State', 'UID', 'County', 'Job', 'City']

    VALID_NORM_INT_DF_COLUMNS = ['Unnamed: 0', 'CaseOrder', 'Zip', 'Population', 'Children', 'Age', 'Email'
                                 'Contacts', 'Yearly_equip_failure', 'item1', 'item2', 'item3', 'item4',
                                 'item5', 'item6', 'item7', 'item8']

    VALID_NORM_FLT_DF_COLUMNS = ['Lat', 'Lng', 'Income', 'Outage_sec_perweek',
                                 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']

    OVERRIDE_PATH = "../../../resources/Output"

    CHURN_KEY = D_212_CHURN

    # test the convert_to_boolean() method and the way it handles errors.
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def test_convert_column_to_boolean_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run analyze columns
        dsa.analyze_columns()

        # create the converter instance
        conv = Converter()

        # we're going to pass in None, None
        with self.assertRaises(SyntaxError) as context:
            # None arguments for Converter()
            conv.convert_column_to_boolean(None, None)

            # validate the error message.
            self.assertTrue("The incoming argument was None or incorrect type." in context)

        # we're going to pass in str, None
        with self.assertRaises(SyntaxError) as context:
            # None argument for Converter()
            conv.convert_column_to_boolean("Foo", None)

            # validate the error message.
            self.assertTrue("The incoming argument was None or incorrect type." in context)

        # we're going to pass in Dataframe, None
        with self.assertRaises(SyntaxError) as context:
            # None argument for Converter()
            conv.convert_column_to_boolean(dsa.the_df, None)

            # validate the error message.
            self.assertTrue("The incoming argument was None or incorrect type." in context)

        # we're going to pass in Dataframe, bad string
        with self.assertRaises(SyntaxError) as context:
            # None argument for Converter()
            conv.convert_column_to_boolean(dsa.the_df, self.INVALID_BOOLEAN_COLUMN)

            # validate the error message.
            self.assertTrue("The incoming column name was not present in the DataFrame." in context)

    # test the convert_column_to_boolean() method
    def test_convert_column_to_boolean(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run analyze columns
        dsa.analyze_columns()

        # create the converter instance
        conv = Converter()

        # invoke the method
        returned_df = conv.convert_column_to_boolean(dsa.the_df, self.VALID_BOOLEAN_COLUMN)

        # run assertions
        self.assertIsNotNone(returned_df)
        self.assertIsInstance(returned_df, DataFrame)
        self.assertTrue(returned_df[self.VALID_BOOLEAN_COLUMN].dtype, bool)

    # negative tests for convert_to_boolean()
    def test_convert_to_boolean_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run analyze columns
        dsa.analyze_columns()

        # create the converter instance
        conv = Converter()

        # run the method.
        dsa.the_df = conv.convert_to_boolean(dsa.the_df, list(dsa.storage[BOOL_VALUE_KEY].keys()))

        # get the list of boolean values
        bool_list = list(dsa.storage[BOOL_VALUE_KEY].keys())

        # loop over all the boolean values
        for next_column in bool_list:
            # run assertions
            self.assertIsNotNone(bool_list)
            self.assertIsInstance(bool_list, list)
            self.assertTrue(next_column in dsa.the_df)
            self.assertTrue(type(next_column in dsa.the_df) == bool)

    # test the replace_nan_with_value() method
    def test_replace_nan_with_value(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run analyze columns
        dsa.analyze_columns()

        # create the converter instance
        conv = Converter()

        # invoke the method
        returned_df = conv.replace_nan_with_value(dsa.the_df, self.VALID_NAN_COLUMN)

        # run assertions
        self.assertIsNotNone(returned_df)
        self.assertIsInstance(returned_df, DataFrame)
        self.assertEqual(returned_df[self.VALID_NAN_COLUMN].isna().sum(), 0)

    # test the replace_nan_with_value() method and the way it handles errors.
    def test_replace_nan_with_value_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run analyze columns
        dsa.analyze_columns()

        # create the converter instance
        conv = Converter()

        # we're going to pass in None, None
        with self.assertRaises(SyntaxError) as context:
            # None arguments for Converter()
            conv.replace_nan_with_value(None, None)

            # validate the error message.
            self.assertTrue("The incoming argument was None or incorrect type." in context)

        # we're going to pass in str, None
        with self.assertRaises(SyntaxError) as context:
            # None argument for Converter()
            conv.replace_nan_with_value("Foo", None)

            # validate the error message.
            self.assertTrue("The incoming argument was None or incorrect type." in context)

        # we're going to pass in Dataframe, None
        with self.assertRaises(SyntaxError) as context:
            # None argument for Converter()
            conv.replace_nan_with_value(dsa.the_df, None)

            # validate the error message.
            self.assertTrue("The incoming argument was None or incorrect type." in context)

        # we're going to pass in Dataframe, bad string
        with self.assertRaises(SyntaxError) as context:
            # None argument for Converter()
            conv.replace_nan_with_value(dsa.the_df, self.INVALID_NAN_COLUMN)

            # validate the error message.
            self.assertTrue("The incoming column name was not present in the DataFrame." in context)

    # test method for clean_floats_that_should_be_int
    def test_clean_floats_that_should_be_int(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh model --- WE HAVE TO RUN TEST IN THIS METHOD TO EXPLICITLY TEST clean_floats_that_should_be_int()
        dsa.refresh_model()

        # run conversion --- WE HAVE TO RUN TEST IN THIS METHOD TO EXPLICITLY TEST clean_floats_that_should_be_int()
        dsa.run_conversion_for_boolean()

        # refresh model --- WE HAVE TO RUN TEST IN THIS METHOD TO EXPLICITLY TEST clean_floats_that_should_be_int()
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # create the converter
        the_converter = Converter()

        # run function
        the_converter.clean_floats_that_should_be_int(dsa.the_df, dsa.storage[FLOAT64_COLUMN_KEY])

        # run assertions
        self.assertEqual(dsa.the_df['Children'].dtype, np.int64)

    # negative tests for normalize_series() method
    def test_normalize_series_negative(self):
        # create a converter
        the_converter = Converter()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.normalize_series(None)

            # validate the error message.
            self.assertTrue("The incoming Series was None or incorrect type." in context)

        # we're going to pass in int
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.normalize_series(5)

            # validate the error message.
            self.assertTrue("The incoming Series was None or incorrect type." in context)

    # test method for normalize_series() method
    def test_normalize_series(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run setup
        dsa.run_complete_setup()

        # create a converter
        the_converter = Converter()

        # invoke the method on VALID_FLOAT_NO_NAN_1
        the_field = self.VALID_FLOAT_NO_NAN_1
        the_series = the_converter.normalize_series(dsa.the_df[the_field])

        # run basic assertions for the_field
        self.assertIsNotNone(the_series)
        self.assertIsInstance(the_series, Series)
        # make sure we know what the type is
        self.assertTrue(is_float_dtype(dsa.the_df[the_field]))
        self.assertTrue(is_float_dtype(the_series))
        # run dimensionality tests
        self.assertEqual(len(the_series), 10000)
        self.assertEqual(len(the_series), len(dsa.the_df[the_field]))
        # make sure we don't have any na's
        self.assertEqual(the_series.isna().sum(), 0)
        self.assertEqual(dsa.the_df[the_field].isna().sum(), 0)

        # invoke the method on VALID_FLOAT_NO_NAN_2
        the_field = self.VALID_FLOAT_NO_NAN_2
        the_series = the_converter.normalize_series(dsa.the_df[the_field])

        # run basic assertions for the_field
        self.assertIsNotNone(the_series)
        self.assertIsInstance(the_series, Series)
        # make sure we know what the type is
        self.assertTrue(is_float_dtype(dsa.the_df[the_field]))
        self.assertTrue(is_float_dtype(the_series))
        # run dimensionality tests
        self.assertEqual(len(the_series), 10000)
        self.assertEqual(len(the_series), len(dsa.the_df[the_field]))
        # make sure we don't have any na's
        self.assertEqual(the_series.isna().sum(), 0)

        # invoke the method on VALID_INT_NO_NAN_1
        the_field = self.VALID_INT_NO_NAN_1
        the_series = the_converter.normalize_series(dsa.the_df[the_field])

        # run basic assertions for the_field
        self.assertIsNotNone(the_series)
        self.assertIsInstance(the_series, Series)
        # make sure we know what the type is
        self.assertTrue(dsa.the_df[the_field].dtype, np.int64)
        self.assertTrue(is_float_dtype(the_series))
        # run dimensionality tests
        self.assertEqual(len(the_series), 10000)
        self.assertEqual(len(the_series), len(dsa.the_df[the_field]))
        # make sure we don't have any na's
        self.assertEqual(the_series.isna().sum(), 0)
        self.assertEqual(dsa.the_df[the_field].isna().sum(), 0)

        # invoke the method on VALID_INT_NO_NAN_2
        the_field = self.VALID_INT_NO_NAN_2
        the_series = the_converter.normalize_series(dsa.the_df[the_field])

        # run basic assertions for the_field
        self.assertIsNotNone(the_series)
        self.assertIsInstance(the_series, Series)
        # make sure we know what the type is
        self.assertTrue(dsa.the_df[the_field].dtype, np.int64)
        self.assertTrue(is_float_dtype(the_series))
        # run dimensionality tests
        self.assertEqual(len(the_series), 10000)
        self.assertEqual(len(the_series), len(dsa.the_df[the_field]))
        # make sure we don't have any na's
        self.assertEqual(the_series.isna().sum(), 0)
        self.assertEqual(dsa.the_df[the_field].isna().sum(), 0)

        # invoke the method on VALID_INT_NAN_1
        the_field = self.VALID_INT_NAN_1
        the_series = the_converter.normalize_series(dsa.the_df[the_field])

        # run basic assertions for the_field
        self.assertIsNotNone(the_series)
        self.assertIsInstance(the_series, Series)
        # make sure we know what the type is
        self.assertEqual(dsa.the_df[the_field].dtype, np.int64)
        self.assertTrue(is_float_dtype(the_series))
        # run dimensionality tests
        self.assertEqual(len(the_series), 10000)
        self.assertEqual(len(the_series), len(dsa.the_df[the_field]))
        # make sure we don't have any na's
        self.assertEqual(the_series.isna().sum(), 0)

        # invoke the method on VALID_INT_NAN_2
        the_field = self.VALID_INT_NAN_2
        the_series = the_converter.normalize_series(dsa.the_df[the_field])

        # run basic assertions for the_field
        self.assertIsNotNone(the_series)
        self.assertIsInstance(the_series, Series)
        # make sure we know what the type is
        self.assertEqual(dsa.the_df[the_field].dtype, np.int64)
        self.assertTrue(is_float_dtype(the_series))
        # run dimensionality tests
        self.assertEqual(len(the_series), 10000)
        self.assertEqual(len(the_series), len(dsa.the_df[the_field]))
        # make sure we don't have any na's in series
        self.assertEqual(the_series.isna().sum(), 0)

    # negative tests for get_normalized_df() method
    def test_get_normalized_df_negative(self):
        # create a converter
        the_converter = Converter()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.get_normalized_df(None)

            # validate the error message.
            self.assertTrue("The argument was None or the wrong type." in context)

    # test for get_normalized_df() method
    def test_get_normalized_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run setup
        dsa.run_complete_setup()

        # create a converter
        the_converter = Converter()

        # invoke the method
        the_norm_df = the_converter.get_normalized_df(dsa.the_df)

        # run assertions
        self.assertIsNotNone(the_norm_df)
        self.assertIsInstance(the_norm_df, DataFrame)
        self.assertEqual(len(the_norm_df.columns), 23)

        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # create the workbook
        excel_manager.create_workbook(self.VALID_NORM_PATH)
        excel_manager.open_workbook(self.VALID_NORM_PATH)
        excel_manager.write_df_into_wb_tab(the_norm_df,
                                           "normalized data",
                                           excel_manager.wb_storage[self.VALID_NORM_PATH])

        excel_manager.close_workbook(self.VALID_NORM_PATH)

    # negative tests for clean_columns_with_blank_name() method
    def test_clean_columns_with_blank_name_negative(self):
        # create a converter
        the_converter = Converter()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.clean_columns_with_blank_name(None)

            # validate the error message.
            self.assertTrue("The incoming Series was None or incorrect type." in context)

    # tests for clean_columns_with_blank_name()
    def test_clean_columns_with_blank_name(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh model --- WE HAVE TO RUN TEST IN THIS METHOD TO EXPLICITLY TEST clean_floats_that_should_be_int()
        dsa.refresh_model()

        # run conversion --- WE HAVE TO RUN TEST IN THIS METHOD TO EXPLICITLY TEST clean_floats_that_should_be_int()
        dsa.run_conversion_for_boolean()

        # refresh model --- WE HAVE TO RUN TEST IN THIS METHOD TO EXPLICITLY TEST clean_floats_that_should_be_int()
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # run int conversion, this function refreshes the model.
        dsa.run_conversion_from_float_2int()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # run assertions to make sure the Unamed column is not there
        self.assertFalse(self.UN_NAMED_COLUMN in dsa.the_df)

        # create the converter
        converter = Converter()

        # run method
        converter.clean_columns_with_blank_name(dsa.the_df)

        # make sure the method removes the column
        self.assertFalse(self.UN_NAMED_COLUMN in dsa.the_df)

    # negative tests for fill_nan_values_for_numerical_series() method
    def test_fill_nan_values_for_numerical_series_negative(self):
        # create a converter
        the_converter = Converter()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.fill_nan_values_for_numerical_series(None)

            # validate the error message.
            self.assertTrue("The incoming Series was None or incorrect type." in context)

        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run setup
        dsa.run_complete_setup()

        # we're going to pass in dsa.
        with self.assertRaises(TypeError) as context:
            # invoke method
            the_converter.fill_nan_values_for_numerical_series(dsa.the_df[self.VALID_BOOLEAN_COLUMN])

            # validate the error message.
            self.assertTrue("The incoming Series was not numeric." in context)

    # test method for fill_nan_values_for_numerical_series() method
    def test_fill_nan_values_for_numerical_series(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.refresh_model()

        # run boolean conversion
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # run int conversion, this function refreshes the model.
        dsa.run_conversion_from_float_2int()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # create a detector
        detector = Detector()

        # create a converter
        converter = Converter()

        # retrieve a INT64 series that is known to have NA
        the_series = dsa.the_df[self.VALID_INT_NAN_1]

        # verify that it would pass the test
        self.assertFalse(detector.detect_int_with_na(the_series))

        # clean the series
        the_series = converter.fill_nan_values_for_numerical_series(the_series)

        # verify that the field is clean
        self.assertFalse(detector.detect_int_with_na(the_series))

    # negative tests for fill_nan_values_for_numerical_df()
    def test_fill_nan_values_for_numerical_df_negative(self):
        # create a converter
        the_converter = Converter()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.fill_nan_values_for_numerical_df(None)

            # validate the error message.
            self.assertTrue("The Dataframe argument was None or the wrong type." in context)

        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.refresh_model()

        # run boolean conversion
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # run int conversion, this function refreshes the model.
        dsa.run_conversion_from_float_2int()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # get reference to column name
        col_name_1 = self.VALID_BOOLEAN_COLUMN

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # pass in a known boolean Series
            dsa.the_df[col_name_1] = the_converter.fill_nan_values_for_numerical_df(col_name_1)

            # validate the error message.
            self.assertTrue("The Dataframe argument was None or the wrong type." in context)

    # test method for fill_nan_values_for_numerical_df()
    def test_fill_nan_values_for_numerical_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # run boolean conversion
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # run int conversion, this function refreshes the model.
        dsa.run_conversion_from_float_2int()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # create converter
        converter = Converter()

        # get series name for known NaN columns
        the_column_1 = self.VALID_INT_NAN_1
        the_column_2 = self.VALID_FLOAT_WITH_NAN_1

        # get initial count of NaN in column
        nan_count_1 = dsa.the_df[the_column_1].isna().sum()
        nan_count_2 = dsa.the_df[the_column_2].isna().sum()

        # run function
        dsa.the_df = converter.fill_nan_values_for_numerical_df(dsa.the_df)

        # get count of NaN in column after call
        nan_count_1_after = dsa.the_df[the_column_1].isna().sum()
        nan_count_2_after = dsa.the_df[the_column_2].isna().sum()

        # run assertions
        self.assertEqual(nan_count_1, 0)
        self.assertEqual(nan_count_2, 0)
        self.assertEqual(nan_count_1, nan_count_1_after)
        self.assertEqual(nan_count_2, nan_count_2_after)
        self.assertEqual(nan_count_1, nan_count_1_after)
        self.assertEqual(nan_count_2, nan_count_2_after)
        self.assertTrue(nan_count_1_after == 0)
        self.assertTrue(nan_count_2_after == 0)

    # negative tests for clean_timezone()
    def test_clean_timezone_negative(self):
        # create a converter
        the_converter = Converter()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.clean_timezone(None)

            # validate the error message.
            self.assertTrue("The Series argument was None or the wrong type." in context)

        # we're going to pass in str
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_converter.clean_timezone("foo")

            # validate the error message.
            self.assertTrue("The Series argument was None or the wrong type." in context)

    # test method for clean_timezone() method
    def test_clean_timezone(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # run boolean conversion
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # run int conversion, this function refreshes the model.
        dsa.run_conversion_from_float_2int()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # create converter
        converter = Converter()

        # get the Series for timezones
        the_series = dsa.the_df[TIMEZONE_COLUMN]

        # invoke the function
        converter.clean_timezone(the_series)

        # get the Series for timezone again
        unique_values = dsa.the_df[TIMEZONE_COLUMN].unique()

        # loop through the keys in the self.TIMEZONE_DICTIONARY
        for bad_time_zone in TIMEZONE_DICTIONARY.keys():
            # get the good timezone.  This line looks weird, but is correct.
            good_time_zone = TIMEZONE_DICTIONARY[bad_time_zone]

            # make sure the time zone lists are different
            self.assertNotEqual(bad_time_zone, good_time_zone)

            # make sure bad_time_zone's are NOT present
            self.assertFalse(bad_time_zone in unique_values)

            # make sure good_time_zone's present
            self.assertTrue(good_time_zone in unique_values)

    # test method for remove_duplicate_rows() using test DataFrame
    def test_remove_duplicate_rows_test_df(self):
        # create a detector
        detector = Detector()

        # create a Converter
        converter = Converter()

        # initialize list of lists
        data = [['rob', 10], ['andree', 39], ['sophia', 21], ['garrett', 19], ['colby', 17]]

        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns=['Name', 'Age'])

        # run validations of the dataframe
        self.assertIsNotNone(df)
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df), 5)
        self.assertEqual(len(df.columns), 2)
        self.assertEqual(df.columns[0], 'Name')
        self.assertEqual(df.columns[1], 'Age')
        self.assertEqual(df.iloc[0, 0], 'rob')
        self.assertEqual(df.iloc[0, 1], 10)
        self.assertEqual(df.iloc[1, 0], 'andree')
        self.assertEqual(df.iloc[1, 1], 39)
        self.assertEqual(df.iloc[2, 0], 'sophia')
        self.assertEqual(df.iloc[2, 1], 21)
        self.assertEqual(df.iloc[3, 0], 'garrett')
        self.assertEqual(df.iloc[3, 1], 19)
        self.assertEqual(df.iloc[4, 0], 'colby')
        self.assertEqual(df.iloc[4, 1], 17)

        # get the duplicated rows
        duplicate_rows_df = df[df.duplicated()]

        # run assertions that we have no duplicates
        self.assertIsNotNone(duplicate_rows_df)
        self.assertIsInstance(duplicate_rows_df, DataFrame)
        self.assertEqual(len(duplicate_rows_df), 0)
        self.assertEqual(len(duplicate_rows_df.columns), 2)
        self.assertEqual(duplicate_rows_df.columns[0], 'Name')
        self.assertEqual(duplicate_rows_df.columns[1], 'Age')

        # invoke the method on df
        self.assertFalse(detector.detect_if_dataframe_has_duplicates(df))

        # add two duplicate rows to df
        df.loc[len(df.index)] = ['rob', 10]
        df.loc[len(df.index)] = ['colby', 17]

        # make sure the dataframe looks like we expect
        self.assertEqual(len(df), 7)
        self.assertEqual(len(df.columns), 2)
        self.assertEqual(df.columns[0], 'Name')
        self.assertEqual(df.columns[1], 'Age')
        self.assertEqual(df.iloc[0, 0], 'rob')
        self.assertEqual(df.iloc[0, 1], 10)
        self.assertEqual(df.iloc[1, 0], 'andree')
        self.assertEqual(df.iloc[1, 1], 39)
        self.assertEqual(df.iloc[2, 0], 'sophia')
        self.assertEqual(df.iloc[2, 1], 21)
        self.assertEqual(df.iloc[3, 0], 'garrett')
        self.assertEqual(df.iloc[3, 1], 19)
        self.assertEqual(df.iloc[4, 0], 'colby')
        self.assertEqual(df.iloc[4, 1], 17)
        self.assertEqual(df.iloc[5, 0], 'rob')
        self.assertEqual(df.iloc[5, 1], 10)
        self.assertEqual(df.iloc[6, 0], 'colby')
        self.assertEqual(df.iloc[6, 1], 17)

        # get the duplicated rows
        duplicate_rows_df = df[df.duplicated()]

        # run assertions that we have no duplicates
        self.assertIsNotNone(duplicate_rows_df)
        self.assertIsInstance(duplicate_rows_df, DataFrame)
        self.assertEqual(len(duplicate_rows_df), 2)
        self.assertEqual(len(duplicate_rows_df.columns), 2)
        self.assertEqual(duplicate_rows_df.columns[0], 'Name')
        self.assertEqual(duplicate_rows_df.columns[1], 'Age')

        # make sure the duplicates are what we think they should be.
        self.assertEqual(duplicate_rows_df.iloc[0, 0], 'rob')
        self.assertEqual(duplicate_rows_df.iloc[0, 1], 10)
        self.assertEqual(duplicate_rows_df.iloc[1, 0], 'colby')
        self.assertEqual(duplicate_rows_df.iloc[1, 1], 17)

        # invoke the method again on df
        self.assertTrue(detector.detect_if_dataframe_has_duplicates(df))

        # invoke the method remove_duplicate_rows()
        df = converter.remove_duplicate_rows(df)

        # run assertions
        self.assertIsNotNone(df)
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df), 5)
        self.assertEqual(len(df.columns), 2)
        self.assertEqual(df.columns[0], 'Name')
        self.assertEqual(df.columns[1], 'Age')
        self.assertEqual(df.iloc[0, 0], 'rob')
        self.assertEqual(df.iloc[0, 1], 10)
        self.assertEqual(df.iloc[1, 0], 'andree')
        self.assertEqual(df.iloc[1, 1], 39)
        self.assertEqual(df.iloc[2, 0], 'sophia')
        self.assertEqual(df.iloc[2, 1], 21)
        self.assertEqual(df.iloc[3, 0], 'garrett')
        self.assertEqual(df.iloc[3, 1], 19)
        self.assertEqual(df.iloc[4, 0], 'colby')
        self.assertEqual(df.iloc[4, 1], 17)

    # test method for remove_duplicate_rows() using actual data
    def test_remove_duplicate_rows_actual(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # variable declarations
        converter = Converter()

        # refresh the model
        dsa.refresh_model()

        # extract boolean fields
        dsa.extract_boolean()

        # extract int fields
        dsa.extract_int_fields()

        # check if we need to clean NA values
        dsa.remove_na_values()

        # check if we need to set an index
        if not dsa.detector.check_if_df_has_named_indexed(dsa.the_df):
            # get the new index
            new_index = dsa.detector.detect_index(dsa.the_df, dsa.storage[UNIQUE_COLUMN_LIST_KEY])

            # set the index
            dsa.the_df.set_index(new_index, inplace=True)

        # clean time zone values
        converter.clean_timezone(dsa.the_df[TIMEZONE_COLUMN])

        # this is where we check that we do not have duplicates
        self.assertFalse(dsa.detector.detect_if_dataframe_has_duplicates(dsa.the_df))

        # verify the size prior to calling method
        self.assertEqual(len(dsa.the_df), 10000)
        self.assertEqual(len(dsa.the_df.columns), 49)

        # invoke the method
        dsa.the_df = converter.remove_duplicate_rows(dsa.the_df)

        # verify the size
        self.assertEqual(len(dsa.the_df), 10000)
        self.assertEqual(len(dsa.the_df.columns), 49)

    # test method for convert_array_to_dataframe()
    def test_convert_array_to_dataframe(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_1)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get a list of original column names
        the_f_df_train_columns = the_f_df_train.columns.to_list()

        # scale the_f_df_train, the_f_df_test.  This converts the the_f_df_train variable from a dataframe to a
        # numpy.ndarry
        the_f_df_train_np = the_scalar.fit_transform(the_f_df_train)

        # run assertions
        self.assertIsInstance(the_f_df_train_np, numpy.ndarray)
        self.assertIsInstance(the_f_df_train_columns, list)

        # invoke method
        the_df = Converter.convert_array_to_dataframe(the_array=the_f_df_train_np, the_columns=the_f_df_train_columns)

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertIsInstance(the_f_df_train_np, np.ndarray)

        # loop through all the elements in the_f_df_train_columns to make sure every element in
        # the_f_df_train_columns is in the_df
        for the_column in the_f_df_train_columns:
            self.assertTrue(the_column in the_df.columns)

        # define a counter variable
        the_counter = 0

        # validate the order of columns matches
        for the_column in the_f_df_train_columns:
            # run assertion
            self.assertEqual(the_column, the_df.iloc[:, the_counter].name)

            # increment counter
            the_counter = the_counter + 1

        # validate the length of both dataframes
        self.assertEqual(len(the_df), len(the_f_df_train))

        # we can't check anything else because the the_df is normalized and the_f_df_train is raw.
