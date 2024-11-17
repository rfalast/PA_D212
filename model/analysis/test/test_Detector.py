import copy
import unittest
import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas.core.dtypes.common import is_float_dtype, is_object_dtype, is_bool_dtype
from model.Project_Assessment import Project_Assessment
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.Linear_Model_Result import Linear_Model_Result
from model.constants.BasicConstants import DEFAULT_INDEX_NAME, ANALYZE_DATASET_FULL, D_209_CHURN, MT_LINEAR_REGRESSION
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Detector import Detector
from model.constants.DatasetConstants import UNIQUE_COLUMN_LIST_KEY
from model.constants.ModelConstants import LM_FINAL_MODEL, LM_INITIAL_MODEL
from model.converters.Converter import Converter
from util.CSV_loader import CSV_Loader


class test_Detector(unittest.TestCase):
    # constants
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    VALID_INT_COLUMN_WITH_NA_1 = "Age"
    VALID_INT_COLUMN_WITH_NA_2 = "Children"
    VALID_INT_COLUMN_WITHOUT_NA_1 = "Population"
    VALID_INT_COLUMN_WITHOUT_NA_2 = "Email"
    VALID_FLOAT_COLUMN_1 = "Income"
    VALID_FLOAT_COLUMN_2 = "Tenure"
    VALID_FLOAT_COLUMN_3 = "Bandwidth_GB_Year"
    VALID_FLOAT_COLUMN_WITHOUT_NA_1 = "Outage_sec_perweek"
    VALID_FLOAT_COLUMN_WITHOUT_NA_2 = "MonthlyCharge"
    VALID_OBJECT_COLUMN_1 = "County"
    VALID_OBJECT_COLUMN_2 = "State"
    VALID_BOOLEAN_COLUMN_1 = "Churn"
    VALID_BOOLEAN_COLUMN_2 = "Tablet"

    VALID_INT_COLUMN_UNIQUE_1 = "CaseOrder"
    VALID_INT_COLUMN_UNIQUE_2 = "Unnamed: 0"
    VALID_OBJ_COLUMN_UNIQUE_1 = "Customer_id"

    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    VALID_FIELD_DICT = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                        "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                        "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    VALID_COLUMN_DROP_LIST_2 = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction',
                                'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_209_CHURN

    # test the init() method
    def test_init(self):
        # instantiate the Detector
        detector = Detector()

        # run assertions
        self.assertIsNotNone(detector)
        self.assertIsInstance(detector, Detector)

    # negative tests for detect_when_float_is_int()
    def test_detect_when_float_is_int_negative(self):
        # create a detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_when_float_is_int(None)

            # validate the error message.
            self.assertTrue("The argument was None or the wrong type." in context)

        # make sure we get a SyntaxError for str
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_when_float_is_int("foo")

            # validate the error message.
            self.assertTrue("The argument was None or the wrong type." in context)

    # test method for detect_when_float_is_int()
    def test_detect_when_float_is_int(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # in order for this test case to work, we need to make sure we do not call
        # dsa.run_complete_setup() explicitly.  Thus, we need to step by step construct
        # the scenario where we get to when detect_when_float_is_int() can be called.

        # run refresh model
        dsa.refresh_model()

        # run the boolean conversions
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # now, instantiate the Detector
        detector = Detector()

        # define series we wil need for the tests
        the_series_1 = dsa.the_df[self.VALID_INT_COLUMN_WITH_NA_1]
        the_series_2 = dsa.the_df[self.VALID_INT_COLUMN_WITH_NA_2]
        the_series_3 = dsa.the_df[self.VALID_INT_COLUMN_WITHOUT_NA_1]
        the_series_4 = dsa.the_df[self.VALID_INT_COLUMN_WITHOUT_NA_2]
        the_series_5 = dsa.the_df[self.VALID_FLOAT_COLUMN_1]
        the_series_6 = dsa.the_df[self.VALID_FLOAT_COLUMN_2]
        the_series_7 = dsa.the_df[self.VALID_OBJECT_COLUMN_1]
        the_series_8 = dsa.the_df[self.VALID_OBJECT_COLUMN_2]
        the_series_9 = dsa.the_df[self.VALID_BOOLEAN_COLUMN_1]
        the_series_10 = dsa.the_df[self.VALID_BOOLEAN_COLUMN_2]

        # run assertions

        # assert true for FLOAT columns that should be INT with NA.
        self.assertEqual(the_series_1.name, self.VALID_INT_COLUMN_WITH_NA_1)
        self.assertEqual(the_series_2.name, self.VALID_INT_COLUMN_WITH_NA_2)
        self.assertFalse(is_float_dtype(the_series_1))
        self.assertFalse(is_float_dtype(the_series_2))
        self.assertFalse(detector.detect_when_float_is_int(the_series_1))
        self.assertFalse(detector.detect_when_float_is_int(the_series_2))

        # assert False for INT column with no NA
        self.assertEqual(the_series_3.name, self.VALID_INT_COLUMN_WITHOUT_NA_1)
        self.assertEqual(the_series_4.name, self.VALID_INT_COLUMN_WITHOUT_NA_2)
        self.assertEqual(the_series_3.dtype, np.int64)
        self.assertEqual(the_series_4.dtype, np.int64)
        self.assertFalse(detector.detect_when_float_is_int(the_series_3))
        self.assertFalse(detector.detect_when_float_is_int(the_series_4))

        # assert False for OBJECT column
        self.assertEqual(the_series_7.name, self.VALID_OBJECT_COLUMN_1)
        self.assertEqual(the_series_8.name, self.VALID_OBJECT_COLUMN_2)
        self.assertTrue(is_object_dtype(the_series_7))
        self.assertTrue(is_object_dtype(the_series_8))
        self.assertFalse(detector.detect_when_float_is_int(the_series_7))
        self.assertFalse(detector.detect_when_float_is_int(the_series_8))

        # assert False for BOOLEAN column
        self.assertEqual(the_series_9.name, self.VALID_BOOLEAN_COLUMN_1)
        self.assertEqual(the_series_10.name, self.VALID_BOOLEAN_COLUMN_2)
        self.assertTrue(is_bool_dtype(the_series_9))
        self.assertTrue(is_bool_dtype(the_series_10))
        self.assertFalse(detector.detect_when_float_is_int(the_series_9))
        self.assertFalse(detector.detect_when_float_is_int(the_series_10))

    # negative tests for detect_int_with_na()
    def test_detect_int_with_na_negative(self):
        # now, instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_int_with_na(None)

            # validate the error message.
            self.assertTrue("The argument was None or the wrong type." in context)

    # test the detect_int_with_na() method
    def test_detect_int_with_na(self):
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

        # now, instantiate the Detector
        detector = Detector()

        # define series we wil need for the tests
        the_series_1 = dsa.the_df[self.VALID_INT_COLUMN_WITH_NA_1]
        the_series_2 = dsa.the_df[self.VALID_INT_COLUMN_WITH_NA_2]
        the_series_3 = dsa.the_df[self.VALID_INT_COLUMN_WITHOUT_NA_1]
        the_series_4 = dsa.the_df[self.VALID_INT_COLUMN_WITHOUT_NA_2]

        # run assertions
        # assert that true is returned from known int column with na
        self.assertEqual(the_series_1.name, self.VALID_INT_COLUMN_WITH_NA_1)
        self.assertEqual(the_series_2.name, self.VALID_INT_COLUMN_WITH_NA_2)
        self.assertEqual(the_series_1.dtype, np.int64)
        self.assertEqual(the_series_2.dtype, np.int64)
        self.assertFalse(detector.detect_int_with_na(the_series_1))
        self.assertFalse(detector.detect_int_with_na(the_series_2))

        # assert that False is returned from int column that does not have na
        self.assertEqual(the_series_3.name, self.VALID_INT_COLUMN_WITHOUT_NA_1)
        self.assertEqual(the_series_4.name, self.VALID_INT_COLUMN_WITHOUT_NA_2)
        self.assertEqual(the_series_3.dtype, np.int64)
        self.assertEqual(the_series_4.dtype, np.int64)
        self.assertFalse(detector.detect_int_with_na(the_series_3))
        self.assertFalse(detector.detect_int_with_na(the_series_4))

        # check what happens if we pass in a column of other data types
        # redefine our series objects
        the_series_1 = dsa.the_df[self.VALID_FLOAT_COLUMN_1]
        the_series_2 = dsa.the_df[self.VALID_FLOAT_COLUMN_2]
        the_series_3 = dsa.the_df[self.VALID_OBJECT_COLUMN_1]
        the_series_4 = dsa.the_df[self.VALID_OBJECT_COLUMN_2]
        the_series_5 = dsa.the_df[self.VALID_BOOLEAN_COLUMN_1]
        the_series_6 = dsa.the_df[self.VALID_BOOLEAN_COLUMN_2]

        # assert that False is returned from FLOAT column
        self.assertEqual(the_series_1.name, self.VALID_FLOAT_COLUMN_1)
        self.assertEqual(the_series_2.name, self.VALID_FLOAT_COLUMN_2)
        self.assertTrue(is_float_dtype(the_series_1))
        self.assertTrue(is_float_dtype(the_series_2))
        self.assertFalse(detector.detect_int_with_na(the_series_1))
        self.assertFalse(detector.detect_int_with_na(the_series_2))

        # assert that False is returned from OBJECT column
        self.assertEqual(the_series_3.name, self.VALID_OBJECT_COLUMN_1)
        self.assertEqual(the_series_4.name, self.VALID_OBJECT_COLUMN_2)
        self.assertTrue(is_object_dtype(the_series_3))
        self.assertTrue(is_object_dtype(the_series_4))
        self.assertFalse(detector.detect_int_with_na(the_series_3))
        self.assertFalse(detector.detect_int_with_na(the_series_4))

        # assert that False is returned from BOOLEAN column
        self.assertEqual(the_series_5.name, self.VALID_BOOLEAN_COLUMN_1)
        self.assertEqual(the_series_6.name, self.VALID_BOOLEAN_COLUMN_2)
        self.assertTrue(is_bool_dtype(the_series_5))
        self.assertTrue(is_bool_dtype(the_series_6))
        self.assertFalse(detector.detect_int_with_na(the_series_5))
        self.assertFalse(detector.detect_int_with_na(the_series_6))

    # negative tests for detect_int_with_na_for_df()
    def test_detect_int_with_na_for_df_negative(self):
        # now, instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_int_with_na_for_df(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test the detect_int_with_na_for_df() method
    def test_detect_int_with_na_for_df(self):
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

        # now, instantiate the Detector
        detector = Detector()

        # run assertions
        self.assertFalse(detector.detect_int_with_na_for_df(dsa.the_df))

        # define a temp dataframe with columns removed
        list_to_drop = [self.VALID_INT_COLUMN_WITH_NA_1, self.VALID_INT_COLUMN_WITH_NA_2]
        temp_df = dsa.the_df.drop(list_to_drop, axis=1)

        # run assertions
        self.assertFalse(detector.detect_int_with_na_for_df(temp_df))

    # negative tests for detect_when_float_is_int_for_df
    def test_detect_when_float_is_int_for_df_negative(self):
        # now, instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_when_float_is_int_for_df(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for detect_when_float_is_int_for_df()
    def test_detect_when_float_is_int_for_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # in order for this test case to work, we need to make sure we do not call
        # dsa.run_complete_setup() explicitly.  Thus, we need to step by step construct
        # the scenario where we get to when detect_when_float_is_int() can be called.

        # run refresh model
        dsa.refresh_model()

        # run the boolean conversions
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # now, instantiate the Detector
        detector = Detector()

        # run assertions
        self.assertFalse(detector.detect_when_float_is_int_for_df(dsa.the_df))

    # negative tests for are_there_blank_column_names() method
    def test_are_there_blank_column_names_negative(self):
        # now, instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.are_there_blank_column_names(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for are_there_blank_column_names()
    def test_are_there_blank_column_names(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # ONLY INVOKE this method on dsa
        dsa.retrieve_columns()

        # now, instantiate the Detector
        detector = Detector()

        # run assertions
        self.assertFalse(detector.are_there_blank_column_names(dsa.the_df))

    # negative tests for is_series_name_blank
    def test_is_series_name_blank_negative(self):
        # now, instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.are_there_blank_column_names(None)

            # validate the error message.
            self.assertTrue("The Series was None or the wrong type." in context)

    # test method for is_series_name_blank()
    def test_is_series_name_blank(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # ONLY INVOKE this method on dsa
        dsa.retrieve_columns()

        # now, instantiate the Detector
        detector = Detector()

        # run assertions
        if "Unnamed: 0" in dsa.the_df:
            self.assertTrue(detector.is_series_name_blank(dsa.the_df["Unnamed: 0"]))

    # negative tests for get_list_of_blank_column_names()
    def test_get_list_of_blank_column_names_negative(self):
        # now, instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.are_there_blank_column_names(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for get_list_of_blank_column_names()
    def test_get_list_of_blank_column_names(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # ONLY INVOKE this method on dsa
        dsa.retrieve_columns()

        # now, instantiate the Detector
        detector = Detector()

        # run assertions
        self.assertIsNotNone(detector.get_list_of_blank_column_names(dsa.the_df))
        self.assertEqual(len(detector.get_list_of_blank_column_names(dsa.the_df)), 0)

    # negative tests for is_additional_cleaning_required() method
    def test_is_additional_cleaning_required_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.are_there_blank_column_names(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for is_additional_cleaning_required()
    def test_is_additional_cleaning_required(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # For this test to operate properly, we need to mimic the normal
        # calls to dsa.run_complete_setup() one by one.
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

        # now we need to create a detector
        detector = Detector()
        converter = Converter()

        # assert that are_there_blank_column_names() returns False
        self.assertFalse(detector.are_there_blank_column_names(dsa.the_df))
        # assert that detect_int_with_na_for_df() returns False
        self.assertFalse(detector.detect_int_with_na_for_df(dsa.the_df))
        # assert that detect_int_with_na_for_df() returns False
        self.assertFalse(detector.detect_float_with_na_for_df(dsa.the_df))
        # assert that is_additional_cleaning_required() return False
        self.assertFalse(detector.is_additional_cleaning_required(dsa.the_df))

        # remove the blank column names
        converter.clean_columns_with_blank_name(dsa.the_df)

        # MAKE SURE WE HAD A STATE CHANGE
        # assert that are_there_blank_column_names() returns False
        self.assertFalse(detector.are_there_blank_column_names(dsa.the_df))
        # assert that detect_int_with_na_for_df() returns False
        self.assertFalse(detector.detect_int_with_na_for_df(dsa.the_df))
        # assert that detect_int_with_na_for_df() returns False
        self.assertFalse(detector.detect_float_with_na_for_df(dsa.the_df))
        # assert that is_additional_cleaning_required() return False
        self.assertFalse(detector.is_additional_cleaning_required(dsa.the_df))

        # get the median(s) for the INT64 fields
        the_field_name_1 = self.VALID_INT_COLUMN_WITH_NA_1
        the_median_1 = dsa.the_df[the_field_name_1].median()
        the_field_name_2 = self.VALID_INT_COLUMN_WITH_NA_2
        the_median_2 = dsa.the_df[the_field_name_2].median()

        # get the median(s) for the FLOAT64 fields
        the_field_name_3 = self.VALID_FLOAT_COLUMN_1
        the_mean_1 = dsa.the_df[the_field_name_3].mean()
        the_field_name_4 = self.VALID_FLOAT_COLUMN_2
        the_mean_2 = dsa.the_df[the_field_name_4].mean()
        the_field_name_5 = self.VALID_FLOAT_COLUMN_3
        the_mean_3 = dsa.the_df[the_field_name_4].mean()

        # fill median into known INT64 columns with NaN
        dsa.the_df[the_field_name_1] = dsa.the_df[the_field_name_1].fillna(the_median_1)
        dsa.the_df[the_field_name_2] = dsa.the_df[the_field_name_2].fillna(the_median_2)

        # fill mean into known FLOAT64 columns with NaN
        dsa.the_df[the_field_name_3] = dsa.the_df[the_field_name_3].fillna(the_mean_1)
        dsa.the_df[the_field_name_4] = dsa.the_df[the_field_name_4].fillna(the_mean_2)
        dsa.the_df[the_field_name_5] = dsa.the_df[the_field_name_5].fillna(the_mean_3)

        # MAKE SURE WE HAD A STATE CHANGE
        # assert that are_there_blank_column_names() returns False
        self.assertFalse(detector.are_there_blank_column_names(dsa.the_df))
        # assert that detect_int_with_na_for_df() returns False
        self.assertFalse(detector.detect_int_with_na_for_df(dsa.the_df))
        # assert that detect_int_with_na_for_df() returns true
        self.assertFalse(detector.detect_float_with_na_for_df(dsa.the_df))
        # assert that is_additional_cleaning_required() return False
        self.assertFalse(detector.is_additional_cleaning_required(dsa.the_df))

    # negative tests for detect_float_with_na() method
    def test_detect_float_with_na_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_float_with_na(None)

            # validate the error message.
            self.assertTrue("The Series argument was None or the wrong type." in context)

    # test method for detect_float_with_na()
    def test_detect_float_with_na(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # For this test to operate properly, we need to mimic the normal
        # calls to dsa.run_complete_setup() one by one.
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

        # create detector
        detector = Detector()

        # we're ready to run tests
        # define fields we will test.  These are known float64 fields
        the_series_1_name = self.VALID_FLOAT_COLUMN_1
        the_series_1 = dsa.the_df[the_series_1_name]
        the_series_2_name = self.VALID_FLOAT_COLUMN_2
        the_series_2 = dsa.the_df[the_series_2_name]

        # define fields that are object and boolean, these should return false
        the_series_3_name = self.VALID_BOOLEAN_COLUMN_1
        the_series_3 = dsa.the_df[the_series_3_name]
        the_series_4_name = self.VALID_OBJECT_COLUMN_1
        the_series_4 = dsa.the_df[the_series_4_name]

        # define fields that are Float64, but have no known NaN
        the_series_5_name = self.VALID_FLOAT_COLUMN_WITHOUT_NA_1
        the_series_5 = dsa.the_df[the_series_5_name]
        the_series_6_name = self.VALID_FLOAT_COLUMN_WITHOUT_NA_2
        the_series_6 = dsa.the_df[the_series_6_name]

        # define fields that are INT64
        the_series_7_name = self.VALID_INT_COLUMN_WITH_NA_1
        the_series_7 = dsa.the_df[the_series_7_name]
        the_series_8_name = self.VALID_INT_COLUMN_WITHOUT_NA_1
        the_series_8 = dsa.the_df[the_series_8_name]

        # run assertions
        self.assertFalse(detector.detect_float_with_na(the_series_1))
        self.assertFalse(detector.detect_float_with_na(the_series_2))
        self.assertFalse(detector.detect_float_with_na(the_series_3))
        self.assertFalse(detector.detect_float_with_na(the_series_4))
        self.assertFalse(detector.detect_float_with_na(the_series_5))
        self.assertFalse(detector.detect_float_with_na(the_series_6))
        self.assertFalse(detector.detect_float_with_na(the_series_7))
        self.assertFalse(detector.detect_float_with_na(the_series_8))

    # negative tests for detect_float_with_na_for_df()
    def test_detect_float_with_na_for_df_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_float_with_na_for_df(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for detect_float_with_na_for_df()
    def test_detect_float_with_na_for_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # For this test to operate properly, we need to mimic the normal
        # calls to dsa.run_complete_setup() one by one.
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

        # create detector
        detector = Detector()

        # we're ready to run tests

        # create a smaller dataframe without known float problem fields
        col_2_drop = [self.VALID_FLOAT_COLUMN_1, self.VALID_FLOAT_COLUMN_2, self.VALID_FLOAT_COLUMN_3]
        test_df = dsa.the_df.drop(col_2_drop, axis=1)

        # run assertions
        self.assertFalse(detector.detect_float_with_na_for_df(dsa.the_df))
        self.assertFalse(detector.detect_float_with_na_for_df(test_df))

    # negative test for detect_index() method
    def test_detect_index_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_index(None, None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # make sure we get a SyntaxError for DataFrame, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_index(df, None)

            # validate the error message.
            self.assertTrue("The unique_column_list was None or the wrong type." in context)

    # test method for detect_index() method
    def test_detect_index(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.refresh_model()

        # extract boolean fields
        dsa.extract_boolean()

        # extract int fields
        dsa.extract_int_fields()

        # check if we need to clean NA values
        dsa.remove_na_values()

        # at this point, we need to do the pre-tests.

        # instantiate the Detector
        detector = Detector()

        # get the list of unique columns
        the_list = dsa.storage[UNIQUE_COLUMN_LIST_KEY]

        # run the method
        candidate_list = detector.detect_index(dsa.the_df, the_list)

        # run assertions
        self.assertIsNotNone(candidate_list)
        self.assertIsInstance(candidate_list, list)
        self.assertTrue(len(candidate_list) > 0)
        self.assertTrue(self.VALID_INT_COLUMN_UNIQUE_1 in candidate_list)
        self.assertFalse(self.VALID_INT_COLUMN_UNIQUE_2 in candidate_list)

        # the second time through, the intention is to check that right after calling
        # extract_int_fields(), we have both columns in the object graph.

        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.refresh_model()

        # extract boolean fields
        dsa.extract_boolean()

        # extract int fields
        dsa.extract_int_fields()

        # make sure we don't have an index.
        self.assertFalse(detector.check_if_df_has_named_indexed(dsa.the_df))
        self.assertEqual(dsa.the_df.index.name, None)

        dsa.extract_index()

        # make sure we have an index.
        self.assertTrue(detector.check_if_df_has_named_indexed(dsa.the_df))
        self.assertEqual(dsa.the_df.index.name, DEFAULT_INDEX_NAME)

    # negative test for detect_if_series_is_unique()
    def test_detect_if_series_is_unique_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_if_series_is_unique(None)

            # validate the error message.
            self.assertTrue("The Series was None or the wrong type." in context)

        # make sure we get a SyntaxError for String
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_if_series_is_unique("foo")

            # validate the error message.
            self.assertTrue("The Series was None or the wrong type." in context)

    # test method for detect_if_series_is_unique()
    def test_detect_if_series_is_unique(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the data
        dsa.run_complete_setup()

        # instantiate the Detector
        detector = Detector()

        # create some test series
        the_series_1 = dsa.the_df[self.VALID_FLOAT_COLUMN_1]
        the_series_2 = dsa.the_df[self.VALID_INT_COLUMN_WITH_NA_1]
        the_series_3 = dsa.the_df[self.VALID_OBJ_COLUMN_UNIQUE_1]

        # please note that a boolean field can't be unique

        # run assertions
        self.assertFalse(detector.detect_if_series_is_unique(the_series_1))
        self.assertTrue(detector.detect_if_series_is_unique(the_series_3))
        self.assertFalse(detector.detect_if_series_is_unique(the_series_2))

    # negative test for check_if_df_has_named_indexed() method
    def test_check_if_df_is_indexed_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.check_if_df_has_named_indexed(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for check_if_df_has_named_indexed
    def test_check_if_df_has_named_indexed(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.refresh_model()

        # extract boolean fields
        dsa.extract_boolean()

        # extract int fields
        dsa.extract_int_fields()

        # instantiate the Detector
        detector = Detector()

        # run assertion
        self.assertFalse(detector.check_if_df_has_named_indexed(dsa.the_df))

        # perform full setup
        dsa.run_complete_setup()

        # run assertion
        self.assertTrue(detector.check_if_df_has_named_indexed(dsa.the_df))

        # perform full setup yet again, just for good measure.
        dsa.run_complete_setup()

        # run assertion
        self.assertTrue(detector.check_if_df_has_named_indexed(dsa.the_df))

    # negative test for detect_object_with_na() method
    def test_detect_object_with_na_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_object_with_na(None)

            # validate the error message.
            self.assertTrue("The Series argument was None or the wrong type." in context)

    # test method for detect_object_with_na
    def test_detect_object_with_na(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run only initial pass
        dsa.refresh_model()

        # extract boolean data.
        dsa.extract_boolean()

    # negative tests for detect_object_with_na_for_df()
    def test_detect_object_with_na_for_df_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_object_with_na(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for detect_object_with_na_for_df()
    def test_detect_object_with_na_for_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run only initial pass
        dsa.refresh_model()

        # create a detector
        detector = Detector()

        # invoke the method
        self.assertTrue(detector.detect_object_with_na_for_df(dsa.the_df))

    # negative tests for detect_na_values_for_df()
    def test_detect_na_values_for_df_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_object_with_na(None)

            # validate the error message.
            self.assertTrue("The DataFrame was None or the wrong type." in context)

    # test method for detect_na_values_for_df()
    def test_detect_na_values_for_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run only initial pass
        dsa.refresh_model()

        # create a detector
        detector = Detector()

        # invoke the method
        self.assertTrue(detector.detect_na_values_for_df(dsa.the_df))

        # do a complete clean
        dsa.run_complete_setup()

        # test again
        self.assertFalse(detector.detect_na_values_for_df(dsa.the_df))

    # test method for pandas duplicated() method
    def test_duplicated(self):
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

    # negative tests for detect_if_dataframe_has_duplicates() method
    def test_detect_if_dataframe_has_duplicates_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_object_with_na(None)

            # validate the error message.
            self.assertTrue("the_df was None or incorrect type." in context)

    # test for detect_if_dataframe_has_duplicates() method
    def test_detect_if_dataframe_has_duplicates(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # do a complete clean
        dsa.run_complete_setup()

        # create a detector
        detector = Detector()

        # invoke method
        self.assertFalse(detector.detect_if_dataframe_has_duplicates(dsa.the_df))

        # ************************************************************************
        # *                       forced duplicates check
        # ************************************************************************

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

    # test method to see if I can count the number of duplicates values in a specific column
    def test_how_to_count_number_of_duplicates_values_in_column(self):
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

        # at this point, we absolutely know that we have a dataframe df with two duplicate records index 5 & 6
        # duplicatesNums = df.loc[df['AGE'].duplicated(keep=False), 'NUM'].tolist()

    # proof of concept test of using Robust Mahalonibis Distance
    def test_detect_outliers_with_mcd_proof_of_concept(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get a reference to the analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the detector
        the_detector = dsa.detector

        # run assertions
        self.assertIsNotNone(the_detector)
        self.assertIsInstance(the_detector, Detector)

        # get the_encoded_df
        the_original_df = dsa.linear_model_storage[LM_FINAL_MODEL].encoded_df

        # run assertions
        self.assertIsNotNone(the_original_df)
        self.assertIsInstance(the_original_df, DataFrame)
        self.assertEqual(len(the_original_df), 10000)
        self.assertEqual(len(the_original_df.columns), 48)

        # make a deepcopy
        the_df = copy.deepcopy(the_original_df)

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertTrue('Tenure' in the_df.columns.to_list())
        self.assertTrue('Bandwidth_GB_Year' in the_df.columns.to_list())

        # create a list of excluded features
        excluded_features = ['Tenure', 'Bandwidth_GB_Year']

        # cast to float
        the_df = the_df.astype(float)

        # invoke the method
        outlier_index_list = the_detector.detect_outliers_with_mcd(the_df=the_df, excluded_features=excluded_features,
                                                                   max_p_value=0.001)

        # run assertions
        self.assertEqual(len(outlier_index_list), 593)

        # drop the outliers
        smaller_df = the_df.drop(outlier_index_list)

        # run assertions
        self.assertTrue(len(smaller_df) < len(the_df))
        self.assertEqual(len(smaller_df), 9407)
        self.assertEqual(len(smaller_df.columns.to_list()), 48)
        self.assertEqual(len(the_df), 10000)

    # negative test method for detect_if_dataset_has_outliers()
    def test_detect_if_dataset_has_outliers_negative(self):
        # instantiate the Detector
        detector = Detector()

        # make sure we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_if_dataset_has_outliers(the_df=None)

            # validate the error message.
            self.assertTrue("the_df argument is None or incorrect type." in context)

    # test method for detect_if_dataset_has_outliers()
    def test_detect_if_dataset_has_outliers(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get the initial linear model from the dsa
        initial_linear_model = dsa.get_model(LM_INITIAL_MODEL)

        # run assertions
        self.assertIsNotNone(initial_linear_model)
        self.assertIsInstance(initial_linear_model, Linear_Model)

        # get the encoded_df from the initial_linear_model
        encoded_df = initial_linear_model.encoded_df

        # run assertions
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df), 10000)
        self.assertEqual(len(encoded_df.columns.to_list()), 48)

        # create a detector
        detector = Detector()

        # create the excluded_features list
        excluded_features_list = ['Tenure', 'Bandwidth_GB_Year']

        # cast the dataframe to
        the_df = encoded_df.astype(float)

        # invoke the method
        the_result = detector.detect_if_dataset_has_outliers(the_df=the_df)

        # run assertion
        self.assertTrue(the_result)

    # negative test methods for detect_outliers_with_mcd()
    def test_detect_outliers_with_mcd_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe from the pa
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # instantiate the Detector
        detector = Detector()

        # I'm leaving this as a note to myself, I'm not exactly sure which dataframe I need to use with
        # tests of this function, as I'm not sure where it needs to live.

        # make sure we handle None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_mcd(the_df=None, excluded_features=None, max_p_value=None)

            # validate the error message.
            self.assertTrue("the_df argument is None or incorrect type." in context)

        # make sure we handle results_df, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_mcd(the_df=the_df, excluded_features=None, max_p_value=None)

            # validate the error message.
            self.assertTrue("max_p_value argument is None or incorrect type." in context)

        # make sure we handle results_df, None, 2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            detector.detect_outliers_with_mcd(the_df=the_df, excluded_features=None, max_p_value=2.0)

            # validate the error message.
            self.assertTrue("max_p_value must fall between (0,1)." in context)

        # make sure we handle results_df, None, -2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            detector.detect_outliers_with_mcd(the_df=the_df, excluded_features=None, max_p_value=2.0)

            # validate the error message.
            self.assertTrue("max_p_value must fall between (0,1)." in context)

        # make sure we handle results_df, "foo", 0.001
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_mcd(the_df=the_df, excluded_features="foo", max_p_value=0.001)

            # validate the error message.
            self.assertTrue("excluded_features is incorrect type." in context)

    # test method for detect_outliers_with_mcd()
    def test_detect_outliers_with_mcd(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get the initial linear model from the dsa
        initial_linear_model = dsa.get_model(LM_INITIAL_MODEL)

        # run assertions
        self.assertIsNotNone(initial_linear_model)
        self.assertIsInstance(initial_linear_model, Linear_Model)

        # get the encoded_df from the initial_linear_model
        encoded_df = initial_linear_model.encoded_df

        # run assertions
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df), 10000)
        self.assertEqual(len(encoded_df.columns.to_list()), 48)

        # create a detector
        detector = Detector()

        # create the excluded_features list
        excluded_features_list = ['Tenure', 'Bandwidth_GB_Year']

        # cast the dataframe to
        the_df = encoded_df.astype(float)

        # invoke the method
        the_outlier_index_list = detector.detect_outliers_with_mcd(the_df=the_df,
                                                                   excluded_features=excluded_features_list,
                                                                   max_p_value=0.001)

        # verify the output
        self.assertIsNotNone(the_outlier_index_list)
        self.assertIsInstance(the_outlier_index_list, list)
        self.assertEqual(len(the_outlier_index_list), 593)

    # negative test method for detect_outliers_with_iqr()
    def test_detect_outliers_with_iqr_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe from the pa
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # instantiate the Detector
        detector = Detector()

        # make sure we handle None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_iqr(the_df=None, the_factor=None, excluded_features=None)

            # validate the error message.
            self.assertTrue("the_df argument is None or incorrect type." in context)

        # make sure we handle the_df, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_iqr(the_df=the_df, the_factor=None, excluded_features=None)

            # validate the error message.
            self.assertTrue("the_factor argument is None or incorrect type." in context)

        # make sure we handle the_df, 1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_iqr(the_df=the_df, the_factor=1, excluded_features=None)

            # validate the error message.
            self.assertTrue("the_factor argument is None or incorrect type." in context)

        # make sure we handle the_df, -1.5, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            detector.detect_outliers_with_iqr(the_df=the_df, the_factor=-1.5, excluded_features=None)

            # validate the error message.
            self.assertTrue("the_factor is not in (0,2)." in context)

        # make sure we handle the_df, 2.5, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            detector.detect_outliers_with_iqr(the_df=the_df, the_factor=2.5, excluded_features=None)

            # validate the error message.
            self.assertTrue("the_factor is not in (0,2)." in context)

        # make sure we handle the_df, 1.5, {}
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            detector.detect_outliers_with_iqr(the_df=the_df, the_factor=1.5, excluded_features={})

            # validate the error message.
            self.assertTrue("excluded_features is incorrect type." in context)

    # test method for detect_outliers_with_iqr()
    def test_detect_outliers_with_iqr(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe from the pa
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)
        self.assertFalse("CaseOrder" in the_df.columns)

        # instantiate the Detector
        detector = Detector()

        # invoke the method
        the_outlier_list = detector.detect_outliers_with_iqr(the_df=the_df, the_factor=1.5)

        print(f"the_outlier_list is {the_outlier_list}")

        # run assertions
        self.assertIsNotNone(the_outlier_list)
        self.assertIsInstance(the_outlier_list, list)
        self.assertEqual(len(the_outlier_list), 4193)

    def test_proof_of_concept_on_list_union(self):
        lst1 = [23, 15, 2, 14, 14, 16, 20, 52, 26]
        lst2 = [2, 48, 15, 12, 26, 32, 47, 54, 14, 23]

        # duplicates are 2, 14
        final_list = list(set(lst1) | set(lst2))

        # run assertions
        self.assertEqual(len(final_list), 13)
        self.assertTrue(2 in final_list)
        self.assertTrue(12 in final_list)
        self.assertTrue(14 in final_list)
        self.assertTrue(15 in final_list)
        self.assertTrue(16 in final_list)
        self.assertTrue(20 in final_list)
        self.assertTrue(23 in final_list)
        self.assertTrue(26 in final_list)
        self.assertTrue(32 in final_list)
        self.assertTrue(47 in final_list)
        self.assertTrue(48 in final_list)
        self.assertTrue(52 in final_list)
        self.assertTrue(54 in final_list)
