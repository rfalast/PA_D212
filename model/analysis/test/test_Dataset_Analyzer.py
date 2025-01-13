import logging
import unittest

from pandas import DataFrame
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Detector import Detector
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.ModelBase import ModelBase
from model.constants.BasicConstants import TIMEZONE_COLUMN, ANALYZE_DATASET_FULL, D_212_CHURN, MT_LINEAR_REGRESSION, \
    MT_LOGISTIC_REGRESSION, MT_KNN_CLASSIFICATION, MT_RF_REGRESSION
from model.constants.DatasetConstants import COLUMN_KEY, COLUMN_COUNT_KEY, COLUMN_NA_COUNT, COLUMN_TOTAL_COUNT_KEY, \
    UNIQUE_COLUMN_VALUES, INT64_COLUMN_KEY, FLOAT64_COLUMN_KEY, OBJECT_COLUMN_KEY, BOOL_COLUMN_KEY, BOOL_VALUE_KEY, \
    UNIQUE_COLUMN_RATIOS, BOOL_COLUMN_COUNT_KEY, DATETIME_COLUMN_KEY, DATASET_TYPES, UNIQUE_COLUMN_FLAG, \
    OBJECT_COLUMN_COUNT_KEY, UNIQUE_COLUMN_LIST_KEY
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_FINAL_MODEL
from util.CSV_loader import CSV_Loader


# test case for the DatasetAnalyzer
class test_DatasetAnalyzer(unittest.TestCase):
    # constants
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    OVERRIDDEN_LOCATION = "../../../resources/Output/"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"
    VALID_COLUMN_1 = "Age"
    VALID_COLUMN_2 = "Tenure"
    Z_SCORE = "_z_score"
    UN_NAMED_COLUMN = "Unnamed: 0"
    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction']

    VALID_NA_INT64_FIELD_1 = "Children"
    VALID_NA_INT64_FIELD_2 = "Age"
    VALID_NA_FLOAT64_FIELD_1 = "Tenure"
    VALID_NA_FLOAT64_FIELD_2 = "Bandwidth_GB_Year"
    VALID_NA_OBJECT_FIELD_1 = "Techie"
    VALID_NA_OBJECT_FIELD_2 = "InternetService"
    VALID_BOOLEAN_FIELD_1 = "Churn"
    VALID_OBJECT_FIELD_1 = "TimeZone"

    VALID_FIELD_DICT = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                        "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                        "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}


    INITIAL_UNIQUE_COLUMN_LIST = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'Bandwidth_GB_Year']

    EXPECTED_NORM_COLUMNS = ['Population_z_score', 'TimeZone_z_score', 'Children_z_score',
                             'Age_z_score', 'Income_z_score', 'Outage_sec_perweek_z_score',
                             'Email_z_score', 'Contacts_z_score', 'Yearly_equip_failure_z_score',
                             'Tenure_z_score', 'MonthlyCharge_z_score', 'Bandwidth_GB_Year_z_score',
                             'Item1_z_score', 'Item2_z_score', 'Item3_z_score', 'Item4_z_score',
                             'Item5_z_score', 'Item6_z_score', 'Item7_z_score', 'Item8_z_score']

    VALID_COLUMN_DROP_LIST_2 = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction',
                                'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_212_CHURN

    # test case for init
    def test_init(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run assertions
        self.assertIsNotNone(dsa.the_df)
        self.assertIsInstance(dsa.the_df, DataFrame)

        # instantiate with the dataset name optional argument
        dsa = DatasetAnalyzer(df, the_name="foo")
        self.assertIsNotNone(dsa.the_df)
        self.assertIsInstance(dsa.the_df, DataFrame)
        self.assertIsNotNone(dsa.dataset_name)
        self.assertEqual(dsa.dataset_name, "foo")

    # negative tests for init() method
    def test_init_negative(self):
        # we're going to pass in None on the init method to get SyntaxError
        with self.assertRaises(SyntaxError) as context:
            # None argument for DatasetAnalyzer()
            DatasetAnalyzer(the_dataframe=None)

        # validate the error message.
        self.assertTrue("The incoming argument was None or incorrect type." in context.exception.msg)

    # test the retrieve_columns() method first pass
    # noinspection DuplicatedCode
    def test_retrieve_columns(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run assertions
        # verify we have a storage
        self.assertIsNotNone(dsa.storage)
        self.assertIsInstance(dsa.storage, dict)

        # verify what is on the keys in storage after first pass
        self.assertIsNotNone(dsa.storage[COLUMN_KEY])
        self.assertIsInstance(dsa.storage[COLUMN_KEY], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_COUNT_KEY])
        self.assertIsInstance(dsa.storage[COLUMN_COUNT_KEY], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_NA_COUNT])
        self.assertIsInstance(dsa.storage[COLUMN_NA_COUNT], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_TOTAL_COUNT_KEY])
        self.assertIsInstance(dsa.storage[COLUMN_TOTAL_COUNT_KEY], dict)
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_VALUES])
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_VALUES], dict)
        self.assertEqual(len(list(dsa.storage[UNIQUE_COLUMN_VALUES].keys())), 27)

        self.assertIsNotNone(dsa.storage[COLUMN_NA_COUNT])
        self.assertIsInstance(dsa.storage[COLUMN_NA_COUNT], dict)
        self.assertEqual(len(dsa.storage[COLUMN_NA_COUNT]), 50)
        self.assertIsNotNone(dsa.storage[INT64_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[INT64_COLUMN_KEY], list)

        # verify the number in the first pass
        self.assertEqual(len(dsa.storage[INT64_COLUMN_KEY]), 16)
        # make sure timezone is not one of them
        self.assertFalse('TimeZone' in dsa.storage[INT64_COLUMN_KEY])

        self.assertIsNotNone(dsa.storage[FLOAT64_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[FLOAT64_COLUMN_KEY], list)
        # verify the number in the first pass
        self.assertEqual(len(dsa.storage[FLOAT64_COLUMN_KEY]), 7)

        self.assertIsNotNone(dsa.storage[OBJECT_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[OBJECT_COLUMN_KEY], list)
        # verify the number in the first pass
        self.assertEqual(len(dsa.storage[OBJECT_COLUMN_KEY]), 27)
        self.assertTrue('TimeZone' in dsa.storage[OBJECT_COLUMN_KEY])

        # verify what is empty after first pass
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[BOOL_COLUMN_KEY], list)
        self.assertEqual(len(dsa.storage[BOOL_COLUMN_KEY]), 0)
        self.assertIsNotNone(dsa.storage[BOOL_VALUE_KEY])
        self.assertIsInstance(dsa.storage[BOOL_VALUE_KEY], dict)
        self.assertEqual(len(dsa.storage[BOOL_VALUE_KEY]), 0)

        # make sure we have all 52 columns first time through.
        self.assertTrue(len(dsa.storage[COLUMN_KEY]) == 50)

        # run assertions on lists
        self.assertTrue(len(dsa.storage[INT64_COLUMN_KEY]) == 16)
        self.assertTrue(len(dsa.storage[FLOAT64_COLUMN_KEY]) == 7)
        self.assertTrue(len(dsa.storage[OBJECT_COLUMN_KEY]) == 27)

        # now we need to make sure that the NaN dict has a column for each field
        self.assertEqual(len(dsa.storage[COLUMN_NA_COUNT]), 50)

        # RUN VALIDATIONS FOR UNIQUE COLUMNS

        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_LIST_KEY])
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_LIST_KEY], list)
        self.assertEqual(len(dsa.storage[UNIQUE_COLUMN_LIST_KEY]), 5)

        # loop over the list
        for the_column in dsa.storage[UNIQUE_COLUMN_LIST_KEY]:
            # run assertion
            self.assertTrue(the_column in self.INITIAL_UNIQUE_COLUMN_LIST)

        # make sure has boolean is false
        self.assertFalse(dsa.has_boolean)

    # test the retrieve_columns() method second pass
    # noinspection DuplicatedCode
    def test_retrieve_columns_second_pass(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # retrieve the columns
        dsa.retrieve_columns()

        # run analysis
        dsa.analyze_columns()

        # re-run retrieve columns
        dsa.retrieve_columns()

        # run assertions
        # verify we have a storage
        self.assertIsNotNone(dsa.storage)
        self.assertIsInstance(dsa.storage, dict)

        # verify what is on the keys in storage after first pass
        self.assertIsNotNone(dsa.storage[COLUMN_KEY])
        self.assertIsInstance(dsa.storage[COLUMN_KEY], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_COUNT_KEY])
        self.assertIsInstance(dsa.storage[COLUMN_COUNT_KEY], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_NA_COUNT])
        self.assertIsInstance(dsa.storage[COLUMN_NA_COUNT], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_TOTAL_COUNT_KEY])
        self.assertIsInstance(dsa.storage[COLUMN_TOTAL_COUNT_KEY], dict)
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_VALUES])
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_VALUES], dict)
        self.assertIsNotNone(dsa.storage[COLUMN_NA_COUNT])
        self.assertIsInstance(dsa.storage[COLUMN_NA_COUNT], dict)

        self.assertIsNotNone(dsa.storage[INT64_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[INT64_COLUMN_KEY], list)
        self.assertIsNotNone(dsa.storage[FLOAT64_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[FLOAT64_COLUMN_KEY], list)
        self.assertIsNotNone(dsa.storage[OBJECT_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[OBJECT_COLUMN_KEY], list)

        # verify what is empty after first pass
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_KEY])
        self.assertIsInstance(dsa.storage[BOOL_COLUMN_KEY], list)
        self.assertEqual(len(dsa.storage[BOOL_COLUMN_KEY]), 0)
        self.assertIsNotNone(dsa.storage[BOOL_VALUE_KEY])
        self.assertIsInstance(dsa.storage[BOOL_VALUE_KEY], dict)
        self.assertEqual(len(dsa.storage[BOOL_VALUE_KEY]), 13)

        # make sure we have all 52 columns first time through.
        self.assertTrue(len(dsa.storage[COLUMN_KEY]) == 50)

        # run assertions on lists
        self.assertEqual(len(dsa.storage[INT64_COLUMN_KEY]), 16)
        self.assertEqual(len(dsa.storage[FLOAT64_COLUMN_KEY]), 7)
        self.assertEqual(len(dsa.storage[OBJECT_COLUMN_KEY]), 27)

        # now we need to make sure that the NaN dict has a column for each field
        self.assertEqual(len(dsa.storage[COLUMN_NA_COUNT]), 50)

        # make sure has boolean is true
        self.assertTrue(dsa.has_boolean)

        # RUN VALIDATIONS FOR UNIQUE COLUMNS

        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_LIST_KEY])
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_LIST_KEY], list)
        self.assertEqual(len(dsa.storage[UNIQUE_COLUMN_LIST_KEY]), 5)

        # loop over the list
        for the_column in dsa.storage[UNIQUE_COLUMN_LIST_KEY]:
            # run assertion
            self.assertTrue(the_column in self.INITIAL_UNIQUE_COLUMN_LIST)

    # test analyze_columns()
    def test_analyze_columns(self):
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

        # run assertions on what is in object graph for storage dict after first pass
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_KEY])
        self.assertIsNotNone(dsa.storage[BOOL_VALUE_KEY])
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_RATIOS])
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_VALUES])

        # verify type of objects in storage object graph
        self.assertIsInstance(dsa.storage[BOOL_COLUMN_KEY], list)
        self.assertIsInstance(dsa.storage[BOOL_VALUE_KEY], dict)
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_RATIOS], dict)
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_VALUES], dict)

        # verify dimensionality
        self.assertEqual(len(dsa.storage[BOOL_COLUMN_KEY]), 0)  # we should have 0 until run_conversion_for_boolean
        self.assertEqual(len(dsa.storage[BOOL_VALUE_KEY]), 13)
        self.assertEqual(len(dsa.storage[OBJECT_COLUMN_KEY]), 27)
        self.assertEqual(len(dsa.storage[UNIQUE_COLUMN_VALUES]), 27)  # has obj and bool counts
        self.assertEqual(len(dsa.storage[UNIQUE_COLUMN_RATIOS]), 27)  # has obj and bool counts

        # loop over all the booleans, make sure they are Yes or No, and the type is a list of str
        for next_value in list(dsa.storage[BOOL_VALUE_KEY].values()):
            # run assertions
            self.assertIsInstance(next_value, list)
            self.assertTrue(len(next_value), 2)

        # loop over all the lists attached to the key UNIQUE_COLUMN_VALUES, make sure they are
        # lists and have a population of 1.
        for next_value in dsa.storage[UNIQUE_COLUMN_VALUES].values():
            # run assertions
            self.assertIsInstance(next_value, list)
            self.assertGreaterEqual(len(next_value), 1)

        # we need to verify that some of the fields that are unique only have a single
        # record stored on UNIQUE_COLUMN_VALUES, and that record is ["TOO MANY TO DISPLAY"]
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_VALUES]['Customer_id'], list)
        self.assertEqual(dsa.storage[UNIQUE_COLUMN_VALUES]['Customer_id'], ["TOO MANY TO DISPLAY"])

        # AFTER CLEANING, make sure the UNIQUE_COLUMN_LIST_KEY is correct

        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_LIST_KEY])
        self.assertIsInstance(dsa.storage[UNIQUE_COLUMN_LIST_KEY], list)
        self.assertEqual(len(dsa.storage[UNIQUE_COLUMN_LIST_KEY]), 5)

        # loop over the list
        for the_column in dsa.storage[UNIQUE_COLUMN_LIST_KEY]:
            # run assertion
            self.assertTrue(the_column in self.INITIAL_UNIQUE_COLUMN_LIST)

    # test cases for analyze_boolean_data()
    def test_analyze_boolean_data(self):
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

        # log that we're starting the function to test
        logging.debug("About to call analyze_boolean_data()")

        # run boolean conversion
        dsa.run_conversion_for_boolean()

        # capture and analyze the columns again, because we've made a conversion
        dsa.refresh_model()

        # analyze boolean data
        dsa.analyze_boolean_data()

        # log that we're starting the function to test
        logging.debug("Finished call to analyze_boolean_data()")

        # capture the created dictionary
        bool_dict = dsa.storage[BOOL_COLUMN_COUNT_KEY]

        # log the output from the function call
        logging.debug("created boolean dict[%s]", str(bool_dict))

        # run assertions
        self.assertIsNotNone(bool_dict)
        self.assertEqual(len(dsa.storage[BOOL_COLUMN_COUNT_KEY]), 13)
        self.assertIsInstance(dsa.storage[BOOL_COLUMN_COUNT_KEY], dict)

        logging.debug("About to check the entire object graph.")

        # loop over all the booleans, make sure they are True or False.
        for next_value in list(dsa.storage[BOOL_COLUMN_COUNT_KEY].values()):
            # log the value
            logging.debug("next_value [%s]", next_value)

            # run assertions
            self.assertIsInstance(next_value, dict)
            self.assertTrue(len(next_value), 2)
            self.assertIsNotNone(dsa.storage[BOOL_COLUMN_COUNT_KEY])

    # test method for has_ints_as_floats()
    def test_has_ints_as_floats(self):
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

        # run assertions.  This dataset is clean.
        self.assertFalse(dsa.has_ints_as_floats())

    # test the run_conversion_from_float_2int() method
    def test_run_conversion_from_float_2int(self):
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

        # invoke the function
        dsa.run_conversion_from_float_2int()

        # run assertions
        self.assertFalse(dsa.has_ints_as_floats())
        self.assertTrue("Children" in dsa.storage[INT64_COLUMN_KEY])
        self.assertTrue("Age" in dsa.storage[INT64_COLUMN_KEY])
        self.assertFalse("Children" in dsa.storage[FLOAT64_COLUMN_KEY])
        self.assertFalse("Age" in dsa.storage[FLOAT64_COLUMN_KEY])
        self.assertEqual(dsa.storage[COLUMN_NA_COUNT]['Children'], 0)
        self.assertEqual(dsa.storage[COLUMN_NA_COUNT]['Age'], 0)

    # test the run_complete_setup() method
    def test_run_complete_setup(self):
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

        # get the primary dataframe
        the_df = dsa.the_df

        # run assertions to make sure we know what we've got
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns.to_list()), 50)
        self.assertEqual(len(the_df), 10000)

        # invoke run_complete_setup()
        dsa.run_complete_setup()

        # create a detector
        detector = Detector()

        # run assertions
        self.assertIsNotNone(dsa.storage[COLUMN_KEY])  # column_dict
        self.assertIsNotNone(dsa.storage[INT64_COLUMN_KEY])  # int_list
        self.assertIsNotNone(dsa.storage[FLOAT64_COLUMN_KEY])  # float_list
        self.assertIsNotNone(dsa.storage[OBJECT_COLUMN_KEY])  # object_list
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_KEY])  # bool_list
        self.assertIsNotNone(dsa.storage[DATETIME_COLUMN_KEY])  # datetime_list
        self.assertIsNotNone(dsa.storage[COLUMN_COUNT_KEY])  # count_dict
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_VALUES])  # unique_dict
        self.assertIsNotNone(dsa.storage[COLUMN_TOTAL_COUNT_KEY])  # total_count_dict
        self.assertIsNotNone(dsa.storage[COLUMN_NA_COUNT])  # column_na_count_dict
        self.assertIsNotNone(dsa.storage[DATASET_TYPES])  # data_type_list
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_FLAG])
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_RATIOS])
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_COUNT_KEY])
        self.assertIsNotNone(dsa.storage[OBJECT_COLUMN_COUNT_KEY])

        # run validations that cleaning is occurring
        # make sure there are no NUMERIC fields with na
        self.assertFalse(detector.are_there_blank_column_names(dsa.the_df))
        self.assertFalse(detector.is_additional_cleaning_required(dsa.the_df))

        # validate index is set to CaseOrder
        self.assertEqual("CaseOrder", dsa.the_df.index.name)

        # make sure we don't have duplicates
        self.assertFalse(detector.detect_if_dataframe_has_duplicates(dsa.the_df))

        # make sure TimeZone is an int
        self.assertTrue('TimeZone' in dsa.storage[INT64_COLUMN_KEY])

        # make sure TimeZone is not a categorical.
        self.assertFalse('TimeZone' in dsa.storage[OBJECT_COLUMN_KEY])

        # validate fields are present on the DataFrame
        self.assertTrue("Churn" in dsa.the_df)
        self.assertTrue("Churn" in list(dsa.storage[UNIQUE_COLUMN_VALUES].keys()))

        # verify columns
        self.assertEqual(len(the_df.columns.to_list()), 49)
        self.assertEqual(len(the_df), 10000)

    # test the run_complete_setup() method
    def test_run_complete_setup_NO_TIMEZONE(self):
        # The purpose of this test is to verify that the code works if the TIME_ZONE feature is
        # removed from the dataset.

        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # drop the TIMEZONE field from dataframe
        dsa.the_df = dsa.the_df.drop(TIMEZONE_COLUMN, axis=1)

        # run the complete setup.
        dsa.run_complete_setup()

        # create a detector
        detector = Detector()

        # run assertions
        self.assertIsNotNone(dsa.storage[COLUMN_KEY])  # column_dict
        self.assertIsNotNone(dsa.storage[INT64_COLUMN_KEY])  # int_list
        self.assertIsNotNone(dsa.storage[FLOAT64_COLUMN_KEY])  # float_list
        self.assertIsNotNone(dsa.storage[OBJECT_COLUMN_KEY])  # object_list
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_KEY])  # bool_list
        self.assertIsNotNone(dsa.storage[DATETIME_COLUMN_KEY])  # datetime_list
        self.assertIsNotNone(dsa.storage[COLUMN_COUNT_KEY])  # count_dict
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_VALUES])  # unique_dict
        self.assertIsNotNone(dsa.storage[COLUMN_TOTAL_COUNT_KEY])  # total_count_dict
        self.assertIsNotNone(dsa.storage[COLUMN_NA_COUNT])  # column_na_count_dict
        self.assertIsNotNone(dsa.storage[DATASET_TYPES])  # data_type_list
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_FLAG])
        self.assertIsNotNone(dsa.storage[UNIQUE_COLUMN_RATIOS])
        self.assertIsNotNone(dsa.storage[BOOL_COLUMN_COUNT_KEY])
        self.assertIsNotNone(dsa.storage[OBJECT_COLUMN_COUNT_KEY])

        # run validations that cleaning is occurring
        # make sure there are no NUMERIC fields with na
        self.assertFalse(detector.are_there_blank_column_names(dsa.the_df))
        self.assertFalse(detector.is_additional_cleaning_required(dsa.the_df))

        # validate index is set to CaseOrder
        self.assertEqual("CaseOrder", dsa.the_df.index.name)

        # make sure we don't have duplicates
        self.assertFalse(detector.detect_if_dataframe_has_duplicates(dsa.the_df))

        # validate fields are present on the DataFrame
        self.assertTrue("Churn" in dsa.the_df)
        self.assertTrue("Churn" in list(dsa.storage[UNIQUE_COLUMN_VALUES].keys()))

    # negative tests for normalize_dataset() method
    def test_normalize_dataset_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # make sure we get a SyntaxError for "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.normalize_dataset(overriden_df="foo")

        # validate the error message.
        self.assertTrue("overriden_df argument must be DataFrame." in context.exception.msg)

    # test method for normalize_dataset() method
    def test_normalize_dataset(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # invoke method
        dsa.normalize_dataset()

        # run assertions
        self.assertIsNotNone(dsa.the_normal_df)
        self.assertIsInstance(dsa.the_normal_df, DataFrame)
        self.assertNotEqual(dsa.the_normal_df[self.VALID_COLUMN_1 + self.Z_SCORE].sum(),
                            dsa.the_df[self.VALID_COLUMN_1].sum())

        # verify the expected number of normalized elements in columns and rows
        self.assertEqual(len(dsa.the_normal_df), 10000)
        self.assertEqual(len(dsa.the_normal_df.columns), 23)

        # the next test is to verify that the optional column_drop_list works properly

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # invoke method
        dsa.normalize_dataset(column_drop_list=self.column_drop_list)

        # run assertions
        self.assertIsNotNone(dsa.the_normal_df)
        self.assertIsInstance(dsa.the_normal_df, DataFrame)

        # make sure drop list columns are not present.
        self.assertFalse("Zip" in dsa.the_normal_df)
        self.assertFalse("Lat" in dsa.the_normal_df)
        self.assertFalse("Lng" in dsa.the_normal_df)
        self.assertFalse("Customer_id" in dsa.the_normal_df)
        self.assertFalse("Interaction" in dsa.the_normal_df)

        # verify the expected number of normalized elements in columns and rows
        self.assertEqual(len(dsa.the_normal_df), 10000)
        self.assertEqual(len(dsa.the_normal_df.columns), 20)
        self.assertEqual(len(self.EXPECTED_NORM_COLUMNS), 20)

        # loop over all the columns in the normalized dataset
        for the_column in dsa.the_normal_df.columns:
            self.assertTrue(the_column in self.EXPECTED_NORM_COLUMNS)

    # test method for normalize_dataset() method
    def test_normalize_dataset_WITH_overriden_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # invoke method
        dsa.normalize_dataset()

        # run assertions
        self.assertIsNotNone(dsa.the_normal_df)
        self.assertIsInstance(dsa.the_normal_df, DataFrame)
        self.assertNotEqual(dsa.the_normal_df[self.VALID_COLUMN_1 + self.Z_SCORE].sum(),
                            dsa.the_df[self.VALID_COLUMN_1].sum())

        # verify the expected number of normalized elements in columns and rows
        self.assertEqual(len(dsa.the_normal_df), 10000)
        self.assertEqual(len(dsa.the_normal_df.columns), 23)

        # the next test is to verify that the optional column_drop_list works properly

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # invoke method
        dsa.normalize_dataset(column_drop_list=self.column_drop_list)

        # run assertions
        self.assertIsNotNone(dsa.the_normal_df)
        self.assertIsInstance(dsa.the_normal_df, DataFrame)

        # make sure drop list columns are not present.
        self.assertFalse("Zip" in dsa.the_normal_df)
        self.assertFalse("Lat" in dsa.the_normal_df)
        self.assertFalse("Lng" in dsa.the_normal_df)
        self.assertFalse("Customer_id" in dsa.the_normal_df)
        self.assertFalse("Interaction" in dsa.the_normal_df)

        # verify the expected number of normalized elements in columns and rows
        self.assertEqual(len(dsa.the_normal_df), 10000)
        self.assertEqual(len(dsa.the_normal_df.columns), 20)
        self.assertEqual(len(self.EXPECTED_NORM_COLUMNS), 20)

        # loop over all the columns in the normalized dataset
        for the_column in dsa.the_normal_df.columns:
            self.assertTrue(the_column in self.EXPECTED_NORM_COLUMNS)

    # test the getName method
    def test_get_name(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df, the_name="foo")

        # run assertions
        self.assertIsNotNone(dsa.get_name())
        self.assertIsInstance(dsa.get_name(), str)
        self.assertEqual(dsa.get_name(), "foo")

    # negative tests for add_original_df() method
    def test_add_original_df_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # make sure we get a SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.add_original_df(the_df=None)

        # validate the error message.
        self.assertTrue("The incoming argument was None or incorrect type." in context.exception.msg)

        # make sure we get a SyntaxError for "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.add_original_df(the_df="foo")

        # validate the error message.
        self.assertTrue("The incoming argument was None or incorrect type." in context.exception.msg)

    # test for the add_original_df() method
    def test_add_original_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # invoke the method
        dsa.add_original_df(cl.get_data_frame_from_csv(self.VALID_CSV_PATH))

        # run assertions
        self.assertIsNotNone(dsa.the_original_df)
        self.assertIsInstance(dsa.the_original_df, DataFrame)

        # clean the dataset
        dsa.run_complete_setup()

        # the dataframes should be different, as dsa.the_df is now cleaned.
        self.assertFalse(dsa.the_df.equals(dsa.the_original_df))

    # test method for remove_na_values()
    def test_remove_na_values(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # invoke the method
        dsa.add_original_df(cl.get_data_frame_from_csv(self.VALID_CSV_PATH))

        # refresh the model
        dsa.refresh_model()

        # extract boolean fields
        dsa.extract_boolean()

        # extract int fields
        dsa.extract_int_fields()

        # check if NA is present for several INT64 and FLOAT fields
        self.assertEqual(dsa.the_df[self.VALID_NA_INT64_FIELD_1].isna().sum(), 0)
        self.assertEqual(dsa.the_df[self.VALID_NA_INT64_FIELD_2].isna().sum(), 0)
        self.assertEqual(dsa.the_df[self.VALID_NA_FLOAT64_FIELD_1].isna().sum(), 0)
        self.assertEqual(dsa.the_df[self.VALID_NA_FLOAT64_FIELD_2].isna().sum(), 0)
        self.assertEqual(dsa.the_df[self.VALID_NA_OBJECT_FIELD_1].isna().sum(), 0)
        self.assertEqual(dsa.the_df[self.VALID_NA_OBJECT_FIELD_1].isna().sum(), 0)

        # call the function
        dsa.remove_na_values()

        # check if NA is present for several INT64 and FLOAT fields
        self.assertFalse(dsa.the_df[self.VALID_NA_INT64_FIELD_1].isna().sum() > 0)
        self.assertFalse(dsa.the_df[self.VALID_NA_INT64_FIELD_2].isna().sum() > 0)
        self.assertFalse(dsa.the_df[self.VALID_NA_FLOAT64_FIELD_1].isna().sum() > 0)
        self.assertFalse(dsa.the_df[self.VALID_NA_FLOAT64_FIELD_2].isna().sum() > 0)
        self.assertFalse(dsa.the_df[self.VALID_NA_OBJECT_FIELD_1].isna().sum() > 0)
        self.assertFalse(dsa.the_df[self.VALID_NA_OBJECT_FIELD_1].isna().sum() > 0)

    # negative test method for validate_field_type()
    def test_validate_field_type_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # make sure we get a SyntaxError for None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.validate_field_type(None, None)

        # validate the error message.
        self.assertTrue("the_field is None or incorrect type." in context.exception.msg)

        # make sure we get a SyntaxError for list, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.validate_field_type(list, None)

        # validate the error message.
        self.assertTrue("the_field is None or incorrect type." in context.exception.msg)

        # make sure we get a SyntaxError for "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.validate_field_type("foo", None)

        # validate the error message.
        self.assertTrue("the_field is not present in DataFrame." in context.exception.msg)

        # make sure we get a SyntaxError for VALID_COLUMN_1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.validate_field_type(self.VALID_COLUMN_1, None)

        # validate the error message.
        self.assertTrue("the_type is None or incorrect type." in context.exception.msg)

        # make sure we get a SyntaxError for VALID_COLUMN_1, "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.validate_field_type(self.VALID_NA_INT64_FIELD_1, "foo")

        # validate the error message.
        self.assertTrue("the_type is None or incorrect type." in context.exception.msg)

    # test method for validate_field_type()
    def test_validate_field_type(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # run assertions
        self.assertTrue(dsa.validate_field_type(self.VALID_OBJECT_FIELD_1, INT64_COLUMN_KEY))
        self.assertTrue(dsa.validate_field_type(self.VALID_NA_INT64_FIELD_1, INT64_COLUMN_KEY))
        self.assertTrue(dsa.validate_field_type(self.VALID_NA_FLOAT64_FIELD_1, FLOAT64_COLUMN_KEY))
        self.assertTrue(dsa.validate_field_type(self.VALID_BOOLEAN_FIELD_1, BOOL_COLUMN_KEY))
        self.assertTrue(dsa.validate_field_type(self.VALID_OBJECT_FIELD_1, INT64_COLUMN_KEY))

        self.assertFalse(dsa.validate_field_type(self.VALID_NA_INT64_FIELD_1, FLOAT64_COLUMN_KEY))
        self.assertFalse(dsa.validate_field_type(self.VALID_NA_FLOAT64_FIELD_1, OBJECT_COLUMN_KEY))
        self.assertFalse(dsa.validate_field_type(self.VALID_BOOLEAN_FIELD_1, INT64_COLUMN_KEY))
        self.assertTrue(dsa.validate_field_type("Gender", OBJECT_COLUMN_KEY))

    # test method for is_data_type_valid()
    def test_is_data_type_valid(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # run assertions
        self.assertFalse(dsa.is_data_type_valid("foo"))
        self.assertTrue(dsa.is_data_type_valid(FLOAT64_COLUMN_KEY))
        self.assertTrue(dsa.is_data_type_valid(INT64_COLUMN_KEY))
        self.assertTrue(dsa.is_data_type_valid(OBJECT_COLUMN_KEY))
        self.assertTrue(dsa.is_data_type_valid(BOOL_COLUMN_KEY))

    # negative test method for get_linear_model() method
    def test_get_linear_model_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # make sure we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.get_model(the_type=None)

        # validate the error message.
        self.assertTrue("the_type is None or incorrect option." in context.exception.msg)

        # make sure we handle valid request when the not fully setup.
        with self.assertRaises(RuntimeError) as context:
            # invoke the method
            dsa.get_model(the_type=LM_INITIAL_MODEL)

        # validate the error message.
        self.assertTrue("the linear_model storage is not setup." in context.exception.args)

    # test method for get_model()
    def test_get_model(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # run assertions that the storage is empty
        self.assertFalse(LM_INITIAL_MODEL in dsa.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in dsa.linear_model_storage)

        # create a linear_model
        the_linear_model = Linear_Model(dataset_analyzer=dsa)

        # put it into the storage
        dsa.linear_model_storage[LM_INITIAL_MODEL] = the_linear_model

        # invoke the method
        the_result = dsa.get_model(LM_INITIAL_MODEL)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model)

    # negative test method for add_linear_model()
    def test_add_linear_model_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.add_model(the_type=None, the_model=None)

        # validate the error message.
        self.assertTrue("the_type is None or incorrect option." in context.exception.msg)

        # make sure we handle LM_INITIAL_MODEL, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.add_model(the_type=LM_INITIAL_MODEL, the_model=None)

        # validate the error message.
        self.assertTrue("the_model is None or incorrect type." in context.exception.msg)

        # make sure we handle LM_INITIAL_MODEL, "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.add_model(the_type=LM_INITIAL_MODEL, the_model="foo")

        # validate the error message.
        self.assertTrue("the_model is None or incorrect type." in context.exception.msg)

    # test method for add_model()
    def test_add_model(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(the_dataframe=df, the_name="first")

        # run the complete setup.
        dsa.run_complete_setup()

        # create a linear_model
        the_linear_model = Linear_Model(dataset_analyzer=dsa)

        # run assertions that the storage is empty
        self.assertFalse(LM_INITIAL_MODEL in dsa.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in dsa.linear_model_storage)

        # invoke the method
        dsa.add_model(the_type=LM_INITIAL_MODEL, the_model=the_linear_model)

        # run assertions
        self.assertIsNotNone(dsa.linear_model_storage[LM_INITIAL_MODEL])
        self.assertFalse(LM_FINAL_MODEL in dsa.linear_model_storage)
        self.assertIsInstance(dsa.linear_model_storage[LM_INITIAL_MODEL], ModelBase)

    # negative test method for clean_up_outliers()
    def test_clean_up_outliers_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dsa
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get the dataframe from the dsa
        the_df = dsa.the_df

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.clean_up_outliers(model_type=None, max_p_value=None)

        # validate the error message.
        self.assertTrue("model_type is None or incorrect type." in context.exception.msg)

        # make sure we handle "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            dsa.clean_up_outliers(model_type="foo", max_p_value=None)

        # validate the error message.
        self.assertTrue("model_type value is unknown." in context.exception.args)

        # make sure we handle MT_LINEAR_REGRESSION, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=None)

        # validate the error message.
        self.assertTrue("max_p_value argument is None or incorrect type." in context.exception.args)

        # make sure we handle MT_LINEAR_REGRESSION, -1.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            dsa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=-1.0)

        # validate the error message.
        self.assertTrue("max_p_value is not in (0,1)." in context.exception.args)

        # make sure we handle MT_LINEAR_REGRESSION, 1.1
        with self.assertRaises(ValueError) as context:
            # invoke the method
            dsa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=1.1)

        # validate the error message.
        self.assertTrue("max_p_value is not in (0,1)." in context.exception.args)

    # test method for clean_up_outliers() on MT_LINEAR_REGRESSION
    def test_clean_up_outliers_linear_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the underlying dataframe
        the_df = dsa.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertEqual(len(the_df.columns), 39)

        # invoke the method
        dsa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9266)

        # this needs to still equal the original, because we're not messing with columns
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # test method for clean_up_outliers() on MT_LOGISTIC_REGRESSION
    def test_clean_up_outliers_logistic_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the underlying dataframe
        the_df = dsa.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertEqual(len(the_df.columns), 39)

        # invoke the method
        dsa.clean_up_outliers(model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9266)

        # this needs to still equal the original, because we're not messing with columns
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # test method for clean_up_outliers() on MT_KNN_CLASSIFICATION
    def test_clean_up_outliers_knn_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the underlying dataframe
        the_df = dsa.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertEqual(len(the_df.columns), 39)

        # invoke the method
        dsa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9582)

        # this needs to still equal the original, because we're not messing with columns
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # test method for clean_up_outliers() on MT_RF_REGRESSION
    def test_clean_up_outliers_rf_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run assertions
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the underlying dataframe
        the_df = dsa.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertEqual(len(the_df.columns), 39)

        # invoke the method
        dsa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9582)

        # this needs to still equal the original, because we're not messing with columns
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # negative test method for retrieve_features_of_specific_type()
    def test_retrieve_features_of_specific_type_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run validations on dsa
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the underlying dataframe
        the_df = dsa.the_df

        # run assertions on the_df
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertEqual(len(the_df.columns), 39)

        # make sure we handle None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            dsa.retrieve_features_of_specific_type(the_f_type=None)

        # validate the error message.
        self.assertTrue("the_f_type is None or incorrect type." in context.exception.args)
        # make sure we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            dsa.retrieve_features_of_specific_type(the_f_type=['foo', 'bad'])

        # validate the error message.
        self.assertTrue("a unexpected data type was found in the_f_type list." in context.exception.msg)

    # test method for retrieve_features_of_specific_type()
    def test_retrieve_features_of_specific_type(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dsa = pa.analyzer

        # run validations on dsa
        self.assertIsNotNone(dsa)
        self.assertIsInstance(dsa, DatasetAnalyzer)

        # get a reference to the underlying dataframe
        the_df = dsa.the_df

        # run assertions on the_df
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertEqual(len(the_df.columns), 39)

        # retrieve INT64_COLUMN_KEY only
        the_f_type = [INT64_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 15)
        self.assertEqual(current_expected_features[0], 'Population')
        self.assertEqual(current_expected_features[1], 'TimeZone')
        self.assertEqual(current_expected_features[2], 'Children')
        self.assertEqual(current_expected_features[3], 'Age')
        self.assertEqual(current_expected_features[4], 'Email')
        self.assertEqual(current_expected_features[5], 'Contacts')
        self.assertEqual(current_expected_features[6], 'Yearly_equip_failure')
        self.assertEqual(current_expected_features[7], 'Timely_Response')
        self.assertEqual(current_expected_features[8], 'Timely_Fixes')
        self.assertEqual(current_expected_features[9], 'Timely_Replacements')
        self.assertEqual(current_expected_features[10], 'Reliability')
        self.assertEqual(current_expected_features[11], 'Options')
        self.assertEqual(current_expected_features[12], 'Respectful_Response')
        self.assertEqual(current_expected_features[13], 'Courteous_Exchange')
        self.assertEqual(current_expected_features[14], 'Active_Listening')

        # retrieve BOOL_COLUMN_KEY only
        the_f_type = [BOOL_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 13)
        self.assertEqual(current_expected_features[0], 'Churn')
        self.assertEqual(current_expected_features[1], 'Techie')
        self.assertEqual(current_expected_features[2], 'Port_modem')
        self.assertEqual(current_expected_features[3], 'Tablet')
        self.assertEqual(current_expected_features[4], 'Phone')
        self.assertEqual(current_expected_features[5], 'Multiple')
        self.assertEqual(current_expected_features[6], 'OnlineSecurity')
        self.assertEqual(current_expected_features[7], 'OnlineBackup')
        self.assertEqual(current_expected_features[8], 'DeviceProtection')
        self.assertEqual(current_expected_features[9], 'TechSupport')
        self.assertEqual(current_expected_features[10], 'StreamingTV')
        self.assertEqual(current_expected_features[11], 'StreamingMovies')
        self.assertEqual(current_expected_features[12], 'PaperlessBilling')

        # retrieve FLOAT64_COLUMN_KEY only
        the_f_type = [FLOAT64_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 5)
        self.assertEqual(current_expected_features[0], 'Income')
        self.assertEqual(current_expected_features[1], 'Outage_sec_perweek')
        self.assertEqual(current_expected_features[2], 'Tenure')
        self.assertEqual(current_expected_features[3], 'MonthlyCharge')
        self.assertEqual(current_expected_features[4], 'Bandwidth_GB_Year')

        # retrieve OBJECT_COLUMN_KEY only
        the_f_type = [OBJECT_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 6)
        self.assertEqual(current_expected_features[0], 'Area')
        self.assertEqual(current_expected_features[1], 'Marital')
        self.assertEqual(current_expected_features[2], 'Gender')
        self.assertEqual(current_expected_features[3], 'Contract')
        self.assertEqual(current_expected_features[4], 'InternetService')
        self.assertEqual(current_expected_features[5], 'PaymentMethod')

        # retrieve OBJECT_COLUMN_KEY and FLOAT64_COLUMN_KEY
        the_f_type = [OBJECT_COLUMN_KEY, FLOAT64_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 11)
        self.assertEqual(current_expected_features[0], 'Area')
        self.assertEqual(current_expected_features[1], 'Marital')
        self.assertEqual(current_expected_features[2], 'Gender')
        self.assertEqual(current_expected_features[3], 'Contract')
        self.assertEqual(current_expected_features[4], 'InternetService')
        self.assertEqual(current_expected_features[5], 'PaymentMethod')
        self.assertEqual(current_expected_features[6], 'Income')
        self.assertEqual(current_expected_features[7], 'Outage_sec_perweek')
        self.assertEqual(current_expected_features[8], 'Tenure')
        self.assertEqual(current_expected_features[9], 'MonthlyCharge')
        self.assertEqual(current_expected_features[10], 'Bandwidth_GB_Year')

        # retrieve FLOAT64_COLUMN_KEY and BOOL_COLUMN_KEY
        the_f_type = [FLOAT64_COLUMN_KEY, BOOL_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 18)
        self.assertEqual(current_expected_features[0], 'Income')
        self.assertEqual(current_expected_features[1], 'Outage_sec_perweek')
        self.assertEqual(current_expected_features[2], 'Tenure')
        self.assertEqual(current_expected_features[3], 'MonthlyCharge')
        self.assertEqual(current_expected_features[4], 'Bandwidth_GB_Year')
        self.assertEqual(current_expected_features[5], 'Churn')
        self.assertEqual(current_expected_features[6], 'Techie')
        self.assertEqual(current_expected_features[7], 'Port_modem')
        self.assertEqual(current_expected_features[8], 'Tablet')
        self.assertEqual(current_expected_features[9], 'Phone')
        self.assertEqual(current_expected_features[10], 'Multiple')
        self.assertEqual(current_expected_features[11], 'OnlineSecurity')
        self.assertEqual(current_expected_features[12], 'OnlineBackup')
        self.assertEqual(current_expected_features[13], 'DeviceProtection')
        self.assertEqual(current_expected_features[14], 'TechSupport')
        self.assertEqual(current_expected_features[15], 'StreamingTV')
        self.assertEqual(current_expected_features[16], 'StreamingMovies')
        self.assertEqual(current_expected_features[17], 'PaperlessBilling')

        # retrieve BOOL_COLUMN_KEY and INT64_COLUMN_KEY
        the_f_type = [BOOL_COLUMN_KEY, INT64_COLUMN_KEY]

        # get the list of features.
        current_expected_features = dsa.retrieve_features_of_specific_type(the_f_type)

        # run assertions
        self.assertIsNotNone(current_expected_features)
        self.assertIsInstance(current_expected_features, list)
        self.assertEqual(len(current_expected_features), 28)
        self.assertEqual(current_expected_features[0], 'Churn')
        self.assertEqual(current_expected_features[1], 'Techie')
        self.assertEqual(current_expected_features[2], 'Port_modem')
        self.assertEqual(current_expected_features[3], 'Tablet')
        self.assertEqual(current_expected_features[4], 'Phone')
        self.assertEqual(current_expected_features[5], 'Multiple')
        self.assertEqual(current_expected_features[6], 'OnlineSecurity')
        self.assertEqual(current_expected_features[7], 'OnlineBackup')
        self.assertEqual(current_expected_features[8], 'DeviceProtection')
        self.assertEqual(current_expected_features[9], 'TechSupport')
        self.assertEqual(current_expected_features[10], 'StreamingTV')
        self.assertEqual(current_expected_features[11], 'StreamingMovies')
        self.assertEqual(current_expected_features[12], 'PaperlessBilling')
        self.assertEqual(current_expected_features[13], 'Population')
        self.assertEqual(current_expected_features[14], 'TimeZone')
        self.assertEqual(current_expected_features[15], 'Children')
        self.assertEqual(current_expected_features[16], 'Age')
        self.assertEqual(current_expected_features[17], 'Email')
        self.assertEqual(current_expected_features[18], 'Contacts')
        self.assertEqual(current_expected_features[19], 'Yearly_equip_failure')
        self.assertEqual(current_expected_features[20], 'Timely_Response')
        self.assertEqual(current_expected_features[21], 'Timely_Fixes')
        self.assertEqual(current_expected_features[22], 'Timely_Replacements')
        self.assertEqual(current_expected_features[23], 'Reliability')
        self.assertEqual(current_expected_features[24], 'Options')
        self.assertEqual(current_expected_features[25], 'Respectful_Response')
        self.assertEqual(current_expected_features[26], 'Courteous_Exchange')
        self.assertEqual(current_expected_features[27], 'Active_Listening')

