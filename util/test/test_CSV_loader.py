import os
import unittest
from os.path import exists
from pathlib import Path, PosixPath

from pandas import DataFrame

from model.Project_Assessment import Project_Assessment
from model.constants.BasicConstants import D_209_CHURN, ANALYZE_DATASET_FULL, CHURN_FINAL, CHURN_CSV_FILE_LOCATION, \
    CHURN_PREP, CHURN_PREP_CSV_FILE_LOCATION, MEDICAL_FINAL, D_209_MEDICAL, MEDICAL_CSV_FILE_LOCATION, MEDICAL_PREP, \
    MEDICAL_PREP_FILE_LOCATION, OUTPUT_OPTIONS, MT_LOGISTIC_REGRESSION
from model.constants.ModelConstants import LM_FINAL_MODEL
from util.CSV_loader import CSV_Loader


# test class for CSV_loader()
class test_CSV_loader(unittest.TestCase):
    # constants
    VALID_PATH = "../../resources/Input/churn_raw_data.csv"
    INVALID_PATH = "foo.txt"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDDEN_LOCATION = "../../resources/Output/"

    VALID_FIELD_DICT = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                        "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                        "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    VALID_COLUMN_DROP_LIST_2 = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction',
                                'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_209_CHURN

    # test the init() method
    def test_init_no_argument(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # run assertions
        self.assertIsNotNone(cl)
        self.assertIsInstance(cl, CSV_Loader)

    # negative test for init() method with base_directory argument
    def test_init_with_base_directory_negative(self):
        # make sure we throw an NotADirectoryError for None argument
        with self.assertRaises(NotADirectoryError) as context:
            # invoke method
            cl = CSV_Loader(base_directory="foo")

            # validate the error message.
            self.assertTrue("base_directory does not exist." in context.exception)

    # test method for init() with base_directory argument
    def test_init_with_base_directory(self):
        # create the instance
        cl = CSV_Loader(base_directory=self.VALID_BASE_DIR)

        # load a file
        self.assertIsNotNone(cl.get_data_frame_from_csv(self.VALID_PATH))
        self.assertIsInstance(cl.get_data_frame_from_csv(self.VALID_PATH), DataFrame)

    # negative tests for the get_data_frame_from_csv() method
    def test_get_data_frame_from_csv_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # make sure we throw an TypeError for None argument
        with self.assertRaises(TypeError) as context:
            # invoke find_in_row()
            cl.get_data_frame_from_csv(None)

            # validate the error message.
            self.assertTrue("csv_path was None or not a string." in context.exception)

        # make sure we throw an FileNotFoundError for bad path
        with self.assertRaises(FileNotFoundError) as context:
            # invoke find_in_row()
            cl.get_data_frame_from_csv(self.INVALID_PATH)

            # validate the error message.
            self.assertTrue("The file references from csv_path does not exist." in context.exception)

    # test the get_data_frame_from_csv() method
    def test_get_data_frame_from_csv(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # run assertions
        self.assertIsNotNone(cl.get_data_frame_from_csv(self.VALID_PATH))
        self.assertIsInstance(cl.get_data_frame_from_csv(self.VALID_PATH), DataFrame)

    # negative tests for generate_output_file_path()
    def test_generate_output_file_path_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # check that we handle None, None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.csv_l.generate_output_file_path(data_set=None, option=None)

            # validate the error message.
            self.assertTrue("data_set is None or incorrect type." in context.exception)

        # check that we handle OUTPUT_OPTIONS[self.CHURN_KEY], None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.csv_l.generate_output_file_path(data_set=OUTPUT_OPTIONS[self.CHURN_KEY],
                                               option=None)

            # validate the error message.
            self.assertTrue("option is None or incorrect type." in context.exception)

    # test method for generate_output_file_path()
    def test_generate_output_file_path(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # TEST #1
        # invoke method for self.CHURN_KEY, CHURN_FINAL
        the_path = pa.csv_l.generate_output_file_path(data_set=self.CHURN_KEY, option=CHURN_FINAL)
        the_test_path = "/Users/robertfalast/PycharmProjects/PA_209/" + CHURN_CSV_FILE_LOCATION

        # run validation
        self.assertIsNotNone(the_path)
        self.assertIsInstance(the_path, Path)
        self.assertEqual(the_path, PosixPath(the_test_path))

        # TEST #2
        # invoke method for self.CHURN_KEY, CHURN_PREP
        the_path = pa.csv_l.generate_output_file_path(data_set=self.CHURN_KEY, option=CHURN_PREP)
        the_test_path = "/Users/robertfalast/PycharmProjects/PA_209/" + CHURN_PREP_CSV_FILE_LOCATION

        # run validation
        self.assertIsNotNone(the_path)
        self.assertIsInstance(the_path, Path)
        self.assertEqual(the_path, PosixPath(the_test_path))

        # TEST #3
        # invoke method for D_209_MEDICAL, MEDICAL_FINAL
        the_path = pa.csv_l.generate_output_file_path(data_set=D_209_MEDICAL, option=MEDICAL_FINAL)
        the_test_path = "/Users/robertfalast/PycharmProjects/PA_209/" + MEDICAL_CSV_FILE_LOCATION

        # run validation
        self.assertIsNotNone(the_path)
        self.assertIsInstance(the_path, Path)
        self.assertEqual(the_path, PosixPath(the_test_path))

        # TEST #4
        # invoke method for D_209_MEDICAL, MEDICAL_FINAL
        the_path = pa.csv_l.generate_output_file_path(data_set=D_209_MEDICAL, option=MEDICAL_PREP)
        the_test_path = "/Users/robertfalast/PycharmProjects/PA_209/" + MEDICAL_PREP_FILE_LOCATION

        # run validation
        self.assertIsNotNone(the_path)
        self.assertIsInstance(the_path, Path)
        self.assertEqual(the_path, PosixPath(the_test_path))

    # negative tests for generate_output_file()
    def test_generate_output_file_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # check that we handle None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.csv_l.generate_output_file(data_set=None, option=None, the_dataframe=None)

            # validate the error message.
            self.assertTrue("data_set is None or incorrect type." in context.exception)

        # check that we handle "foo", None, None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.csv_l.generate_output_file(data_set="foo", option=None, the_dataframe=None)

            # validate the error message.
            self.assertTrue("data_set is None or incorrect type." in context.exception)

        # check that we handle self.CHURN_KEY, None, None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.csv_l.generate_output_file(data_set=self.CHURN_KEY,
                                          option=None,
                                          the_dataframe=None)

            # validate the error message.
            self.assertTrue("option is None or incorrect type." in context.exception)

        # check that we handle OUTPUT_OPTIONS[self.CHURN_KEY], "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.csv_l.generate_output_file(data_set=self.CHURN_KEY,
                                          option="foo",
                                          the_dataframe=None)

            # validate the error message.
            self.assertTrue("option is None or incorrect type." in context.exception)

        # check that we handle OUTPUT_OPTIONS[self.CHURN_KEY], OUTPUT_OPTIONS[self.CHURN_KEY][CHURN_FINAL], None
        with self.assertRaises(SyntaxError) as context:
            # invoke method()
            pa.csv_l.generate_output_file(data_set=self.CHURN_KEY,
                                          option=CHURN_FINAL,
                                          the_dataframe=None)

            # validate the error message.
            self.assertTrue("option is None or incorrect type." in context.exception)

    # test method for generate_output_file()
    def test_generate_output_file(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # check if the previous file is there, and if so, delete it.
        if exists("../../resources/Output/churn_cleaned.csv"):
            os.remove("../../resources/Output/churn_cleaned.csv")

        # run assertions
        self.assertFalse(exists("../../resources/Output/churn_cleaned.csv"))

        # generate output file for OUTPUT_OPTIONS[self.CHURN_KEY], OUTPUT_OPTIONS[self.CHURN_KEY][CHURN_FINAL], pa.df
        pa.csv_l.generate_output_file(data_set=self.CHURN_KEY,
                                      option=CHURN_FINAL,
                                      the_dataframe=pa.df)

        # run assertions
        self.assertTrue(exists("../../resources/Output/churn_cleaned.csv"))

        # check if the previous file is there, and if so, delete it.
        if exists("../../resources/Output/churn_prepared.csv"):
            os.remove("../../resources/Output/churn_prepared.csv")

        # run assertions
        self.assertFalse(exists("../../resources/Output/churn_prepared.csv"))

        # generate output file for self.CHURN_KEY, CHURN_PREP from
        # pa.analyzer.linear_model_storage[LM_FINAL_MODEL].encoded_df
        pa.csv_l.generate_output_file(data_set=self.CHURN_KEY,
                                      option=CHURN_PREP,
                                      the_dataframe=pa.analyzer.linear_model_storage[LM_FINAL_MODEL].encoded_df)

        # run assertions
        self.assertTrue(exists("../../resources/Output/churn_prepared.csv"))

