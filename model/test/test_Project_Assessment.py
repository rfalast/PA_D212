import logging
import os
import unittest

from os.path import exists
from pandas import DataFrame
from model.Project_Assessment import Project_Assessment
from model.analysis.PlotGenerator import PlotGenerator
from model.analysis.models.KNN_Model import KNN_Model
from model.analysis.models.KNN_Model_Result import KNN_Model_Result
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.Linear_Model_Result import Linear_Model_Result
from model.analysis.models.Logistic_Model import Logistic_Model
from model.analysis.models.Logistic_Model_Result import Logistic_Model_Result
from model.analysis.models.Random_Forest_Model import Random_Forest_Model
from model.analysis.models.Random_Forest_Model_Result import Random_Forest_Model_Result
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_212_CHURN, CHURN_CSV_FILE_LOCATION, \
    MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION, MT_KNN_CLASSIFICATION, MT_RF_REGRESSION
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.PCA_Analysis import PCA_Analysis
from model.constants.DatasetConstants import UNIQUE_COLUMN_VALUES, BOOL_COLUMN_KEY, BOOL_VALUE_KEY, BOOL_CLEANED_FLAG, \
    COLUMN_KEY, OBJECT_COLUMN_KEY
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_FINAL_MODEL
from model.constants.PlotConstants import GENERAL, PLOT_TYPE_CM_HEATMAP, PLOT_TYPE_HEATMAP, PLOT_TYPE_ROC_AUC
from model.constants.StatisticsConstants import DIST_NAME, DIST_PARAMETERS
from util.CommonUtils import are_tuples_the_same
from util.FileUtils import FileUtils


# test case for Project_Assessment class
class test_Project_Assessment(unittest.TestCase):
    # constants
    INVALID_DATASET_KEY = "foo"
    REPORT_BASE_NAME = "DATASET_ANALYSIS.xlsx"
    REPORT_DF_BASE_NAME = "DATAFRAME_OUTPUT.xlsx"
    LOG_FILE_LOCATION = "../../resources/Output/Output.log"
    OVERRIDDEN_LOCATION = "../../resources/Output/"
    DATASET_ANALYSIS_LOC = "../../resources/Output/DATASET_ANALYSIS.xlsx"
    BAD_LOG_FILE_LOCATION = "/foo/Output.log"
    VALID_COLUMN_DICT = {"PC1": "Age", "PC2": "Tenure", "PC3": "MonthlyCharge", "PC4": "Bandwidth_GB_Year",
                         "PC5": "Yearly_equip_failure", "PC6": "Contacts", "PC7": "Outage_sec_perweek",
                         "PC8": "Children", "PC9": "Income"}
    INVALID_COLUMN_DICT = {"PC1": "Age", "PC2": "Tenure", "PC3": "MonthlyCharge", "PC4": "Bandwidth_GB_Year",
                           "PC5": "Yearly_equip_failure", "PC6": "Contacts", "PC7": "Outage_sec_perweek", "PC8": "foo"}

    INVALID_BASE_DIR = "/users/foo"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"

    INVALID_FIELD_DICT = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                          "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                          "Item7": "Courteous_Exchange", "Item9": "Active_Listening"}

    VALID_FIELD_DICT = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                        "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                        "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    VALID_COLUMN_DROP_LIST_1 = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction']
    VALID_COLUMN_DROP_LIST_2 = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction',
                                'State', 'UID', 'County', 'Job', 'City']

    EXPECTED_NORM_COLUMNS = ['Population_z_score', 'TimeZone_z_score', 'Children_z_score',
                             'Age_z_score', 'Income_z_score', 'Outage_sec_perweek_z_score',
                             'Email_z_score', 'Contacts_z_score', 'Yearly_equip_failure_z_score',
                             'Tenure_z_score', 'MonthlyCharge_z_score', 'Bandwidth_GB_Year_z_score',
                             'Timely_Response_z_score', 'Timely_Fixes_z_score',
                             'Timely_Replacements_z_score', 'Reliability_z_score', 'Options_z_score',
                             'Respectful_Response_z_score', 'Courteous_Exchange_z_score',
                             'Active_Listening_z_score']

    CHURN_KEY = D_212_CHURN

    # negative tests for init() method
    def test_init_negative(self):
        # make sure we handle argument None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            Project_Assessment(None, None)

            # validate the error message.
            self.assertTrue("The base_directory argument was None or invalid type." in context.exception)

        # make sure we handle argument BAD_LOG_FILE_LOCATION, None
        with self.assertRaises(NotADirectoryError) as context:
            # invoke method
            Project_Assessment(self.BAD_LOG_FILE_LOCATION, None)

            # validate the error message.
            self.assertTrue(f"base_directory does not exist." in context.exception)

        # make sure we handle argument VALID_BASE_DIR, 1.0
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            Project_Assessment(self.VALID_BASE_DIR, 1.0)

            # validate the error message.
            self.assertTrue("The report_loc_override argument is invalid type." in context.exception)

        # make sure we handle argument VALID_BASE_DIR, None
        with self.assertRaises(NotADirectoryError) as context:
            # invoke method
            Project_Assessment(self.VALID_BASE_DIR, self.INVALID_BASE_DIR)

            # validate the error message.
            self.assertTrue("report_loc_override does not exist." in context.exception)

    # test init() method with no overrides
    def test_init(self):
        # basic instantiation
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # run validations
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)
        self.assertIsNotNone(pa.base_directory)
        self.assertEqual(pa.base_directory, self.VALID_BASE_DIR)

    # happy path test case for init() method
    def test_init_overriden(self):
        # create an instance of the Project_Assessment class
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        # run validations
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)
        self.assertIsNotNone(pa.dataset_keys)
        self.assertTrue(exists(self.OVERRIDDEN_LOCATION))
        self.assertEqual(pa.r_gen.output_path, self.OVERRIDDEN_LOCATION + self.REPORT_BASE_NAME)
        self.assertEqual(pa.r_gen.dataframe_path, self.OVERRIDDEN_LOCATION + self.REPORT_DF_BASE_NAME)
        logging.debug("logging should work.")

        # create an instance only using the argument base_directory
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # run assertions
        self.assertEqual(pa.base_directory, self.VALID_BASE_DIR)

    # negative test cases for load_dataset()
    def test_load_dataset_negative(self):
        # create an instance of the Project_Assessment class
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # make sure we throw an TypeError for None argument
        with self.assertRaises(TypeError) as context:
            # None argument for load_dataset()
            pa.load_dataset(None)

            # validate the error message.
            self.assertTrue("The dataset_name_key as None or the incorrect type." in context.exception)

        # make sure we throw an SyntaxError for None argument
        with self.assertRaises(SyntaxError) as context:
            # None argument for load_dataset()
            pa.load_dataset(self.INVALID_DATASET_KEY)

            # validate the error message.
            self.assertTrue("The dataset_name_key as None or the incorrect type." in context.exception)

    # test method for load_dataset(
    def test_load_data_set(self):
        # create an instance of the Project_Assessment class
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # load the dataframe
        pa.load_dataset(self.CHURN_KEY)

        # run assertions
        self.assertIsNotNone(pa.df)
        self.assertIsInstance(pa.df, DataFrame)
        self.assertEqual(pa.CSV_FILE_LOCATION, CHURN_CSV_FILE_LOCATION)

        # log that this test case ran
        logging.debug("test case ran.")

    # negative test method for analyze_dataset()
    def test_analyze_dataset_negative(self):
        # instantiate a PA
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # make sure we can handle None
        with self.assertRaises(Exception) as context:
            # None argument for load_dataset()
            pa.analyze_dataset(None)

            # validate the error message.
            self.assertTrue("The analysis_type was None or invalid type." in context)

        # make sure we can handle bad option "foo"
        with self.assertRaises(Exception) as context:
            # None argument for load_dataset()
            pa.analyze_dataset("foo")

            # validate the error message.
            self.assertTrue(f"The analysis_type [foo] is unknown." in context)

        # we're going to call analyze_dataset() without having loaded it first
        with self.assertRaises(Exception) as context:
            # None argument for load_dataset()
            pa.analyze_dataset("FULL")

            # validate the error message.
            self.assertTrue("Dataframe was not populated." in context)

    # test method for analyze_dataset()
    def test_analyze_dataset(self):
        # instantiate a PA
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # load the dataset
        pa.load_dataset(self.CHURN_KEY)

        # analyze the FULL dataset
        pa.analyze_dataset("FULL")

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)
        self.assertTrue("Churn" in pa.df)
        self.assertTrue("Churn" in pa.analyzer.the_df)

        # instantiate a PA
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # load the dataset
        pa.load_dataset(self.CHURN_KEY)

        # analyze the INITIAL dataset
        pa.analyze_dataset("INITIAL")
        self.assertTrue("Churn" in pa.df)
        self.assertTrue("Churn" in pa.analyzer.the_df)

        # get the dataframe
        self.assertTrue("Churn" in list(pa.analyzer.storage[UNIQUE_COLUMN_VALUES].keys()))

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)

    # test method for analyze_dataset()
    def test_analyze_dataset_full(self):
        # instantiate a PA
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # load the dataset
        pa.load_dataset(self.CHURN_KEY)

        # analyze the FULL dataset
        pa.analyze_dataset("FULL")

        # run assertions
        self.assertTrue("Churn" in pa.df)
        self.assertTrue("Churn" in pa.analyzer.the_df)
        self.assertTrue(pa.analyzer.contains_boolean())
        self.assertTrue(BOOL_VALUE_KEY in pa.analyzer.storage)
        self.assertTrue("Churn" in pa.analyzer.storage[BOOL_VALUE_KEY])
        self.assertTrue(pa.analyzer.storage[BOOL_CLEANED_FLAG])
        self.assertTrue("Churn" in pa.analyzer.storage[COLUMN_KEY])
        self.assertFalse("Churn" in pa.analyzer.storage[OBJECT_COLUMN_KEY])
        self.assertTrue("Churn" in pa.analyzer.storage[BOOL_COLUMN_KEY])
        self.assertTrue("Churn" in pa.analyzer.storage[UNIQUE_COLUMN_VALUES])

        # update the column names
        pa.update_column_names(self.VALID_FIELD_DICT)

        # run assertions
        self.assertTrue(pa.analyzer.storage[BOOL_CLEANED_FLAG])

        # re-analyze dataset
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # run assertions
        self.assertTrue("Churn" in pa.df)
        self.assertTrue("Churn" in pa.analyzer.the_df)
        self.assertTrue(pa.analyzer.contains_boolean())
        self.assertTrue(BOOL_CLEANED_FLAG in pa.analyzer.storage)
        self.assertTrue(pa.analyzer.storage[BOOL_CLEANED_FLAG])
        self.assertTrue("Churn" in pa.analyzer.storage[COLUMN_KEY])
        self.assertFalse("Churn" in pa.analyzer.storage[OBJECT_COLUMN_KEY])
        self.assertTrue("Churn" in pa.analyzer.storage[BOOL_COLUMN_KEY])
        self.assertTrue("Churn" in pa.analyzer.storage[UNIQUE_COLUMN_VALUES])

    # negative tests for generate_output_report() method
    def test_generate_output_report_negative(self):
        # this test case was developed due to a bug in the code.  Leave the check in even if it
        # doesn't make sense.

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        # run assertions
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # force the report to none.
        pa.r_gen = None

        # generate the output report
        pa.generate_output_report(the_model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(pa.r_gen)

    # test the generate_output_report() method for LINEAR model
    def test_generate_output_report_linear_model(self):
        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        if exists("../../resources/Output/churn_cleaned.csv"):
            os.remove("../../resources/Output/churn_cleaned.csv")

        if exists("../../resources/Output/churn_prepared.csv"):
            os.remove("../../resources/Output/churn_prepared.csv")

        # delete HEAT_MAP_1.png, the correlation heatmap
        if exists("../../resources/Output/HEAT_MAP_1.png"):
            os.remove("../../resources/Output/HEAT_MAP_1.png")

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        # run assertions
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Linear_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Linear_Model)

        # run assertions that the files don't exist
        self.assertFalse(exists("../../resources/Output/churn_prepared.csv"))
        self.assertFalse(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertFalse(exists("../../resources/Output/HEAT_MAP_1.png"))
        self.assertFalse(exists(self.DATASET_ANALYSIS_LOC))

        # generate the output report with the_model_type=None
        pa.generate_output_report(the_model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(pa.s_gen.relevant_corr_storage)
        self.assertEqual(len(pa.s_gen.relevant_corr_storage), 5)
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))
        self.assertTrue(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../resources/Output/churn_prepared.csv"))
        self.assertTrue(exists("../../resources/Output/HEAT_MAP_1.png"))

        # get the plot generator
        the_plot_generator = pa.p_gen

        # run assertions on the PlotGenerator
        self.assertIsNotNone(the_plot_generator)
        self.assertIsInstance(the_plot_generator, PlotGenerator)
        self.assertTrue(GENERAL in the_plot_generator.plot_storage)
        self.assertTrue(PLOT_TYPE_HEATMAP in the_plot_generator.plot_storage[GENERAL])
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_HEATMAP],
                         "../../resources/Output/HEAT_MAP_1.png")

    # test the generate_output_report() method for LOGISTIC model
    def test_generate_output_report_logistic_model(self):
        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # delete churn_cleaned.csv
        if exists("../../resources/Output/churn_cleaned.csv"):
            os.remove("../../resources/Output/churn_cleaned.csv")

        # delete churn_prepared.csv
        if exists("../../resources/Output/churn_prepared.csv"):
            os.remove("../../resources/Output/churn_prepared.csv")

        # delete HEAT_MAP_1.png, the correlation heatmap
        if exists("../../resources/Output/HEAT_MAP_1.png"):
            os.remove("../../resources/Output/HEAT_MAP_1.png")

        # delete HEAT_MAP_2.png, the confusion matrix heatmap
        if exists("../../resources/Output/HEAT_MAP_2.png"):
            os.remove("../../resources/Output/HEAT_MAP_2.png")

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        # run assertions
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Logistic_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Logistic_Model)

        # run assertions that the files don't exist
        self.assertFalse(exists("../../resources/Output/churn_prepared.csv"))
        self.assertFalse(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertFalse(exists("../../resources/Output/HEAT_MAP_1.png"))
        self.assertFalse(exists("../../resources/Output/HEAT_MAP_2.png"))
        self.assertFalse(exists(self.DATASET_ANALYSIS_LOC))

        # generate the output report with the_model_type argument
        pa.generate_output_report(the_model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(pa.s_gen.relevant_corr_storage)
        self.assertEqual(len(pa.s_gen.relevant_corr_storage), 5)
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))
        self.assertTrue(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../resources/Output/churn_prepared.csv"))
        self.assertTrue(exists("../../resources/Output/HEAT_MAP_1.png"))
        self.assertTrue(exists("../../resources/Output/HEAT_MAP_2.png"))

        # get the plot generator
        the_plot_generator = pa.p_gen

        # run assertions on the PlotGenerator
        self.assertIsNotNone(the_plot_generator)
        self.assertIsInstance(the_plot_generator, PlotGenerator)
        self.assertTrue(GENERAL in the_plot_generator.plot_storage)
        self.assertTrue(PLOT_TYPE_HEATMAP in the_plot_generator.plot_storage[GENERAL])
        self.assertTrue(PLOT_TYPE_CM_HEATMAP in the_plot_generator.plot_storage[GENERAL])
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_HEATMAP],
                         "../../resources/Output/HEAT_MAP_1.png")
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP],
                         "../../resources/Output/HEAT_MAP_2.png")

    # test the generate_output_report() method for KNN model
    def test_generate_output_report_knn_model(self):
        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # delete churn_cleaned.csv
        if exists("../../resources/Output/churn_cleaned.csv"):
            os.remove("../../resources/Output/churn_cleaned.csv")

        if exists("../../resources/Output/churn_X_train.csv"):
            os.remove("../../resources/Output/churn_X_train.csv")

        if exists("../../resources/Output/churn_X_test.csv"):
            os.remove("../../resources/Output/churn_X_test.csv")

        if exists("../../resources/Output/churn_Y_train.csv"):
            os.remove("../../resources/Output/churn_Y_train.csv")

        if exists("../../resources/Output/churn_Y_test.csv"):
            os.remove("../../resources/Output/churn_Y_test.csv")

        # delete HEAT_MAP_1.png, the correlation heatmap
        if exists("../../resources/Output/HEAT_MAP_1.png"):
            os.remove("../../resources/Output/HEAT_MAP_1.png")

        # delete HEAT_MAP_2.png, the confusion matrix heatmap
        if exists("../../resources/Output/HEAT_MAP_2.png"):
            os.remove("../../resources/Output/HEAT_MAP_2.png")

        # delete PLOT_TYPE_ROC_AUC.png, the ROC/AUC plot
        if exists("../../resources/Output/ROC_AUC_PLOT_1.png"):
            os.remove("../../resources/Output/ROC_AUC_PLOT_1.png")

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        # run assertions
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], KNN_Model)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # run assertions that the files don't exist
        self.assertFalse(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertFalse(exists("../../resources/Output/churn_X_train.csv"))
        self.assertFalse(exists("../../resources/Output/churn_X_test.csv"))
        self.assertFalse(exists("../../resources/Output/churn_Y_train.csv"))
        self.assertFalse(exists("../../resources/Output/churn_Y_test.csv"))
        self.assertFalse(exists("../../resources/Output/HEAT_MAP_1.png"))
        self.assertFalse(exists("../../resources/Output/HEAT_MAP_2.png"))
        self.assertFalse(exists("../../resources/Output/ROC_AUC_PLOT_1.png"))
        self.assertFalse(exists(self.DATASET_ANALYSIS_LOC))

        # generate the output report with the_model_type argument
        pa.generate_output_report(the_model_type=MT_KNN_CLASSIFICATION)

        # run assertions
        self.assertIsNotNone(pa.s_gen.relevant_corr_storage)
        self.assertEqual(len(pa.s_gen.relevant_corr_storage), 5)
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))
        self.assertTrue(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../resources/Output/churn_X_train.csv"))
        self.assertTrue(exists("../../resources/Output/churn_X_test.csv"))
        self.assertTrue(exists("../../resources/Output/churn_Y_train.csv"))
        self.assertTrue(exists("../../resources/Output/churn_Y_test.csv"))
        self.assertTrue(exists("../../resources/Output/HEAT_MAP_1.png"))
        self.assertTrue(exists("../../resources/Output/HEAT_MAP_2.png"))
        self.assertTrue(exists("../../resources/Output/ROC_AUC_PLOT_1.png"))

        # get the plot generator
        the_plot_generator = pa.p_gen

        # run assertions on the PlotGenerator
        self.assertIsNotNone(the_plot_generator)
        self.assertIsInstance(the_plot_generator, PlotGenerator)
        self.assertTrue(GENERAL in the_plot_generator.plot_storage)
        self.assertTrue(PLOT_TYPE_HEATMAP in the_plot_generator.plot_storage[GENERAL])
        self.assertTrue(PLOT_TYPE_CM_HEATMAP in the_plot_generator.plot_storage[GENERAL])
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_HEATMAP],
                         "../../resources/Output/HEAT_MAP_1.png")
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP],
                         "../../resources/Output/HEAT_MAP_2.png")
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_ROC_AUC],
                         "../../resources/Output/ROC_AUC_PLOT_1.png")

    # test the generate_output_report() method for Random Forest
    def test_generate_output_report_random_forest_model(self):
        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # delete churn_cleaned.csv
        if exists("../../resources/Output/churn_cleaned.csv"):
            os.remove("../../resources/Output/churn_cleaned.csv")

        if exists("../../resources/Output/churn_prepared.csv"):
            os.remove("../../resources/Output/churn_prepared.csv")

        if exists("../../resources/Output/churn_X_train.csv"):
            os.remove("../../resources/Output/churn_X_train.csv")

        if exists("../../resources/Output/churn_X_test.csv"):
            os.remove("../../resources/Output/churn_X_test.csv")

        if exists("../../resources/Output/churn_Y_train.csv"):
            os.remove("../../resources/Output/churn_Y_train.csv")

        if exists("../../resources/Output/churn_Y_test.csv"):
            os.remove("../../resources/Output/churn_Y_test.csv")

        # delete HEAT_MAP_1.png, the correlation heatmap
        if exists("../../resources/Output/HEAT_MAP_1.png"):
            os.remove("../../resources/Output/HEAT_MAP_1.png")

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        # run assertions
        self.assertIsNotNone(pa)
        self.assertIsInstance(pa, Project_Assessment)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # run assertions
        self.assertIsNotNone(pa.analyzer)
        self.assertIsInstance(pa.analyzer, DatasetAnalyzer)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Bandwidth_GB_Year', model_type=MT_RF_REGRESSION, max_p_value=0.001)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Random_Forest_Model)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # run assertions that the files don't exist
        self.assertFalse(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertFalse(exists("../../resources/Output/churn_prepared.csv"))
        self.assertFalse(exists("../../resources/Output/churn_X_train.csv"))
        self.assertFalse(exists("../../resources/Output/churn_X_test.csv"))
        self.assertFalse(exists("../../resources/Output/churn_Y_train.csv"))
        self.assertFalse(exists("../../resources/Output/churn_Y_test.csv"))
        self.assertFalse(exists("../../resources/Output/HEAT_MAP_1.png"))
        self.assertFalse(exists(self.DATASET_ANALYSIS_LOC))

        # generate the output report with the_model_type argument
        pa.generate_output_report(the_model_type=MT_RF_REGRESSION)

        # run assertions
        self.assertIsNotNone(pa.s_gen.relevant_corr_storage)
        self.assertEqual(len(pa.s_gen.relevant_corr_storage), 5)
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))
        self.assertTrue(exists("../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../resources/Output/churn_X_train.csv"))
        self.assertTrue(exists("../../resources/Output/churn_X_test.csv"))
        self.assertTrue(exists("../../resources/Output/churn_Y_train.csv"))
        self.assertTrue(exists("../../resources/Output/churn_Y_test.csv"))
        self.assertTrue(exists("../../resources/Output/HEAT_MAP_1.png"))

        # get the plot generator
        the_plot_generator = pa.p_gen

        # run assertions on the PlotGenerator
        self.assertIsNotNone(the_plot_generator)
        self.assertIsInstance(the_plot_generator, PlotGenerator)
        self.assertTrue(GENERAL in the_plot_generator.plot_storage)
        self.assertTrue(PLOT_TYPE_HEATMAP in the_plot_generator.plot_storage[GENERAL])
        self.assertEqual(the_plot_generator.plot_storage[GENERAL][PLOT_TYPE_HEATMAP],
                         "../../resources/Output/HEAT_MAP_1.png")

    # negative test cases for perform_pca_on_dataset()
    def test_perform_pca_on_dataset_negative(self):
        # create an instance of the Project_Assessment class
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR)

        # load the dataset
        pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        pa.analyze_dataset("FULL")

        # create the column_dict
        column_dict = {}

        # make sure we throw an TypeError for column_dict=None
        with self.assertRaises(TypeError) as context:
            # invoke perform_pca_on_dataset()
            pa.perform_pca_on_dataset(column_dict=None)

            # validate the error message.
            self.assertTrue("column_dict is None or wrong type." in context)

        # make sure we throw an ValueError for column_dict=column_dict
        with self.assertRaises(ValueError) as context:
            # invoke perform_pca_on_dataset()
            pa.perform_pca_on_dataset(column_dict=column_dict)

            # validate the error message.
            self.assertTrue("column_dict is None or wrong type." in context)

        # make sure we throw an ValueError for column_dict=self.INVALID_COLUMN_DICT
        with self.assertRaises(ValueError) as context:
            # invoke perform_pca_on_dataset()
            pa.perform_pca_on_dataset(column_dict=self.INVALID_COLUMN_DICT)

            # validate the error message.
            self.assertTrue("column [foo] not present on dataframe." in context)

        # make sure we throw an TypeError for column_dict=self.VALID_COLUMN_DICT, column_drop_list={}
        with self.assertRaises(TypeError) as context:
            # invoke perform_pca_on_dataset()
            pa.perform_pca_on_dataset(column_dict=self.INVALID_COLUMN_DICT, column_drop_list={})

            # validate the error message.
            self.assertTrue("column_drop_list is incorrect type." in context)

    # test the perform_pca_on_dataset() method
    def test_perform_pca_on_dataset(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # invoke generate output report, or else some internal variables are not correct.
        pa.generate_output_report(the_model_type=MT_LINEAR_REGRESSION)

        # perform pca
        pa.perform_pca_on_dataset(self.VALID_COLUMN_DICT)

        # run assertions
        self.assertIsNotNone(pa.pca_analysis)
        self.assertIsInstance(pa.pca_analysis, PCA_Analysis)

    # negative tests for update_base_directory() method
    def test_update_base_directory_negative(self):  # create a PA
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # check for SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke perform_pca_on_dataset()
            the_pa.update_base_directory(None)

            # validate the error message.
            self.assertTrue("The path was None or incorrect type." in context.exception)

        # check for SyntaxError for None
        with self.assertRaises(NotADirectoryError) as context:
            # invoke perform_pca_on_dataset()
            the_pa.update_base_directory(self.INVALID_BASE_DIR)

            # validate the error message.
            self.assertTrue(f"the path [{self.INVALID_BASE_DIR}] does not exist." in context.exception)

    # test method for add_base_directory() method
    def test_add_base_directory(self):
        # create a PA
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # create a file util
        file_util = FileUtils()

        # get the path
        the_path = file_util.get_base_directory()

        # invoke method
        the_pa.update_base_directory(the_path=the_path)

        # run assertions
        self.assertEqual(the_pa.base_directory, file_util.get_base_directory() + "/")

    # negative tests for update_column_name() method
    def test_update_column_name_negative(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # check for SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke perform_pca_on_dataset()
            the_pa.update_column_names(None)

            # validate the error message.
            self.assertTrue("The variable_dict was None or incorrect type." in context.exception)

        # check for ValueError for None
        with self.assertRaises(ValueError) as context:
            # invoke perform_pca_on_dataset()
            the_pa.update_column_names(self.INVALID_FIELD_DICT)

            # validate the error message.
            self.assertTrue("The variable_dict was None or incorrect type." in context.exception)

    # test method for update_column_names() method
    def test_update_column_names(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")
        self.assertTrue("Churn" in the_pa.df)
        self.assertTrue("Churn" in the_pa.analyzer.the_df)

        # invoke the function
        the_pa.update_column_names(self.VALID_FIELD_DICT)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")
        self.assertTrue("Churn" in the_pa.df)
        self.assertTrue("Churn" in the_pa.analyzer.the_df)

        # get the dataframe
        the_df = the_pa.analyzer.the_df

        # run assertions
        for key in list(self.VALID_FIELD_DICT.keys()):
            self.assertFalse(key in the_df)
            self.assertTrue(self.VALID_FIELD_DICT[key] in the_df)

        self.assertEqual(len(the_pa.analyzer.the_df), 10000)
        self.assertEqual(len(the_pa.analyzer.the_df.columns), 49)

    # negative tests for drop_column_from_dataset() method
    def test_drop_column_from_dataset_negative(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # check for SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke drop_column_from_dataset()
            the_pa.drop_column_from_dataset(None)

            # validate the error message.
            self.assertTrue("The columns_to_drop_list was None or incorrect type." in context.exception)

    # test method for drop_column_from_dataset() method
    def test_drop_column_from_dataset(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # invoke the function
        the_pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_1)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # get the dataframe
        the_pa_df = the_pa.df
        the_dsa_df = the_pa.analyzer.the_df

        # run assertions
        for column in self.VALID_COLUMN_DROP_LIST_1:
            self.assertFalse(column in the_pa_df.columns)
            self.assertFalse(column in the_dsa_df.columns)

        self.assertEqual(len(the_pa_df), 10000)
        self.assertEqual(len(the_pa_df.columns), 44)
        self.assertEqual(len(the_dsa_df), 10000)
        self.assertEqual(len(the_dsa_df.columns), 44)

    # test method for drop_column_from_dataset() with emtpy list
    def test_drop_column_from_dataset_empty_list(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # get the current dataset columns
        current_columns = the_pa.df.columns.to_list()

        # run validations
        self.assertIsNotNone(current_columns)
        self.assertIsInstance(current_columns, list)
        self.assertEqual(len(current_columns), 49)

        # invoke the function
        the_pa.drop_column_from_dataset([])

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # get the dataframe
        the_pa_df = the_pa.df
        the_dsa_df = the_pa.analyzer.the_df

        # run validations
        self.assertIsNotNone(the_pa_df)
        self.assertIsInstance(the_pa_df, DataFrame)
        self.assertEqual(len(the_pa_df), 10000)
        self.assertEqual(len(the_pa_df.columns.to_list()), 49)

        # run assertions
        for column in the_pa_df.columns.to_list():
            self.assertTrue(column in the_pa_df.columns)
            self.assertTrue(column in the_dsa_df.columns)
            self.assertTrue(column in current_columns)

    # negative test method for calculate_internal_statistics
    def test_calculate_internal_statistics_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            pa.calculate_internal_statistics(None)

            # validate the error message.
            self.assertTrue("the_level argument is None or incorrect type." in context.exception)

        # verify we handle -2.0
        with self.assertRaises(ValueError) as context:
            # invoke method
            pa.calculate_internal_statistics(the_level=-2.0)

            # validate the error message.
            self.assertTrue("the_level is not in [0,1]." in context.exception)

        # verify we handle 2.0
        with self.assertRaises(ValueError) as context:
            # invoke method
            pa.calculate_internal_statistics(the_level=2.0)

            # validate the error message.
            self.assertTrue("the_level is not in [0,1]." in context.exception)

    # test method for calculate_internal_statistics()
    def test_calculate_internal_statistics(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # update column names
        the_pa.update_column_names(self.VALID_FIELD_DICT)

        # drop unused columns
        the_pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # calculate the internal statistics
        the_pa.calculate_internal_statistics(the_level=0.5)

        # run assertions
        self.assertIsNotNone(the_pa.s_gen)
        self.assertEqual(len(the_pa.s_gen.relevant_corr_storage), 5)
        self.assertTrue(are_tuples_the_same(the_pa.s_gen.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))
        print(the_pa.s_gen.chi_square_results)
        self.assertEqual(len(the_pa.s_gen.chi_square_results), 171)

        # make sure that fit_theoretical_dist_to_all_columns() is called.
        self.assertIsNotNone(the_pa.s_gen.distribution)
        self.assertIsInstance(the_pa.s_gen.distribution, dict)
        self.assertEqual(len(the_pa.s_gen.distribution), 20)

        # get a little more specific
        self.assertTrue('Age' in the_pa.s_gen.distribution)
        self.assertTrue(DIST_NAME in the_pa.s_gen.distribution['Age'])
        self.assertTrue(DIST_PARAMETERS in the_pa.s_gen.distribution['Age'])
        self.assertEqual(the_pa.s_gen.distribution['Age'][DIST_NAME], "beta")
        self.assertEqual(the_pa.s_gen.distribution['Age'][DIST_PARAMETERS], (0.9507834161575333, 1.0049760804615488,
                                                                             17.999999999999996, 71.01646413274422))

    # test method for normalize_dataset() method
    def test_normalize_dataset(self):
        the_pa = Project_Assessment(self.VALID_BASE_DIR)

        # load the dataset
        the_pa.load_dataset(self.CHURN_KEY)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # update column names
        the_pa.update_column_names(self.VALID_FIELD_DICT)

        # drop unused columns
        the_pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)

        # analyze the dataset
        the_pa.analyze_dataset("FULL")

        # get the dataframe in the dsa
        the_df = the_pa.analyzer.the_df

        # run some validations against the dataframe in the dsa
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # invoke the method
        the_pa.normalize_dataset()

        # get the normalized dataframe in the dsa
        the_normalized_df = the_pa.analyzer.the_normal_df

        # run some validations against the normalized dataframe in the dsa
        self.assertIsNotNone(the_normalized_df)
        self.assertIsInstance(the_normalized_df, DataFrame)

        # we can only normalize columns that are INT64 and FLOAT
        self.assertEqual(len(the_normalized_df.columns), 20)

        # assert that our test data is also the same length
        self.assertEqual(len(the_normalized_df.columns), len(self.EXPECTED_NORM_COLUMNS))

        # loop over all the columns in EXPECTED_NORM_COLUMNS
        for the_column in self.EXPECTED_NORM_COLUMNS:
            # make sure the column is in
            self.assertTrue(the_column in the_normalized_df.columns)

        # validate the other direction
        for the_column in the_normalized_df:
            # make sure the column is in
            self.assertTrue(the_column in self.EXPECTED_NORM_COLUMNS)

    # negative tests for build_model()
    def test_build_model_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # check that we handle None, None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke drop_column_from_dataset()
            pa.build_model(the_target_column=None, model_type=None, max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # check that we handle "foo", None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke drop_column_from_dataset()
            pa.build_model(the_target_column="foo", model_type=None, max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle "Churn", None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=None, max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("model_type is None or incorrect value." in context.exception)

        # verify we handle "Churn", "foo", None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type="foo", max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("model_type is None or incorrect value." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=MT_LOGISTIC_REGRESSION,
                           max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_p_value was None or incorrect type." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, 1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=MT_LOGISTIC_REGRESSION,
                           max_p_value=1, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_p_value was None or incorrect type." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, the_max_vif=None, 2.0, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=MT_LOGISTIC_REGRESSION,
                           max_p_value=2.0, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_p_value must be in range (0,1)." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, -1, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=MT_LOGISTIC_REGRESSION,
                           max_p_value=-1.0, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_p_value must be in range (0,1)." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, 0.5, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=MT_LOGISTIC_REGRESSION,
                           max_p_value=0.5, the_max_vif=None)

            # validate the error message.
            self.assertTrue("the_max_vif was None or incorrect type." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, 0.5, 0.5
        with self.assertRaises(ValueError) as context:
            # invoke the method
            pa.build_model(the_target_column="Churn", model_type=MT_LOGISTIC_REGRESSION,
                           max_p_value=0.5, the_max_vif=0.5)

            # validate the error message.
            self.assertTrue("the_max_vif must be > 1.0" in context.exception)

    # test method for build_model() looking at the INITIAL linear model
    def test_build_model_INITIAL_linear_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001, the_max_vif=10.0)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Linear_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Linear_Model)

        # get a reference to the initial linear_model
        the_linear_model = pa.analyzer.linear_model_storage[LM_INITIAL_MODEL]

        # get the Linear_Model_Result
        the_lm_result = the_linear_model.get_the_result()

        # run validations
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get the Linear_Model_Result
        the_result = the_linear_model.get_the_result()

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertEqual(len(results_df), 47)

        # make sure the Linear_Model_Result is stored on the Linear_Model
        self.assertIsNotNone(the_linear_model.get_the_result())
        self.assertIsInstance(the_linear_model.get_the_result(), Linear_Model_Result)

        # make sure the model result stored is the same.
        self.assertTrue(len(the_linear_model.get_the_result().get_results_dataframe()) > 0)
        self.assertFalse(the_linear_model.get_the_result().get_results_dataframe()['p-value'].max() < 0.001)
        self.assertEqual(len(the_linear_model.get_the_result().get_results_dataframe()), 47)

        # loop over columns of results_df and compare to the stored linear model result
        for the_column in results_df.columns:
            self.assertTrue(the_column in the_linear_model.get_the_result().get_results_dataframe().columns)

    # test method for build_model() looking at the INITIAL Logistic model
    def test_build_model_INITIAL_logistic_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001, the_max_vif=10.0)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Logistic_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Logistic_Model)

        # get a reference to the initial linear_model
        the_logistic_model = pa.analyzer.linear_model_storage[LM_INITIAL_MODEL]

        # get the Linear_Model_Result
        the_lm_result = the_logistic_model.get_the_result()

        # run validations
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # get the Linear_Model_Result
        the_result = the_logistic_model.get_the_result()

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertEqual(len(results_df), 47)

        # make sure the Linear_Model_Result is stored on the the_logistic_model
        self.assertIsNotNone(the_logistic_model.get_the_result())
        self.assertIsInstance(the_logistic_model.get_the_result(), Logistic_Model_Result)

        # make sure the model result stored is the same.
        self.assertTrue(len(the_logistic_model.get_the_result().get_results_dataframe()) > 0)
        self.assertFalse(the_logistic_model.get_the_result().get_results_dataframe()['p-value'].max() < 0.001)
        self.assertEqual(len(the_logistic_model.get_the_result().get_results_dataframe()), 47)

        # loop over columns of results_df and compare to the_logistic_model
        for the_column in results_df.columns:
            self.assertTrue(the_column in the_logistic_model.get_the_result().get_results_dataframe().columns)

    # test method for build_model() for INITIAL KNN model
    def test_build_model_INITIAL_knn_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_KNN_CLASSIFICATION,
                       max_p_value=0.001, the_max_vif=10.0, suppress_console=True)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], KNN_Model)
        self.assertTrue(LM_FINAL_MODEL not in pa.analyzer.linear_model_storage)

        # get a reference to the initial KNN model
        the_knn_model = pa.analyzer.linear_model_storage[LM_INITIAL_MODEL]

        # get the KNN_Model_Result
        the_result = the_knn_model.get_the_result()

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, KNN_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

    # test method for build_model() for INITIAL Random Forest model
    def test_build_model_INITIAL_random_forest_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Bandwidth_GB_Year', model_type=MT_RF_REGRESSION,
                       max_p_value=0.001, the_max_vif=10.0, suppress_console=True)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Random_Forest_Model)
        self.assertTrue(LM_FINAL_MODEL not in pa.analyzer.linear_model_storage)

        # get a reference to the initial KNN model
        the_knn_model = pa.analyzer.linear_model_storage[LM_INITIAL_MODEL]

        # get the KNN_Model_Result
        the_result = the_knn_model.get_the_result()

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)

    # test method for build_model() looking at the FINAL linear model (MT_LINEAR_REGRESSION)
    def test_build_model_FINAL_linear_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION,
                       max_p_value=0.001, the_max_vif=10.0)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Linear_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Linear_Model)

        # get a reference to the final linear_model
        the_linear_model = pa.analyzer.linear_model_storage[LM_FINAL_MODEL]

        # get the Linear_Model_Result
        the_lm_result = the_linear_model.get_the_result()

        # run validations
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get the Linear_Model_Result
        the_result = the_linear_model.get_the_result()

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertTrue(results_df['p-value'].max() < 0.001)
        self.assertEqual(len(results_df), 13)

        # make sure the Linear_Model_Result is stored on the Linear_Model
        self.assertIsNotNone(the_linear_model.get_the_result())
        self.assertIsInstance(the_linear_model.get_the_result(), Linear_Model_Result)

        # make sure the model result stored is the same.
        self.assertTrue(len(the_linear_model.get_the_result().get_results_dataframe()) > 0)
        self.assertTrue(the_linear_model.get_the_result().get_results_dataframe()['p-value'].max() < 0.001)
        self.assertEqual(len(the_linear_model.get_the_result().get_results_dataframe()), 13)

        # loop over columns of results_df and compare to the stored linear model result
        for the_column in results_df.columns:
            self.assertTrue(the_column in the_linear_model.get_the_result().get_results_dataframe().columns)

    # test method for build_model() looking at the FINAL logistic model (MT_LOGISTIC_REGRESSION)
    def test_build_model_FINAL_logistic_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001, the_max_vif=10.0)

        # run assertions on the linear model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Logistic_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Logistic_Model)

        # get a reference to the final linear_model
        the_logistic_model = pa.analyzer.linear_model_storage[LM_FINAL_MODEL]

        # get the Linear_Model_Result
        the_lm_result = the_logistic_model.get_the_result()

        # run validations
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # get the results_df
        results_df = the_lm_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertTrue(results_df['p-value'].max() < 0.001)
        self.assertEqual(len(results_df), 14)

        # make sure the Logistic_Model_Result is stored on the Linear_Model
        self.assertIsNotNone(the_logistic_model.get_the_result())
        self.assertIsInstance(the_logistic_model.get_the_result(), Logistic_Model_Result)

        # make sure the model result stored is the same.
        self.assertTrue(len(the_logistic_model.get_the_result().get_results_dataframe()) > 0)
        self.assertTrue(the_logistic_model.get_the_result().get_results_dataframe()['p-value'].max() < 0.001)
        self.assertEqual(len(the_logistic_model.get_the_result().get_results_dataframe()), 14)

        # loop over columns of results_df and compare to the stored linear model result
        for the_column in results_df.columns:
            self.assertTrue(the_column in the_logistic_model.get_the_result().get_results_dataframe().columns)

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

        # check that we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method()
            pa.clean_up_outliers(model_type=None, max_p_value=None)

            # validate the error message.
            self.assertTrue("model_type argument is None or incorrect type." in context.exception)

        # check that we handle "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.clean_up_outliers(model_type="foo", max_p_value=None)

            # validate the error message.
            self.assertTrue("model_type value is unknown." in context.exception)

        # check that we handle "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke method()
            pa.clean_up_outliers(model_type="foo", max_p_value=None)

            # validate the error message.
            self.assertTrue("model_type value is unknown." in context.exception)

        # verify we handle MT_LINEAR_REGRESSION, -1.1
        with self.assertRaises(ValueError) as context:
            # invoke method
            pa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=-1.1)

            # validate the error message.
            self.assertTrue("max_p_value argument is None or incorrect type." in context.exception)

        # verify we handle 1.1
        with self.assertRaises(ValueError) as context:
            # invoke method
            pa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=1.1)

            # validate the error message.
            self.assertTrue("max_p_value is not in (0,1)." in context.exception)

    # test method for clean_up_outliers()
    def test_clean_up_outliers_linear_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDDEN_LOCATION)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.VALID_FIELD_DICT)
        pa.drop_column_from_dataset(self.VALID_COLUMN_DROP_LIST_2)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

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
        pa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get the internal dataframe
        the_df = pa.df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9266)  # used to be 9407
        self.assertEqual(len(the_df.columns.to_list()), 39)

        # run assertions that the analyzer dataframe looks the same

        # get the internal dataframe
        the_df = pa.analyzer.the_df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9266)
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # test method for clean_up_outliers()
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
        pa.clean_up_outliers(model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # get the internal dataframe
        the_df = pa.df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9266)     # used to be 9407
        self.assertEqual(len(the_df.columns.to_list()), 39)

        # run assertions that the analyzer dataframe looks the same

        # get the internal dataframe
        the_df = pa.analyzer.the_df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9266)     # used to be 9407
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # test method for clean_up_outliers() for KNN
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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # get the internal dataframe
        the_df = pa.df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9582)
        self.assertEqual(len(the_df.columns.to_list()), 39)

        # run assertions that the analyzer dataframe looks the same

        # get the internal dataframe
        the_df = pa.analyzer.the_df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9582)
        self.assertEqual(len(the_df.columns.to_list()), 39)

    # test method for clean_up_outliers() for a RANDOM FOREST
    def test_clean_up_outliers_random_forest_model(self):
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
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)

        # get the internal dataframe
        the_df = pa.df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9582)
        self.assertEqual(len(the_df.columns.to_list()), 39)

        # run assertions that the analyzer dataframe looks the same

        # get the internal dataframe
        the_df = pa.analyzer.the_df

        # run assertions that the dataframe currently looks like we think it does
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 9582)
        self.assertEqual(len(the_df.columns.to_list()), 39)
