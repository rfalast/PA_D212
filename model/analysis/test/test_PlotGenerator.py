import os
import unittest

import numpy as np
import pandas as pd
import seaborn as sns

from os.path import exists
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.regression.linear_model import RegressionResultsWrapper
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models.KNN_Model import KNN_Model
from model.analysis.models.KNN_Model_Result import KNN_Model_Result
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.Linear_Model_Result import Linear_Model_Result
from model.analysis.PCA_Analysis import PCA_Analysis
from model.analysis.PlotGenerator import PlotGenerator
from model.analysis.StatisticsGenerator import StatisticsGenerator
from model.analysis.models.Logistic_Model import Logistic_Model
from model.analysis.models.Logistic_Model_Result import Logistic_Model_Result
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_209_CHURN, ANALYZE_DATASET_INITIAL, \
    MT_LINEAR_REGRESSION, MT_LOGISTIC_REGRESSION, MT_KNN_CLASSIFICATION, MT_RF_REGRESSION
from model.constants.DatasetConstants import FLOAT64_COLUMN_KEY, INT64_COLUMN_KEY, OBJECT_COLUMN_KEY, BOOL_COLUMN_KEY
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_FINAL_MODEL
from model.constants.PlotConstants import PLOT_TYPE_HIST, PLOT_TYPE_BOX_PLOT, PLOT_TYPE_BAR_CHART, GENERAL, \
    PLOT_TYPE_HEATMAP, PLOT_TYPE_SCATTER_CHART, PLOT_TYPE_BIVARIATE_COUNT, PLOT_TYPE_JOINT_PLOT, PLOT_TYPE_Q_Q_PLOT, \
    PLOT_TYPE_STD_RESIDUAL, PLOT_TYPE_BIVARIATE_BOX, PLOT_TYPE_Q_Q_RESIDUAL_PLOT, PLOT_TYPE_LONG_ODDS, \
    PLOT_TYPE_CM_HEATMAP, PLOT_TYPE_ROC_AUC
from util.CSV_loader import CSV_Loader


# Test the PlotGenerator class.
class test_PlotGenerator(unittest.TestCase):
    # constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    DEFAULT_PATH = "resources/Output/Hist_#.png"
    OVERRIDE_PATH = "../../../resources/Output/"
    BAD_OVERRIDE_PATH = "/bar/foo/"
    FULL_OVERRIDE_PATH = "../../../resources/Output/Hist_#.png"
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    BAD_COLUMN = "foo"
    VALID_COLUMN_1 = "Children"
    VALID_COLUMN_2 = "Income"
    VALID_COLUMN_3 = "Population"
    VALID_COLUMN_4 = "Yearly_equip_failure"
    VALID_COLUMN_5 = "State"
    VALID_COLUMN_6 = "Area"
    VALID_COLUMN_7 = "Tenure"
    VALID_COLUMN_8 = "Churn"
    VALID_HIST_FILE_PATH_1 = "resources/Output/FLOAT_HIST_1.png"
    VALID_HIST_FILE_PATH_2 = "resources/Output/FLOAT_HIST_5.png"
    VALID_BOX_PLOT_FILE_PATH_1 = "resources/Output/FLOAT_BOX_PLOT_1.png"
    VALID_SCREE_PLOT_FILE_PATH_1 = "../../../resources/Output/SCREE_CHART_1.png"
    VALID_SCREE_PLOT_FILE_PATH_2 = "../../../resources/Output/SCREE_CHART_2.png"
    VALID_HEATMAP_FILE_PATH_1 = "../../../resources/Output/HEAT_MAP_1.png"
    VALID_HEATMAP_FILE_PATH_2 = "../../../resources/Output/HEAT_MAP_2.png"
    VALID_QQ_PLOT_FILE_PATH_1 = "resources/Output/FLOAT_Q_Q_PLOT_0.png"
    VALID_QQ_PLOT_FILE_PATH_2 = "resources/Output/FLOAT_Q_Q_PLOT_1.png"

    VALID_INITIAL_REPORT = "../../../resources/Output/INITIAL_DATASET_ANALYSIS.xlsx"

    VALID_OVERRIDEN_HIST_FILE_PATH_1 = "../../../resources/Output/FLOAT_HIST_1.png"
    VALID_OVERRIDEN_HIST_FILE_PATH_2 = "../../../resources/Output/FLOAT_HIST_5.png"
    VALID_OVERRIDEN_HIST_FILE_PATH_3 = "../../../resources/Output/INT_HIST_1.png"
    VALID_OVERRIDEN_HIST_FILE_PATH_4 = "../../../resources/Output/FLOAT_HIST_4.png"
    VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_1 = "../../../resources/Output/FLOAT_BOX_PLOT_1.png"
    VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_2 = "../../../resources/Output/FLOAT_BOX_PLOT_5.png"
    VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_3 = "../../../resources/Output/INT_BOX_PLOT_7.png"
    VALID_OVERRIDEN_BAR_CHART_FILE_PATH_1 = "../../../resources/Output/OBJECT_BAR_CHART_1.png"
    VALID_OVERRIDEN_BAR_CHART_FILE_PATH_2 = "../../../resources/Output/OBJECT_BAR_CHART_5.png"
    VALID_OVERRIDEN_BAR_CHART_FILE_PATH_3 = "../../../resources/Output/BOOL_BAR_CHART_6.png"

    CHILDREN_HIST_FILE_PATH = "../../../resources/Output/FLOAT_HIST_2.png"
    CHILDREN_HIST_FILE_PATH_2 = "../../../resources/Output/FLOAT_HIST_4.png"
    INCOME_HIST_FILE_PATH = "../../../resources/Output/FLOAT_HIST_4.png"
    INCOME_HIST_FILE_PATH_2 = "../../../resources/Output/FLOAT_HIST_2.png"
    POP_HIST_FILE_PATH = "../../../resources/Output/INT_HIST_1.png"
    YEF_HIST_FILE_PATH = "../../../resources/Output/INT_HIST_6.png"

    VALID_SCATTER_PLOT_1 = '../../../resources/Output/SCATTER_PLOT_0.png'
    VALID_SCATTER_PLOT_2 = '../../../resources/Output/SCATTER_PLOT_1.png'
    VALID_SCATTER_PLOT_3 = '../../../resources/Output/SCATTER_PLOT_2.png'
    VALID_SCATTER_PLOT_4 = '../../../resources/Output/SCATTER_PLOT_3.png'
    VALID_SCATTER_PLOT_5 = '../../../resources/Output/SCATTER_PLOT_4.png'
    VALID_SCATTER_PLOT_6 = '../../../resources/Output/SCATTER_PLOT_5.png'

    VALID_COUNT_PLOT_1 = '../../../resources/Output/BIVARIATE_COUNT_PLOT_0.png'
    VALID_COUNT_PLOT_2 = '../../../resources/Output/BIVARIATE_COUNT_PLOT_1.png'
    VALID_COUNT_PLOT_3 = '../../../resources/Output/BIVARIATE_COUNT_PLOT_2.png'
    VALID_COUNT_PLOT_4 = '../../../resources/Output/BIVARIATE_COUNT_PLOT_3.png'
    VALID_COUNT_PLOT_5 = '../../../resources/Output/BIVARIATE_COUNT_PLOT_4.png'
    VALID_COUNT_PLOT_6 = '../../../resources/Output/BIVARIATE_COUNT_PLOT_5.png'

    VALID_JOINT_PLOT_1 = '../../../resources/Output/JOINT_PLOT_1.png'
    VALID_JOINT_PLOT_2 = '../../../resources/Output/JOINT_PLOT_2.png'
    VALID_JOINT_PLOT_3 = '../../../resources/Output/JOINT_PLOT_3.png'
    VALID_JOINT_PLOT_4 = '../../../resources/Output/JOINT_PLOT_4.png'
    VALID_JOINT_PLOT_5 = '../../../resources/Output/JOINT_PLOT_5.png'
    VALID_JOINT_PLOT_6 = '../../../resources/Output/JOINT_PLOT_6.png'

    VALID_QQ_PLOT_1 = "../../../resources/Output/FLOAT_Q_Q_PLOT_0.png"
    VALID_QQ_PLOT_2 = "../../../resources/Output/FLOAT_Q_Q_PLOT_5.png"
    VALID_QQ_PLOT_3 = "../../../resources/Output/INT_Q_Q_PLOT_1.png"

    BASE_HIST_KEY = "BASE_HIST_KEY"

    VALID_DATA_TYPE = FLOAT64_COLUMN_KEY
    VALID_DATA_TYPE_2 = INT64_COLUMN_KEY
    VALID_DATA_TYPE_3 = OBJECT_COLUMN_KEY
    VALID_DATA_TYPE_4 = BOOL_COLUMN_KEY
    INVALID_DATA_TYPE = "foo"

    VALID_PLOT_TYPE_1 = PLOT_TYPE_HIST
    VALID_PLOT_TYPE_2 = PLOT_TYPE_BOX_PLOT
    VALID_PLOT_TYPE_3 = PLOT_TYPE_BAR_CHART
    VALID_PLOT_TYPE_4 = PLOT_TYPE_JOINT_PLOT
    VALID_PLOT_TYPE_5 = PLOT_TYPE_Q_Q_PLOT

    COLUMN_DICT = {"PC1": "Age", "PC2": "Tenure", "PC3": "MonthlyCharge", "PC4": "Bandwidth_GB_Year",
                   "PC5": "Yearly_equip_failure", "PC6": "Contacts", "PC7": "Outage_sec_perweek",
                   "PC8": "Children", "PC9": "Income"}

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_209_CHURN

    # test init method()
    def test_init_negative(self):
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

        # we're going to pass in None on the init method to get SyntaxError
        with self.assertRaises(SyntaxError) as context:
            # instantiate a PlotGenerator
            PlotGenerator(None)

            # validate the error message.
            self.assertTrue("The data_analyzer argument is None or incorrect type." in context.exception)

        # we're going to call PlotGenerator() with a bad path
        with self.assertRaises(FileNotFoundError) as context:
            # instantiate a ReportGenerator
            PlotGenerator(dsa, self.BAD_OVERRIDE_PATH)

            # validate the error message.
            self.assertTrue(f"override path [{self.BAD_OVERRIDE_PATH}] does not exist." in context.exception)

    # tests for the init method
    def test_init(self):
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

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa)

        # run assertions
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        self.assertIsNotNone(the_pg.data_analyzer)
        self.assertIsInstance(the_pg.data_analyzer, DatasetAnalyzer)

        self.assertIsNotNone(the_pg.data_type_map)
        self.assertIsInstance(the_pg.data_type_map, dict)
        self.assertTrue(len(the_pg.data_type_map) > 0)

        self.assertIsNotNone(the_pg.plot_type_list)
        self.assertIsInstance(the_pg.plot_type_list, list)
        self.assertTrue(len(the_pg.plot_type_list) > 0)

        self.assertIsNotNone(the_pg.plot_storage)
        self.assertIsInstance(the_pg.plot_storage, dict)
        self.assertTrue(len(the_pg.plot_storage) == 0)

    # test the override final report option for init() method
    def test_init_with_override(self):
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

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # run assertions
        self.assertIsNotNone(the_pg.output_path)
        self.assertEqual(the_pg.output_path, self.OVERRIDE_PATH)

    # negative test method for generate_hist()
    def test_generate_hist_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa)

        # make sure None, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(None, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # make sure INVALID_DATA_TYPE, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(self.INVALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(self.VALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("The column is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, BAD_COLUMN, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(self.VALID_DATA_TYPE, self.BAD_COLUMN, None)

            # validate the error message.
            self.assertTrue("The column was not found on list for the data_type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, None)

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, str
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, "foo")

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate that we handle VALID_DATA_TYPE, VALID_COLUMN_1, -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_hist(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, -1)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # test method for generate_hist()
    def test_generate_hist(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # run the method generate_hist() for FLOAT64
        the_pg.generate_hist(self.VALID_DATA_TYPE, self.VALID_COLUMN_7, 1)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1],
                         self.VALID_OVERRIDEN_HIST_FILE_PATH_1)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1]))

        # run second test, we need to make sure logic of how data is stored in the structure works
        # correctly.
        # run the method generate_hist() for FLOAT64
        the_pg.generate_hist(self.VALID_DATA_TYPE, self.VALID_COLUMN_2, 5)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1],
                         self.VALID_OVERRIDEN_HIST_FILE_PATH_2)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1]))

        # run the method generate_hist() for INT64 that has been moved from float with nan to int64 with nan.
        the_pg.generate_hist(self.VALID_DATA_TYPE_2, self.VALID_COLUMN_1, 1)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_1],
                         self.VALID_OVERRIDEN_HIST_FILE_PATH_3)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_1]))

    # tests for get_plot_file_name()
    def test_get_plot_file_name(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa)

        # run the method FOR FLOAT
        the_name = the_pg.get_plot_file_name(self.VALID_DATA_TYPE, self.VALID_PLOT_TYPE_1, 1)

        # run assertions
        self.assertIsNotNone(the_name)
        self.assertIsInstance(the_name, str)
        self.assertEqual(the_name, self.VALID_HIST_FILE_PATH_1)

        # run the method FOR FLOAT with a different count value
        the_name = the_pg.get_plot_file_name(self.VALID_DATA_TYPE, self.VALID_PLOT_TYPE_1, 5)

        # run assertions
        self.assertIsNotNone(the_name)
        self.assertIsInstance(the_name, str)
        self.assertEqual(the_name, self.VALID_HIST_FILE_PATH_2)

        # run the method FOR FLOAT with a different count value
        the_name = the_pg.get_plot_file_name(self.VALID_DATA_TYPE, self.VALID_PLOT_TYPE_2, 1)

        # run assertions
        self.assertIsNotNone(the_name)
        self.assertIsInstance(the_name, str)
        self.assertEqual(the_name, self.VALID_BOX_PLOT_FILE_PATH_1)

        # create a file name for a QQ Plot
        the_name = the_pg.get_plot_file_name(self.VALID_DATA_TYPE, PLOT_TYPE_Q_Q_PLOT, 0)

        # run assertions
        self.assertIsNotNone(the_name)
        self.assertIsInstance(the_name, str)
        self.assertEqual(the_name, self.VALID_QQ_PLOT_FILE_PATH_1)

    # negative tests for get_plot_file_name() method
    def test_get_plot_file_name_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa)

        # make sure None, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.get_plot_file_name(None, None, None)

            # validate the error message.
            self.assertTrue("The data_type argument is None or incorrect type." in context.exception)

        # make sure str, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.get_plot_file_name("foo", None, None)

            # validate the error message.
            self.assertTrue("The data_type argument is unknown." in context.exception)

        # make sure VALID_DATA_TYPE, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.get_plot_file_name(self.VALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("The chart_type argument is None or incorrect type." in context.exception)

        # make sure VALID_DATA_TYPE, str is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.get_plot_file_name(self.VALID_DATA_TYPE, "foo", None)

            # validate the error message.
            self.assertTrue("The chart_type argument is unknown." in context.exception)

        # make sure VALID_DATA_TYPE, str is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.get_plot_file_name(self.VALID_DATA_TYPE, self.VALID_PLOT_TYPE_1, None)

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # make sure VALID_DATA_TYPE, negative int is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.get_plot_file_name(self.VALID_DATA_TYPE, self.VALID_PLOT_TYPE_1, -4)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # negative tests for generate_all_dataset_plots()
    def test_generate_all_dataset_plots_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION,
                             max_p_value=0.01)  # remove outliers
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # make sure None is handled for None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_all_dataset_plots(statistics_generator=None, the_model_type=None, the_version=None)

        # validate the error message.
        self.assertEqual("the_version is None or unknown value.", context.exception.msg)

        # make sure None is handled for None, None, "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_all_dataset_plots(statistics_generator=None, the_model_type=None, the_version="foo")

        # validate the error message.
        self.assertEqual("the_version is None or unknown value.", context.exception.msg)

        # make sure None is handled for None, None, ANALYZE_DATASET_FULL
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_all_dataset_plots(statistics_generator=None, the_model_type=None,
                                              the_version=ANALYZE_DATASET_FULL)

        # validate the error message.
        self.assertEqual("the_model_type is None or unknown value.", context.exception.msg)

        # make sure None is handled for None, MT_KNN_CLASSIFICATION, ANALYZE_DATASET_FULL
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_all_dataset_plots(statistics_generator=None, the_model_type=MT_KNN_CLASSIFICATION,
                                              the_version=ANALYZE_DATASET_FULL)

        # validate the error message.
        self.assertEqual("statistics_generator is None or incorrect type.", context.exception.msg)

    # test method for generate_all_plots()
    def test_generate_all_plots_INITIAL(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)  # can be used interchangeably with either data set
        pa.analyze_dataset(ANALYZE_DATASET_INITIAL)  # initial dataset analysis

        # remove the INITIAL REPORT
        if exists(self.VALID_INITIAL_REPORT):
            os.remove(self.VALID_INITIAL_REPORT)

        # run the method
        pa.generate_initial_report()

        # run assertions
        self.assertTrue(exists(self.VALID_INITIAL_REPORT))

    # test case for generate_all_dataset_plots() method using a linear model
    def test_generate_all_plots_FULL_linear_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # we can't invoke clean_up_outliers() because it errors out for some inexplicable reason.
        # pa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=0.01)

        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # get a reference to the dataset analyzer
        dsa = pa.analyzer

        # call the method
        the_pg.generate_all_dataset_plots(statistics_generator=pa.s_gen,
                                          the_model_type=MT_LINEAR_REGRESSION,
                                          the_version=ANALYZE_DATASET_FULL)

        # run validations for FLOAT
        # first, check to see if we can find on the storage a valid HIST plot for "Tenure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2],
                         "../../../resources/Output/FLOAT_BOX_PLOT_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5],
                         "../../../resources/Output/FLOAT_Q_Q_PLOT_2.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1]))  # HIST
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2]))  # BOX
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5]))  # QQ PLOT

        # next, check to see if we can find on the storage a valid HIST plot for "Income"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1]))

        # run validations for INT
        # check to see if we can find on the storage a valid HIST plot for "Population"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1]))

        # next, check to see if we can find on the storage a valid HIST plot for "Yearly_equip_failure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_6.png")

        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1]))

        # run validations for BOOLEAN
        # check to see if we can find ton the storage a VALID BAR_CHART for "Churn"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3],
                         "../../../resources/Output/BOOL_BAR_CHART_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3]))

        # validate the plot storage for SCATTER plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART]), 5)

        # get the list of tuples
        corr_list = pa.s_gen.get_list_of_correlations()

        # run assertions
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[0]],
                         self.VALID_SCATTER_PLOT_1)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[1]],
                         self.VALID_SCATTER_PLOT_2)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[2]],
                         self.VALID_SCATTER_PLOT_3)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[3]],
                         self.VALID_SCATTER_PLOT_4)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[4]],
                         self.VALID_SCATTER_PLOT_5)

        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]), 171)

        # get the list of statistics_generator.get_list_of_variable_relationships():
        count_plot_list = pa.s_gen.get_chi_squared_results()

        # create reference to clean count_plot_list
        clean_count_plot_list = []

        # clean the list to only include OBJECT to OBJECT
        for the_tuple in count_plot_list:
            # make sure the tuple is only OBJECT to OBJECT
            if dsa.validate_field_type(the_tuple[0], OBJECT_COLUMN_KEY) \
                    and dsa.validate_field_type(the_tuple[1], OBJECT_COLUMN_KEY):
                # add to the clean_count_plot_list
                clean_count_plot_list.append(the_tuple)

        # run assertions on clean_count_plot_list
        self.assertEqual(len(clean_count_plot_list), 15)
        self.assertEqual(clean_count_plot_list[0], ('Area', 'Marital', 0.503361))
        self.assertEqual(clean_count_plot_list[1], ('Area', 'Gender', 0.738468))
        self.assertEqual(clean_count_plot_list[2], ('Area', 'Contract', 0.848445))
        self.assertEqual(clean_count_plot_list[3], ('Area', 'InternetService', 0.98648))
        self.assertEqual(clean_count_plot_list[4], ('Area', 'PaymentMethod', 0.640729))

        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[0]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_156.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[1]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_157.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[2]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_158.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[3]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_159.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[4]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_160.png")

        # validate joint plots
        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 190)

        # validate bi-variate count plots
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'Techie', 0.066722)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'Port_modem', 0.008157)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'Tablet', -0.002779)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'Phone', -0.026297)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'Multiple', 0.131771)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'OnlineSecurity', -0.01354)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'OnlineBackup', 0.050508)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'DeviceProtection', 0.056489)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'TechSupport', 0.018838)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'StreamingTV', 0.230151)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'StreamingMovies', 0.289262)])
        # self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][('Churn', 'PaperlessBilling', 0.00703)])

        # *****************************************************************
        # run validations for the residual plots
        # *****************************************************************

        # get the LM_INITIAL_MODEL from the dsa
        the_linear_model = dsa.get_model(the_type=LM_INITIAL_MODEL)

        # run assertions on the_linear_model
        self.assertIsNotNone(the_linear_model)
        self.assertIsInstance(the_linear_model, Linear_Model)
        self.assertEqual(len(the_linear_model.encoded_df.columns.to_list()), 48)

        # get the list of features from the_linear_model
        the_features_list = the_linear_model.get_encoded_variables(the_target_column='MonthlyCharge')

        # run additional assertions
        self.assertFalse('MonthlyCharge' in the_features_list)

        # MonthlyCharge would not have a residual plot

        # loop over the features
        for the_feature in the_features_list:
            self.assertIsNotNone(the_pg.plot_storage[the_feature][PLOT_TYPE_STD_RESIDUAL])

        # *****************************************************************
        # run validations for the bi-variate box plots
        # *****************************************************************

        # get the list of OBJECT columns
        the_object_column_list = dsa.storage[OBJECT_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_object_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # get the list of BOOLEAN columns
        the_boolean_column_list = dsa.storage[BOOL_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_boolean_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # *****************************************************************
        # run validations for long odds linear plots
        # *****************************************************************
        # get the initial_linear_model from the dsa
        the_linear_model = dsa.get_model(the_type=LM_INITIAL_MODEL)

        # get the list of features from the_linear_model
        the_features_list = the_linear_model.get_encoded_variables(the_target_column='MonthlyCharge')

        # run additional assertions
        self.assertFalse('MonthlyCharge' in the_features_list)

        # loop over the features
        for the_feature in the_features_list:
            self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_LONG_ODDS][the_feature])

        # run specific assertions
        self.assertTrue('Area_Suburban' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Area_Urban' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Married' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Never Married' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Separated' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Widowed' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Gender_Male' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Gender_Nonbinary' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Contract_One year' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Contract_Two Year' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('InternetService_Fiber Optic' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('InternetService_No response' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('PaymentMethod_Electronic Check' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('PaymentMethod_Mailed Check' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])

    # test case for generate_all_dataset_plots() method
    def test_generate_all_plots_FULL_logistic_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # we can't invoke clean_up_outliers() because it errors out for some inexplicable reason.
        # pa.clean_up_outliers(model_type=MT_LINEAR_REGRESSION, max_p_value=0.01)

        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # get a reference to the dataset analyzer
        dsa = pa.analyzer

        # call the method
        the_pg.generate_all_dataset_plots(statistics_generator=pa.s_gen,
                                          the_model_type=MT_LOGISTIC_REGRESSION,
                                          the_version=ANALYZE_DATASET_FULL)

        # run validations for FLOAT
        # first, check to see if we can find on the storage a valid HIST plot for "Tenure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2],
                         "../../../resources/Output/FLOAT_BOX_PLOT_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5],
                         "../../../resources/Output/FLOAT_Q_Q_PLOT_2.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1]))  # HIST
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2]))  # BOX
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5]))  # QQ PLOT

        # next, check to see if we can find on the storage a valid HIST plot for "Income"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1]))

        # run validations for INT
        # check to see if we can find on the storage a valid HIST plot for "Population"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1]))

        # next, check to see if we can find on the storage a valid HIST plot for "Yearly_equip_failure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_6.png")

        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1]))

        # run validations for BOOLEAN
        # check to see if we can find ton the storage a VALID BAR_CHART for "Churn"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3],
                         "../../../resources/Output/BOOL_BAR_CHART_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3]))

        # validate the plot storage for SCATTER plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART]), 5)

        # get the list of tuples
        corr_list = pa.s_gen.get_list_of_correlations()

        # run assertions
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[0]],
                         self.VALID_SCATTER_PLOT_1)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[1]],
                         self.VALID_SCATTER_PLOT_2)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[2]],
                         self.VALID_SCATTER_PLOT_3)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[3]],
                         self.VALID_SCATTER_PLOT_4)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[4]],
                         self.VALID_SCATTER_PLOT_5)

        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]), 171)

        # get the list of statistics_generator.get_list_of_variable_relationships():
        count_plot_list = pa.s_gen.get_chi_squared_results()

        # create reference to clean count_plot_list
        clean_count_plot_list = []

        # clean the list to only include OBJECT to OBJECT
        for the_tuple in count_plot_list:
            # make sure the tuple is only OBJECT to OBJECT
            if dsa.validate_field_type(the_tuple[0], OBJECT_COLUMN_KEY) \
                    and dsa.validate_field_type(the_tuple[1], OBJECT_COLUMN_KEY):
                # add to the clean_count_plot_list
                clean_count_plot_list.append(the_tuple)

        # run assertions on clean_count_plot_list
        self.assertEqual(len(clean_count_plot_list), 15)
        self.assertEqual(clean_count_plot_list[0], ('Area', 'Marital', 0.503361))
        self.assertEqual(clean_count_plot_list[1], ('Area', 'Gender', 0.738468))
        self.assertEqual(clean_count_plot_list[2], ('Area', 'Contract', 0.848445))
        self.assertEqual(clean_count_plot_list[3], ('Area', 'InternetService', 0.98648))
        self.assertEqual(clean_count_plot_list[4], ('Area', 'PaymentMethod', 0.640729))

        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[0]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_156.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[1]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_157.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[2]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_158.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[3]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_159.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[4]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_160.png")

        # validate joint plots
        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 190)

        # *****************************************************************
        # THERE ARE NO RESIDUAL PLOTS FOR LOGISITIC REGRESSION
        # *****************************************************************

        # get the LM_INITIAL_MODEL from the dsa
        the_linear_model = dsa.get_model(the_type=LM_INITIAL_MODEL)

        # get the list of features from the_linear_model
        the_features_list = the_linear_model.get_encoded_variables(the_target_column='Churn')

        # # loop over the features
        # for the_feature in the_features_list:
        #     self.assertIsNotNone(the_pg.plot_storage[the_feature][PLOT_TYPE_STD_RESIDUAL])

        # *****************************************************************
        # run validations for the bi-variate box plots
        # *****************************************************************

        # get the list of OBJECT columns
        the_object_column_list = dsa.storage[OBJECT_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_object_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # get the list of BOOLEAN columns
        the_boolean_column_list = dsa.storage[BOOL_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_boolean_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # *****************************************************************
        # run validations for long odds linear plots
        # *****************************************************************
        # get the initial_linear_model from the dsa
        the_linear_model = dsa.get_model(the_type=LM_INITIAL_MODEL)

        # get the list of features from the_linear_model
        the_features_list = the_linear_model.get_encoded_variables(the_target_column='Churn')

        # loop over the features
        for the_feature in the_features_list:
            self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_LONG_ODDS][the_feature])

        # run specific assertions
        self.assertTrue('Area_Suburban' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Area_Urban' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Married' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Never Married' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Separated' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Marital_Widowed' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Gender_Male' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Gender_Nonbinary' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Contract_One year' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('Contract_Two Year' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('InternetService_Fiber Optic' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('InternetService_No response' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('PaymentMethod_Electronic Check' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])
        self.assertTrue('PaymentMethod_Mailed Check' in the_pg.plot_storage[PLOT_TYPE_LONG_ODDS])

    # test case for generate_all_dataset_plots() method using a KNN model
    def test_generate_all_dataset_plots_FULL_knn_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
        pa.build_model(the_target_column='Churn', model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # get a reference to the dataset analyzer
        dsa = pa.analyzer

        the_pg.generate_all_dataset_plots(statistics_generator=pa.s_gen,
                                          the_model_type=MT_KNN_CLASSIFICATION,
                                          the_version=ANALYZE_DATASET_FULL)

        # run validations for FLOAT
        # first, check to see if we can find on the storage a valid HIST plot for "Tenure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2],
                         "../../../resources/Output/FLOAT_BOX_PLOT_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5],
                         "../../../resources/Output/FLOAT_Q_Q_PLOT_2.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1]))  # HIST
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2]))  # BOX
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5]))  # QQ PLOT

        # next, check to see if we can find on the storage a valid HIST plot for "Income"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1]))

        # run validations for INT
        # check to see if we can find on the storage a valid HIST plot for "Population"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1]))

        # next, check to see if we can find on the storage a valid HIST plot for "Yearly_equip_failure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_6.png")

        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1]))

        # run validations for BOOLEAN
        # check to see if we can find ton the storage a VALID BAR_CHART for "Churn"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3],
                         "../../../resources/Output/BOOL_BAR_CHART_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3]))

        # validate the plot storage for SCATTER plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART]), 5)

        # get the list of tuples
        corr_list = pa.s_gen.get_list_of_correlations()

        # run assertions
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[0]],
                         self.VALID_SCATTER_PLOT_1)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[1]],
                         self.VALID_SCATTER_PLOT_2)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[2]],
                         self.VALID_SCATTER_PLOT_3)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[3]],
                         self.VALID_SCATTER_PLOT_4)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[4]],
                         self.VALID_SCATTER_PLOT_5)

        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]), 171)

        # get the list of statistics_generator.get_list_of_variable_relationships():
        count_plot_list = pa.s_gen.get_chi_squared_results()

        # create reference to clean count_plot_list
        clean_count_plot_list = []

        # clean the list to only include OBJECT to OBJECT
        for the_tuple in count_plot_list:
            # make sure the tuple is only OBJECT to OBJECT
            if dsa.validate_field_type(the_tuple[0], OBJECT_COLUMN_KEY) \
                    and dsa.validate_field_type(the_tuple[1], OBJECT_COLUMN_KEY):
                # add to the clean_count_plot_list
                clean_count_plot_list.append(the_tuple)

        # run assertions on clean_count_plot_list
        self.assertEqual(len(clean_count_plot_list), 15)
        self.assertEqual(clean_count_plot_list[0], ('Area', 'Marital', 0.503361))
        self.assertEqual(clean_count_plot_list[1], ('Area', 'Gender', 0.738468))
        self.assertEqual(clean_count_plot_list[2], ('Area', 'Contract', 0.848445))
        self.assertEqual(clean_count_plot_list[3], ('Area', 'InternetService', 0.98648))
        self.assertEqual(clean_count_plot_list[4], ('Area', 'PaymentMethod', 0.640729))

        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[0]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_156.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[1]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_157.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[2]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_158.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[3]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_159.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[4]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_160.png")

        # validate joint plots
        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 190)

        # *****************************************************************
        # THERE ARE NO RESIDUAL PLOTS FOR KNN
        # *****************************************************************

        # get the LM_INITIAL_MODEL from the dsa
        the_initial_model = dsa.get_model(the_type=LM_INITIAL_MODEL)

        # get the list of features from the_linear_model
        the_features_list = the_initial_model.get_encoded_variables(the_target_column='Churn')

        # *****************************************************************
        # run validations for the bi-variate box plots
        # *****************************************************************

        # get the list of OBJECT columns
        the_object_column_list = dsa.storage[OBJECT_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_object_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # get the list of BOOLEAN columns
        the_boolean_column_list = dsa.storage[BOOL_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_boolean_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # *****************************************************************
        # KNN does not have long odds linear plots
        # *****************************************************************

        # validate internal storage
        self.assertTrue("../../../resources/Output/ROC_AUC_PLOT_1.png"
                        in the_pg.plot_storage[GENERAL][PLOT_TYPE_ROC_AUC])

    # test case for generate_all_dataset_plots() method using a Random Forest model
    def test_generate_all_dataset_plots_FULL_random_forest_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
        pa.build_model(the_target_column='Bandwidth_GB_Year', model_type=MT_RF_REGRESSION, max_p_value=0.001)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # get a reference to the dataset analyzer
        dsa = pa.analyzer

        the_pg.generate_all_dataset_plots(statistics_generator=pa.s_gen,
                                          the_model_type=MT_RF_REGRESSION,
                                          the_version=ANALYZE_DATASET_FULL)

        # run validations for FLOAT
        # first, check to see if we can find on the storage a valid HIST plot for "Tenure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2],
                         "../../../resources/Output/FLOAT_BOX_PLOT_2.png")
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5],
                         "../../../resources/Output/FLOAT_Q_Q_PLOT_2.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_1]))  # HIST
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2]))  # BOX
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5]))  # QQ PLOT

        # next, check to see if we can find on the storage a valid HIST plot for "Income"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/FLOAT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_1]))

        # run validations for INT
        # check to see if we can find on the storage a valid HIST plot for "Population"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_1]))

        # next, check to see if we can find on the storage a valid HIST plot for "Yearly_equip_failure"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1],
                         "../../../resources/Output/INT_HIST_6.png")

        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_4][self.VALID_PLOT_TYPE_1]))

        # run validations for BOOLEAN
        # check to see if we can find ton the storage a VALID BAR_CHART for "Churn"
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3],
                         "../../../resources/Output/BOOL_BAR_CHART_0.png")
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3]))

        # validate the plot storage for SCATTER plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART]), 5)

        # get the list of tuples
        corr_list = pa.s_gen.get_list_of_correlations()

        # run assertions
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[0]],
                         self.VALID_SCATTER_PLOT_1)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[1]],
                         self.VALID_SCATTER_PLOT_2)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[2]],
                         self.VALID_SCATTER_PLOT_3)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[3]],
                         self.VALID_SCATTER_PLOT_4)
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][corr_list[4]],
                         self.VALID_SCATTER_PLOT_5)

        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]), 171)

        # get the list of statistics_generator.get_list_of_variable_relationships():
        count_plot_list = pa.s_gen.get_chi_squared_results()

        # create reference to clean count_plot_list
        clean_count_plot_list = []

        # clean the list to only include OBJECT to OBJECT
        for the_tuple in count_plot_list:
            # make sure the tuple is only OBJECT to OBJECT
            if dsa.validate_field_type(the_tuple[0], OBJECT_COLUMN_KEY) \
                    and dsa.validate_field_type(the_tuple[1], OBJECT_COLUMN_KEY):
                # add to the clean_count_plot_list
                clean_count_plot_list.append(the_tuple)

        # run assertions on clean_count_plot_list
        self.assertEqual(len(clean_count_plot_list), 15)
        self.assertEqual(clean_count_plot_list[0], ('Area', 'Marital', 0.503361))
        self.assertEqual(clean_count_plot_list[1], ('Area', 'Gender', 0.738468))
        self.assertEqual(clean_count_plot_list[2], ('Area', 'Contract', 0.848445))
        self.assertEqual(clean_count_plot_list[3], ('Area', 'InternetService', 0.98648))
        self.assertEqual(clean_count_plot_list[4], ('Area', 'PaymentMethod', 0.640729))

        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[0]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_156.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[1]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_157.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[2]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_158.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[3]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_159.png")
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][clean_count_plot_list[4]],
                         "../../../resources/Output/BIVARIATE_COUNT_PLOT_160.png")

        # validate joint plots
        # validate the plot storage for COUNT plots
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 190)

        # *****************************************************************
        # THERE ARE NO RESIDUAL PLOTS FOR Random Forest
        # *****************************************************************

        # get the LM_INITIAL_MODEL from the dsa
        the_initial_model = dsa.get_model(the_type=LM_INITIAL_MODEL)

        # get the list of features from the_linear_model
        the_features_list = the_initial_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # *****************************************************************
        # run validations for the bi-variate box plots
        # *****************************************************************

        # get the list of OBJECT columns
        the_object_column_list = dsa.storage[OBJECT_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_object_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # get the list of BOOLEAN columns
        the_boolean_column_list = dsa.storage[BOOL_COLUMN_KEY]

        # get the second column list
        the_second_column_list = dsa.storage[INT64_COLUMN_KEY] + dsa.storage[FLOAT64_COLUMN_KEY]

        # create the tuples
        for the_object_colum in the_boolean_column_list:
            for the_second_column in the_second_column_list:
                # create the tuple
                the_tuple = (the_object_colum, the_second_column)

                # run assertion
                self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple])

        # *****************************************************************
        # Random Forest does not have long odds linear plots
        # *****************************************************************

        # *****************************************************************
        # Random Forest does not have ROC / AUC plots
        # *****************************************************************

    # negative test method for generate_hist()
    def test_generate_boxplot_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa)

        # make sure None, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(None, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # make sure INVALID_DATA_TYPE, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(self.INVALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(self.VALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("The column is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, BAD_COLUMN, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(self.VALID_DATA_TYPE, self.BAD_COLUMN, None)

            # validate the error message.
            self.assertTrue("The column was not found on list for the data_type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, None)

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, str
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, "foo")

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate that we handle VALID_DATA_TYPE, VALID_COLUMN_1, -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_boxplot(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, -1)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # test method for generate_boxplot()
    def test_generate_boxplot(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # remove the previous box plot file if it is there
        if exists(self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_1):
            os.remove(self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_1)

        # run the method generate_boxplot()
        the_pg.generate_boxplot(self.VALID_DATA_TYPE, self.VALID_COLUMN_7, 1)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2],
                         self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_1)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_2]))

        # remove the previous box plot file if it is there
        if exists(self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_2):
            os.remove(self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_2)

        # add a second, we need to make sure the other logic on placing data works
        # run the method generate_boxplot()
        the_pg.generate_boxplot(self.VALID_DATA_TYPE, self.VALID_COLUMN_2, 5)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_2])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_2], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_2],
                         self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_2)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_2]))

        # remove the previous box plot file if it is there
        if exists(self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_3):
            os.remove(self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_3)

        # add a third,
        the_pg.generate_boxplot(self.VALID_DATA_TYPE_2, self.VALID_COLUMN_3, 7)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_2])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_2], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_2],
                         self.VALID_OVERRIDEN_BOX_PLOT_FILE_PATH_3)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_3][self.VALID_PLOT_TYPE_2]))

    # negative test method for generate_bar_chart()
    def test_generate_bar_chart_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa)

        # make sure None, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(None, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # make sure INVALID_DATA_TYPE, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(self.INVALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(self.VALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("The column is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, BAD_COLUMN, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(self.VALID_DATA_TYPE, self.BAD_COLUMN, None)

            # validate the error message.
            self.assertTrue("The column was not found on list for the data_type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, None)

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, str
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, "foo")

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate that we handle VALID_DATA_TYPE, VALID_COLUMN_1, -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_bar_chart(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, -1)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # test method for generate_bar_chart()
    def test_generate_bar_chart(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # remove the previous box plot file if it is there
        if exists(self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_1):
            os.remove(self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_1)

        # run the method generate_boxplot()
        the_pg.generate_bar_chart(self.VALID_DATA_TYPE_3, self.VALID_COLUMN_5, 1)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_5][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_5][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_5][self.VALID_PLOT_TYPE_3],
                         self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_1)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_5][self.VALID_PLOT_TYPE_3]))

        # remove the previous box plot file if it is there
        if exists(self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_2):
            os.remove(self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_2)

        # add a second, we need to make sure the other logic on placing data works
        # run the method generate_boxplot()
        the_pg.generate_bar_chart(self.VALID_DATA_TYPE_3, self.VALID_COLUMN_6, 5)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_6][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_6][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_6][self.VALID_PLOT_TYPE_3],
                         self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_2)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_6][self.VALID_PLOT_TYPE_3]))

        # remove the previous box plot file if it is there
        if exists(self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_3):
            os.remove(self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_3)

        # add a third, we need to make sure the other logic on placing data works
        # run the method generate_boxplot()
        the_pg.generate_bar_chart(self.VALID_DATA_TYPE_4, self.VALID_COLUMN_8, 6)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3],
                         self.VALID_OVERRIDEN_BAR_CHART_FILE_PATH_3)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_8][self.VALID_PLOT_TYPE_3]))

    # negative test method for generate_scree_plots
    def test_generate_scree_plots_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # make sure we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scree_plots(None)

            # validate the error message.
            self.assertTrue("The PCA_Analysis was None or incorrect type." in context.exception)

    # test method for generate_scree_plots()
    def test_generate_scree_plots(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create PCA Analysis object
        pca_analysis = PCA_Analysis(dsa, self.COLUMN_DICT)

        # run pca analysis
        pca_analysis.perform_analysis()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # generate scree plot
        the_pg.generate_scree_plots(pca_analysis)

        # run assertions
        self.assertTrue(exists(self.VALID_SCREE_PLOT_FILE_PATH_1))
        self.assertTrue(exists(self.VALID_SCREE_PLOT_FILE_PATH_2))

    # negative test method for generate_correlation_heatmap() method
    def test_generate_correlation_heatmap_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # make sure we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_correlation_heatmap(None)

            # validate the error message.
            self.assertTrue("The list of correlations was None or incorrect type." in context.exception)

    # test method for generate_correlation_heatmap()
    def test_generate_correlation_heatmap(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke the method
        the_sg.find_correlations()

        # remove the previous heatmap if it is there
        if exists(self.VALID_HEATMAP_FILE_PATH_1):
            os.remove(self.VALID_HEATMAP_FILE_PATH_1)

        # invoke the method
        the_pg.generate_correlation_heatmap(the_sg.get_list_of_correlations())

        # verify that the file was created
        self.assertTrue(exists(self.VALID_HEATMAP_FILE_PATH_1))

        # verify that the plot is in storage
        self.assertIsInstance(the_pg.plot_storage[GENERAL][PLOT_TYPE_HEATMAP], str)
        self.assertEqual(the_pg.plot_storage[GENERAL][PLOT_TYPE_HEATMAP], self.VALID_HEATMAP_FILE_PATH_1)

    # negative test method for generate_confusion_matrix_heatmap() method
    def test_generate_confusion_matrix_heatmap_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics
        pa.build_model(the_target_column='Churn',
                       model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001,
                       suppress_console=True)  # build a model

        pa.analyze_dataset(ANALYZE_DATASET_FULL)  # full dataset analysis
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics
        pa.build_model(the_target_column='Churn',
                       model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001,
                       the_max_vif=7.0,
                       suppress_console=False)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

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

        # get the underlying dataframe
        the_df = the_logistic_model.encoded_df

        # run assertions on the_df
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertTrue('Churn' in the_df.columns)

        # get the confusion matrix
        confusion_matrix = the_lm_result.get_confusion_matrix(the_encoded_df=the_df)

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method()
            the_pg.generate_confusion_matrix_heatmap(confusion_matrix=None, the_count=None)

            # validate the error message.
            self.assertTrue("confusion_matrix was None or incorrect type." in context.exception)

        # make sure we handle confusion_matrix, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method()
            the_pg.generate_confusion_matrix_heatmap(confusion_matrix=confusion_matrix, the_count=None)

            # validate the error message.
            self.assertTrue("the_count argument is None or incorrect type." in context.exception)

        # make sure we handle confusion_matrix, -1
        with self.assertRaises(ValueError) as context:
            # invoke method()
            the_pg.generate_confusion_matrix_heatmap(confusion_matrix=confusion_matrix, the_count=-1)

            # validate the error message.
            self.assertTrue("the_count argument must be positive." in context.exception)

    # test method for generate_correlation_heatmap() for logistic model
    def test_generate_confusion_matrix_heatmap_logistic_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # Note, something is causing the outlier clean up to blow up.

        # build a model
        pa.build_model(the_target_column='Churn',
                       model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001,
                       suppress_console=True)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

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

        # get the underlying dataframe
        the_df = the_logistic_model.encoded_df

        # run assertions on the_df
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertTrue('Churn' in the_df.columns)

        # remove the previous correlation heatmap if it is there
        if exists(self.VALID_HEATMAP_FILE_PATH_1):
            os.remove(self.VALID_HEATMAP_FILE_PATH_1)

        # remove the previous confusion matrix heatmap if it is there
        if exists(self.VALID_HEATMAP_FILE_PATH_2):
            os.remove(self.VALID_HEATMAP_FILE_PATH_2)

        # get the confusion matrix
        confusion_matrix = the_lm_result.get_confusion_matrix(the_encoded_df=the_df)

        # invoke the method with count = 2
        the_pg.generate_confusion_matrix_heatmap(confusion_matrix=confusion_matrix, the_count=2)

        # verify that the correlation heatmap was NOT created.  The code to invoke this is located
        # not in the PlotGenerator, but in the Project_Assessment in generate_output_report() method
        self.assertFalse(exists(self.VALID_HEATMAP_FILE_PATH_1))

        # verify that the confusion matrix heatmap was created
        self.assertTrue(exists(self.VALID_HEATMAP_FILE_PATH_2))

        # verify that the plots are in storage
        self.assertFalse(PLOT_TYPE_HEATMAP in the_pg.plot_storage[GENERAL])
        self.assertIsInstance(the_pg.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP], str)
        self.assertEqual(the_pg.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP], self.VALID_HEATMAP_FILE_PATH_2)

    # test method for generate_correlation_heatmap() for knn model
    def test_generate_confusion_matrix_heatmap_knn_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)  # calculate internal statistics

        # calling clean up outliers blows up, not sure why.

        pa.analyze_dataset(ANALYZE_DATASET_FULL)  # full dataset analysis

        pa.build_model(the_target_column='Churn',
                       model_type=MT_KNN_CLASSIFICATION,
                       max_p_value=0.001,
                       the_max_vif=7.0,
                       suppress_console=False)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions on the model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], KNN_Model)

        # we should not have a final model.
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # get a reference to the initial model
        the_model = pa.analyzer.linear_model_storage[LM_INITIAL_MODEL]

        # get the_model_result
        the_model_result = the_model.get_the_result()

        # run validations
        self.assertIsNotNone(the_model_result)
        self.assertIsInstance(the_model_result, KNN_Model_Result)

        # get the underlying dataframe
        the_df = the_model_result.the_df

        # run assertions on the_df
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 3000)  # match the 70% train, 30% test split.

        self.assertFalse('Churn' in the_df.columns)

        # remove the previous correlation heatmap if it is there
        if exists(self.VALID_HEATMAP_FILE_PATH_1):
            os.remove(self.VALID_HEATMAP_FILE_PATH_1)

        # remove the previous confusion matrix heatmap if it is there
        if exists(self.VALID_HEATMAP_FILE_PATH_2):
            os.remove(self.VALID_HEATMAP_FILE_PATH_2)

        # get the confusion matrix
        confusion_matrix = the_model_result.get_confusion_matrix()

        # invoke the method with count = 2
        the_pg.generate_confusion_matrix_heatmap(confusion_matrix=confusion_matrix, the_count=2)

        # verify that the correlation heatmap was NOT created.  The code to invoke this is located
        # not in the PlotGenerator, but in the Project_Assessment in generate_output_report() method
        self.assertFalse(exists(self.VALID_HEATMAP_FILE_PATH_1))

        # verify that the confusion matrix heatmap was created
        self.assertTrue(exists(self.VALID_HEATMAP_FILE_PATH_2))

        # verify that the plots are in storage
        self.assertFalse(PLOT_TYPE_HEATMAP in the_pg.plot_storage[GENERAL])
        self.assertIsInstance(the_pg.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP], str)
        self.assertEqual(the_pg.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP], self.VALID_HEATMAP_FILE_PATH_2)

    # negative tests for generate_scatter_plot() method
    def test_generate_scatter_plot_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # refresh the model
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(dsa, self.OVERRIDE_PATH)

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot(None, None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot("foo", None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle (None, "MonthlyCharge", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot((None, "MonthlyCharge", 0.60812), 1)

            # validate the error message.
            self.assertTrue("x_series is None or incorrect type." in context.exception)

        # make sure we handle ("foo", "MonthlyCharge", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot(("foo", "MonthlyCharge", 0.60812), 1)

            # validate the error message.
            self.assertTrue("x_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ("StreamingMovies", None, 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot(("StreamingMovies", None, 0.60812), 1)

            # validate the error message.
            self.assertTrue("y_series is None or incorrect type." in context.exception)

        # make sure we handle ("StreamingMovies", "foo", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot(("StreamingMovies", "foo", 0.60812), 1)

            # validate the error message.
            self.assertTrue("y_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ("StreamingMovies", "MonthlyCharge", 0.60812), None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot(("StreamingMovies", "MonthlyCharge", 0.60812), None)

            # validate the error message.
            self.assertTrue("the_count argument is None or incorrect type." in context.exception)

        # make sure we handle ("StreamingMovies", "MonthlyCharge", 0.60812), -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_scatter_plot(("StreamingMovies", "MonthlyCharge", 0.60812), -1)

            # validate the error message.
            self.assertTrue("the_count argument must be positive." in context.exception)

    # test method for generate_scatter_plot()
    def test_generate_scatter_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # define the tuple
        the_tuple_1 = ("StreamingMovies", "MonthlyCharge", 0.60812)
        the_tuple_2 = ("Tenure", "Bandwidth_GB_Year", 0.99150)

        # invoke the function
        the_pg.generate_scatter_plot(the_tuple_1, 1)

        # run assertions
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertTrue(the_tuple_1 in the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][the_tuple_1], self.VALID_SCATTER_PLOT_2)

        # invoke the function again
        the_pg.generate_scatter_plot(the_tuple_2, 2)

        # run assertions
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertTrue(the_tuple_2 in the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART])
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART][the_tuple_2], self.VALID_SCATTER_PLOT_3)

        # now, attempt to add the_tuple_1 again
        the_pg.generate_scatter_plot(the_tuple_1, 1)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_SCATTER_CHART]), 2)

    # negative test method for generate_count_plot()
    def test_generate_count_plot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot(None, None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot("foo", None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle (None, "MonthlyCharge", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot((None, "MonthlyCharge", 0.60812), 1)

            # validate the error message.
            self.assertTrue("x_series is None or incorrect type." in context.exception)

        # make sure we handle ("foo", "MonthlyCharge", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot(("foo", "MonthlyCharge", 0.60812), 1)

            # validate the error message.
            self.assertTrue("x_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ("StreamingMovies", None, 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot(("StreamingMovies", None, 0.60812), 1)

            # validate the error message.
            self.assertTrue("y_series is None or incorrect type." in context.exception)

        # make sure we handle ("StreamingMovies", "foo", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot(("StreamingMovies", "foo", 0.60812), 1)

            # validate the error message.
            self.assertTrue("y_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ("StreamingMovies", "MonthlyCharge", 0.60812), None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot(("StreamingMovies", "MonthlyCharge", 0.60812), None)

            # validate the error message.
            self.assertTrue("the_count argument is None or incorrect type." in context.exception)

        # make sure we handle ("StreamingMovies", "MonthlyCharge", 0.60812), -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_count_plot(("StreamingMovies", "MonthlyCharge", 0.60812), -1)

            # validate the error message.
            self.assertTrue("the_count argument must be positive." in context.exception)

    # test method for generate_count_plot()
    def test_generate_count_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # define the tuples
        the_tuple_1 = ("Area", "TimeZone", 0.98015)
        the_tuple_2 = ("Marital", "Gender", 0.82888)
        the_tuple_3 = ("Contract", "PaymentMethod", 0.71048)

        # invoke the function. Please note that the count is set to 1 for this test, which means the first
        # plot generated will start with 1.
        the_pg.generate_count_plot(the_tuple_1, 1)

        # run assertions
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertTrue(the_tuple_1 in the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        # self.VALID_COUNT_PLOT_2 is associated with
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][the_tuple_1], self.VALID_COUNT_PLOT_2)

        # invoke the function again
        the_pg.generate_count_plot(the_tuple_2, 2)

        # run assertions
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertTrue(the_tuple_2 in the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT])
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][the_tuple_2], self.VALID_COUNT_PLOT_3)

        # now, attempt to add the_tuple_1 again
        the_pg.generate_count_plot(the_tuple_1, 1)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]), 2)

        # now, add tuple 3, and make sure the storage increases again.
        the_pg.generate_count_plot(the_tuple_3, 3)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]), 3)

    # test method for generate_count_plot() for churn
    def test_generate_count_plot_churn(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get a reference to the statistics generator
        the_sg = pa.s_gen

        # create a counter variable
        i = 21

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # invoke the method for all churn variables.
        the_list = the_sg.filter_tuples_by_column(the_sg.all_corr_storage, "Churn")

        # run assertions
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)

        # loop over all the tuples in the_list
        for the_tuple in the_list:
            the_pg.generate_count_plot(the_tuple, i)

            # increment i
            i = i + 1

            # run assertions
            self.assertTrue(exists(the_pg.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][the_tuple]))

    # negative test method for generate_joint_plot()
    def test_generate_joint_plot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot(None, None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot("foo", None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle (None, "MonthlyCharge", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot((None, "MonthlyCharge", 0.60812), 1)

            # validate the error message.
            self.assertTrue("x_series is None or incorrect type." in context.exception)

        # make sure we handle ("foo", "MonthlyCharge", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot(("foo", "MonthlyCharge", 0.60812), 1)

            # validate the error message.
            self.assertTrue("x_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ("StreamingMovies", None, 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot(("StreamingMovies", None, 0.60812), 1)

            # validate the error message.
            self.assertTrue("y_series is None or incorrect type." in context.exception)

        # make sure we handle ("StreamingMovies", "foo", 0.60812), 1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot(("StreamingMovies", "foo", 0.60812), 1)

            # validate the error message.
            self.assertTrue("y_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ("Population", "Children", 0.60812), None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot(("Population", "Children", 0.60812), None)

            # validate the error message.
            self.assertTrue("the_count argument is None or incorrect type." in context.exception)

        # make sure we handle ("Population", "Children", 0.60812), -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_joint_plot(("Population", "Children", 0.60812), -1)

            # validate the error message.
            self.assertTrue("the_count argument must be positive." in context.exception)

    # test method for generate_joint_plot()
    def test_generate_joint_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # define the tuples
        the_tuple_1 = ("Population", "Children", 0.00251)  # integer vs. integer
        the_tuple_2 = ("Income", "Outage_sec_perweek", 0.82888)  # float vs. float
        the_tuple_3 = ("Population", "Income", 0.05512)  # integer vs. float
        the_tuple_4 = ("Population", "Churn", 0.05512)  # integer vs. bool
        the_tuple_5 = ("Income", "Churn", 0.05512)  # float vs. bool

        # remove the previous joint plot file if it is there
        if exists(self.VALID_JOINT_PLOT_1):
            os.remove(self.VALID_JOINT_PLOT_1)

        # invoke the function. Please note that the count is set to 1 for this test, which means the first
        # plot generated will start with 1.
        the_pg.generate_joint_plot(the_tuple_1, 1)

        # run assertions
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertTrue(the_tuple_1 in the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        # self.VALID_COUNT_PLOT_2 is associated with
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT][the_tuple_1], self.VALID_JOINT_PLOT_1)
        self.assertTrue(exists(self.VALID_JOINT_PLOT_1))

        # remove the previous joint plot file if it is there
        if exists(self.VALID_JOINT_PLOT_2):
            os.remove(self.VALID_JOINT_PLOT_2)

        # invoke the function again
        the_pg.generate_joint_plot(the_tuple_2, 2)

        # run assertions
        self.assertIsNotNone(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertTrue(the_tuple_2 in the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT])
        self.assertEqual(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT][the_tuple_2], self.VALID_JOINT_PLOT_2)
        self.assertTrue(exists(self.VALID_JOINT_PLOT_2))

        # now, attempt to add the_tuple_1 again
        the_pg.generate_joint_plot(the_tuple_1, 1)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 2)

        # now, add tuple 3, and make sure the storage increases again.
        the_pg.generate_joint_plot(the_tuple_3, 3)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 3)

        # now, add tuple 4, and make sure the storage increases again.
        the_pg.generate_joint_plot(the_tuple_4, 4)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 4)

        # now, add tuple 5, and make sure the storage increases again.
        the_pg.generate_joint_plot(the_tuple_5, 5)

        # run assertions
        self.assertEqual(len(the_pg.plot_storage[PLOT_TYPE_JOINT_PLOT]), 5)

    # negative test for generate_q_q_plot() method
    def test_generate_q_q_plot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # make sure we handle None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(None, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # make sure INVALID_DATA_TYPE, None, None is handled.
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(self.INVALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("data_type argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, None, None)

            # validate the error message.
            self.assertTrue("The column is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, BAD_COLUMN, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, self.BAD_COLUMN, None)

            # validate the error message.
            self.assertTrue("The column was not found on list for the data_type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, None)

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate we handle VALID_DATA_TYPE, VALID_COLUMN_1, str
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, "foo")

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # validate that we handle VALID_DATA_TYPE, VALID_COLUMN_1, -1
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_hist()
            the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, self.VALID_COLUMN_1, -1)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # test method for generate_q_q_plot() method
    def test_generate_q_q_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # remove the test files if they already exist.
        if exists(self.VALID_QQ_PLOT_1):
            os.remove(self.VALID_QQ_PLOT_1)
        elif exists(self.VALID_QQ_PLOT_2):
            os.remove(self.VALID_QQ_PLOT_2)
        elif exists(self.VALID_QQ_PLOT_3):
            os.remove(self.VALID_QQ_PLOT_3)

        # run the method generate_q_q_plot() for FLOAT64
        the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, self.VALID_COLUMN_7, 0)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5], self.VALID_QQ_PLOT_1)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_7][self.VALID_PLOT_TYPE_5]))

        # run second test, we need to make sure logic of how data is stored in the structure works
        # correctly.
        # run the method generate_hist() for FLOAT64
        the_pg.generate_q_q_plot(self.VALID_DATA_TYPE, self.VALID_COLUMN_2, 5)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_5])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_5], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_5], self.VALID_QQ_PLOT_2)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_2][self.VALID_PLOT_TYPE_5]))

        # run the method generate_hist() for INT64 that has been moved from float with nan to int64 with nan.
        the_pg.generate_q_q_plot(self.VALID_DATA_TYPE_2, self.VALID_COLUMN_1, 1)

        # run validations
        self.assertIsNotNone(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_5])
        self.assertIsInstance(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_5], str)
        self.assertEqual(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_5], self.VALID_QQ_PLOT_3)
        self.assertTrue(exists(the_pg.plot_storage[self.VALID_COLUMN_1][self.VALID_PLOT_TYPE_5]))

    # negative test method for is_plot_for_tuple_already_created()
    def test_is_plot_for_tuple_already_created_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.is_plot_for_tuple_already_created(None, None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.is_plot_for_tuple_already_created("foo", None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle ('Churn', 'Techie', 0.066722), None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.is_plot_for_tuple_already_created(('Churn', 'Techie', 0.066722), None)

            # validate the error message.
            self.assertTrue("the_type argument is None or unknown." in context.exception)

        # make sure we handle ('Churn', 'Techie', 0.066722), "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.is_plot_for_tuple_already_created(('Churn', 'Techie', 0.066722), "foo")

            # validate the error message.
            self.assertTrue("the_type argument is None or unknown." in context.exception)

    # test method for is_plot_for_tuple_already_created()
    def test_is_plot_for_tuple_already_created(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # define the tuple
        the_tuple_1 = ('Churn', 'Techie', 0.0)

        # run assertions
        self.assertFalse(the_pg.is_plot_for_tuple_already_created(the_tuple_1, "BIVARIATE_COUNT_PLOT"))
        the_pg.generate_count_plot(the_tuple_1, 1)
        self.assertTrue(the_pg.is_plot_for_tuple_already_created(the_tuple_1, "BIVARIATE_COUNT_PLOT"))

        # define the tuple
        the_tuple_2 = ('Churn', 'Contract', 0.0)

        self.assertFalse(the_pg.is_plot_for_tuple_already_created(the_tuple_2, "BIVARIATE_COUNT_PLOT"))
        the_pg.generate_count_plot(the_tuple_2, 2)
        self.assertTrue(the_pg.is_plot_for_tuple_already_created(the_tuple_1, "BIVARIATE_COUNT_PLOT"))
        self.assertTrue(the_pg.is_plot_for_tuple_already_created(the_tuple_2, "BIVARIATE_COUNT_PLOT"))

        # define the tuple
        the_tuple_3 = ('Churn', 'Port_modem', 0.427757)

        self.assertFalse(the_pg.is_plot_for_tuple_already_created(the_tuple_3, "BIVARIATE_COUNT_PLOT"))
        the_pg.generate_count_plot(the_tuple_3, 3)
        self.assertTrue(the_pg.is_plot_for_tuple_already_created(the_tuple_1, "BIVARIATE_COUNT_PLOT"))
        self.assertTrue(the_pg.is_plot_for_tuple_already_created(the_tuple_2, "BIVARIATE_COUNT_PLOT"))
        self.assertTrue(the_pg.is_plot_for_tuple_already_created(the_tuple_3, "BIVARIATE_COUNT_PLOT"))

    # proof of concept work to attempt to generate residual charts for a linear model
    def test_proof_of_concept_for_residual_plots(self, sm=None):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # build a linear model
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get access to the FINAL Linear_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions on the_lm_result
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, RegressionResultsWrapper)

        # retrieve the entire encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df.columns.to_list()), 48)

        # get the list of features for the model
        the_features_list = the_lm_result.get_the_variables_list()

        # run assertions on the_features_list
        self.assertIsNotNone(the_features_list)
        self.assertIsInstance(the_features_list, list)
        self.assertEqual(len(the_features_list), 12)

        # loop over the_features_list and verify each element is in encoded_df.columns.to_list()
        for the_feature in the_features_list:
            self.assertTrue(the_feature in encoded_df.columns.to_list())

        # clear everything
        plt.clf()

        # generate the path to the file name, and verify it is there
        self.assertTrue(exists("../../../resources/Output/"))

        # generate the plot name for MonthlyCharge residuals
        the_plot_name = "../../../resources/Output/residual_plot_MonthlyCharge.png"

        # we need to make sure the previous image is removed
        if exists(the_plot_name):
            os.remove(the_plot_name)

        # create instance of influence
        influence = the_model.get_influence()

        # get the standard residuals
        standardized_residuals = influence.resid_studentized_internal

        plt.scatter(encoded_df['MonthlyCharge'], standardized_residuals)
        plt.xlabel('MonthlyCharge')
        plt.ylabel('Standardized Residuals')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.savefig(the_plot_name)
        plt.close()

        # make sure the plot exists
        self.assertTrue(exists(the_plot_name))

        # repeat the process for feature TechSupport

        # clear everything
        plt.clf()

        # generate the plot name for TechSupport residuals
        the_plot_name = "../../../resources/Output/residual_plot_TechSupport.png"

        # we need to make sure the previous image is removed
        if exists(the_plot_name):
            os.remove(the_plot_name)

        plt.scatter(encoded_df['TechSupport'], standardized_residuals)
        plt.xlabel('TechSupport')
        plt.ylabel('Standardized Residuals')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.savefig(the_plot_name)
        plt.close()

        # make sure the plot exists
        self.assertTrue(exists(the_plot_name))

    # negative test method for generate_standardized_residual_plot() method
    def test_generate_standardized_residual_plot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a linear model
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get access to the FINAL Linear_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions on the_lm_result
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, RegressionResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # make sure we handle None, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_standardized_residual_plot()
            the_pg.generate_standardized_residual_plot(the_model=None, the_encoded_df=None,
                                                       the_column=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_model argument is None or incorrect type." in context.exception)

        # make sure we handle the_model, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_standardized_residual_plot()
            the_pg.generate_standardized_residual_plot(the_model=the_model, the_encoded_df=None,
                                                       the_column=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_encoded_df argument is None or incorrect type." in context.exception)

        # make sure we handle the_model, encoded_df, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_standardized_residual_plot()
            the_pg.generate_standardized_residual_plot(the_model=the_model, the_encoded_df=encoded_df,
                                                       the_column=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_column argument is None or incorrect type." in context.exception)

        # make sure we handle the_model, encoded_df, "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke generate_standardized_residual_plot()
            the_pg.generate_standardized_residual_plot(the_model=the_model, the_encoded_df=encoded_df,
                                                       the_column="foo", the_count=None)

            # validate the error message.
            self.assertTrue("the_column argument is not present in the_encoded_df." in context.exception)

        # make sure we handle the_model, encoded_df, 'MonthlyCharge', None
        with self.assertRaises(SyntaxError) as context:
            # invoke generate_standardized_residual_plot()
            the_pg.generate_standardized_residual_plot(the_model=the_model, the_encoded_df=encoded_df,
                                                       the_column='MonthlyCharge', the_count=None)

            # validate the error message.
            self.assertTrue("the_count argument is None or incorrect type." in context.exception)

        # make sure we handle the_model, encoded_df, 'MonthlyCharge', -3
        with self.assertRaises(ValueError) as context:
            # invoke generate_standardized_residual_plot()
            the_pg.generate_standardized_residual_plot(the_model=the_model, the_encoded_df=encoded_df,
                                                       the_column='MonthlyCharge', the_count=-3)

            # validate the error message.
            self.assertTrue("the_count argument must be positive." in context.exception)

    # test method for generate_standardized_residual_plot()
    def test_generate_standardized_residual_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a linear model
        pa.build_model(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get access to the FINAL Linear_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions on the_lm_result
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, RegressionResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # delete the file if it previously existed
        if exists("../../../resources/Output/STANDARDIZED_RESIDUAL_PLOT_1.png"):
            os.remove("../../../resources/Output/STANDARDIZED_RESIDUAL_PLOT_1.png")

        # the_model, the_encoded_df, the_column, the_count
        # invoke the method
        the_pg.generate_standardized_residual_plot(the_model=the_model, the_encoded_df=encoded_df,
                                                   the_column='MonthlyCharge', the_count=1)

        # run final assertion that the plot exists
        self.assertTrue(exists("../../../resources/Output/STANDARDIZED_RESIDUAL_PLOT_1.png"))

    # proof of concept of a boxplot between MonthlyCharge and Churn
    def test_proof_of_concept_MonthlyCharge_and_Churn_boxplot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get data_df
        data_df = pa.analyzer.the_df

        # run assertions on the_lm_result
        self.assertIsNotNone(data_df)
        self.assertIsInstance(data_df, DataFrame)

        # delete the file if it previously existed
        if exists("../../../resources/Output/BIVARIATE_BOXPLOT_1.png"):
            os.remove("../../../resources/Output/BIVARIATE_BOXPLOT_1.png")

        # clear everything
        plt.clf()

        # generate bivariate box plot between MonthlyCharge and Churn
        # plt.subplot(1, 2, 2)
        plt.title("MonthlyCharge vs. Churn")
        sns.boxplot(data=data_df, x="Churn", y="MonthlyCharge", color="cyan")

        plt.xlabel("Churn")
        plt.ylabel("Monthly Charge")

        # save
        plt.savefig("../../../resources/Output/BIVARIATE_BOXPLOT_1.png")
        plt.close()

    # negative test method for generate_bivariate_boxplot()
    def test_generate_bivariate_boxplot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)

        # make sure we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot(None, None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot("foo", None)

            # validate the error message.
            self.assertTrue("the_tuple argument is None or incorrect type." in context.exception)

        # make sure we handle (None, 'Techie', 0.066722), None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot((None, 'Techie', 0.066722), None)

            # validate the error message.
            self.assertTrue("x_series is None or incorrect type." in context.exception)

        # make sure we handle ('foo', 'Techie', 0.066722), None
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot(('foo', 'Techie', 0.066722), None)

            # validate the error message.
            self.assertTrue("x_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ('Churn', None, 0.066722), None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot(('Churn', None, 0.066722), None)

            # validate the error message.
            self.assertTrue("y_series is None or incorrect type." in context.exception)

        # make sure we handle ('Churn', 'foo', 0.066722), None
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot(('Churn', 'foo', 0.066722), None)

            # validate the error message.
            self.assertTrue("x_series is not a valid field on dataframe." in context.exception)

        # make sure we handle ('Churn', 'MonthlyCharge', 0.066722), None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot(('Churn', 'MonthlyCharge', 0.066722), None)

            # validate the error message.
            self.assertTrue("the_count argument is None or incorrect type." in context.exception)

        # make sure we handle ('Churn', 'MonthlyCharge', 0.066722), -2
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_bivariate_boxplot(('Churn', 'MonthlyCharge', 0.066722), -2)

            # validate the error message.
            self.assertTrue("the_count argument must be positive." in context.exception)

    # test method for generate_bivariate_boxplot()
    def test_generate_bivariate_boxplot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # *******************************************************
        # create the a BOOLEAN and FLOAT tuple
        # *******************************************************
        the_tuple = ('Churn', 'MonthlyCharge', 0.066722)

        # delete the file if it previously existed
        if exists("../../../resources/Output/BIVARIATE_BOX_PLOT_1.png"):
            os.remove("../../../resources/Output/BIVARIATE_BOX_PLOT_1.png")

        # invoke the method
        the_pg.generate_bivariate_boxplot(the_tuple=the_tuple, the_count=1)

        # run assertions
        self.assertTrue(exists("../../../resources/Output/BIVARIATE_BOX_PLOT_1.png"))

        # get the storage
        the_plot_storage = the_pg.plot_storage

        # run assertions
        self.assertIsNotNone(the_plot_storage)
        self.assertIsInstance(the_plot_storage, dict)
        self.assertEqual(the_plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple],
                         "../../../resources/Output/BIVARIATE_BOX_PLOT_1.png")

        # *******************************************************
        # create the a OBJECT and FLOAT tuple
        # *******************************************************
        the_tuple = ('Gender', 'MonthlyCharge', 0.066722)

        # delete the file if it previously existed
        if exists("../../../resources/Output/BIVARIATE_BOX_PLOT_2.png"):
            os.remove("../../../resources/Output/BIVARIATE_BOX_PLOT_2.png")

        # invoke the method
        the_pg.generate_bivariate_boxplot(the_tuple=the_tuple, the_count=2)

        # run assertions
        self.assertTrue(exists("../../../resources/Output/BIVARIATE_BOX_PLOT_2.png"))

        # run assertions
        self.assertEqual(the_plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple],
                         "../../../resources/Output/BIVARIATE_BOX_PLOT_2.png")

    # negative tests for generate_q_q_plot_for_residuals()
    def test_generate_q_q_plot_for_residuals_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a linear model
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get access to the FINAL Linear_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, RegressionResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # make sure we handle None, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type=None, the_column=None,
                                                   the_model=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_model_type is None or incorrect type." in context.exception)

        # make sure we handle "foo", None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type="foo", the_column=None,
                                                   the_model=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_model_type is not a valid value." in context.exception)

        # make sure we handle LM_FINAL_MODEL, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column=None,
                                                   the_model=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_model_type is None or incorrect type." in context.exception)

        # make sure we handle LM_FINAL_MODEL, "foo", None, None
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column="foo",
                                                   the_model=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_model_type is None or incorrect type." in context.exception)

        # make sure we handle LM_FINAL_MODEL, "Tenure", None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column="Tenure",
                                                   the_model=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_model argument is None or incorrect type." in context.exception)

        # make sure we handle LM_FINAL_MODEL, "Tenure", the_model, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column="Tenure",
                                                   the_model=the_model, the_count=None)

            # validate the error message.
            self.assertTrue("The count argument is None or incorrect type." in context.exception)

        # make sure we handle LM_FINAL_MODEL, "Tenure", the_model, -1
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column="Tenure",
                                                   the_model=the_model, the_count=-1)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # test method for generate_q_q_plot_for_residuals()
    def test_generate_q_q_plot_for_residuals_linear(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a linear model
        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # get access to the FINAL Linear_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).the_result

        # run assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, RegressionResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # check if the file to be generated already exists
        if exists("../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png"):
            os.remove("../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png")

        # invoke the method
        # the_model_type, the_column, the_model, the_count
        the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column="Tenure",
                                               the_model=the_model, the_count=1)

        # run assertions
        self.assertTrue(exists("../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png"))

        # get the storage from the_pg
        the_storage = the_pg.plot_storage

        # run assertions
        self.assertIsNotNone(the_storage)
        self.assertIsInstance(the_storage, dict)
        self.assertIsNotNone(the_storage[PLOT_TYPE_Q_Q_RESIDUAL_PLOT]["Tenure"])
        self.assertEqual(the_storage[PLOT_TYPE_Q_Q_RESIDUAL_PLOT]["Tenure"],
                         "../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png")

    # test method for generate_q_q_plot_for_residuals()
    def test_generate_q_q_plot_for_residuals_logistic(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a linear model
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # get access to the FINAL Linear_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, BinaryResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # check if the file to be generated already exists
        if exists("../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png"):
            os.remove("../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png")

        # invoke the method
        # the_model_type, the_column, the_model, the_count
        the_pg.generate_q_q_plot_for_residuals(the_model_type=LM_FINAL_MODEL, the_column="Tenure",
                                               the_model=the_model, the_count=1)

        # run assertions
        self.assertTrue(exists("../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png"))

        # get the storage from the_pg
        the_storage = the_pg.plot_storage

        # run assertions
        self.assertIsNotNone(the_storage)
        self.assertIsInstance(the_storage, dict)
        self.assertIsNotNone(the_storage[PLOT_TYPE_Q_Q_RESIDUAL_PLOT]["Tenure"])
        self.assertEqual(the_storage[PLOT_TYPE_Q_Q_RESIDUAL_PLOT]["Tenure"],
                         "../../../resources/Output/Q_Q_RESIDUAL_PLOT_1.png")

    # negative tests for generate_long_odds_linear_plot()
    def test_generate_long_odds_linear_plot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a logistic model
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001, the_max_vif=7.0)

        # get access to the FINAL Logistic_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, BinaryResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # make sure we handle None, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable=None, the_independent_variable=None,
                                                  the_df=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_target_variable is None or incorrect type." in context.exception)

        # make sure we handle "foo", None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable="foo", the_independent_variable=None,
                                                  the_df=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_independent_variable is None or incorrect type." in context.exception)

        # make sure we handle "foo", "foo", None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable="foo", the_independent_variable="foo",
                                                  the_df=None, the_count=None)

            # validate the error message.
            self.assertTrue("the_df is None or incorrect type." in context.exception)

        # make sure we handle "foo", "foo", encoded_df, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable="foo", the_independent_variable="foo",
                                                  the_df=encoded_df, the_count=None)

            # validate the error message.
            self.assertTrue("the_count is None or incorrect type." in context.exception)

        # make sure we handle "foo", "foo", encoded_df, -1
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable="foo", the_independent_variable="foo",
                                                  the_df=encoded_df, the_count=-1)

            # validate the error message.
            self.assertTrue("the_target_variable is not in the_df." in context.exception)

        # make sure we handle "Churn", "foo", encoded_df, -1
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable="Churn", the_independent_variable="foo",
                                                  the_df=encoded_df, the_count=-1)

            # validate the error message.
            self.assertTrue("the_independent_variable is not in the_df." in context.exception)

        # make sure we handle "Churn", "StreamingTV", encoded_df, -1
        with self.assertRaises(ValueError) as context:
            # invoke method
            the_pg.generate_long_odds_linear_plot(the_target_variable="Churn", the_independent_variable="StreamingTV",
                                                  the_df=encoded_df, the_count=-1)

            # validate the error message.
            self.assertTrue("The count argument must be positive." in context.exception)

    # test method for generate_long_odds_linear_plot()
    def test_generate_long_odds_linear_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # build a logistic model
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001, the_max_vif=7.0)

        # get access to the FINAL Logistic_Model
        the_lm_result = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).get_the_result()

        # run assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # get access to the_model
        the_model = the_lm_result.model

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, BinaryResultsWrapper)

        # get access to the_lm_result.encoded_df
        encoded_df = pa.analyzer.get_model(the_type=LM_FINAL_MODEL).encoded_df

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # check if the file to be generated already exists
        if exists("../../../resources/Output/LONG_ODDS_PLOT_1.png"):
            os.remove("../../../resources/Output/LONG_ODDS_PLOT_1.png")

        # invoke the method
        the_pg.generate_long_odds_linear_plot(the_target_variable="Churn",
                                              the_independent_variable="StreamingTV",
                                              the_df=encoded_df,
                                              the_count=1)

        # run assertions
        self.assertTrue(exists("../../../resources/Output/LONG_ODDS_PLOT_1.png"))

        # get the storage from the_pg
        the_storage = the_pg.plot_storage

        # run assertions
        self.assertIsNotNone(the_storage)
        self.assertIsInstance(the_storage, dict)
        self.assertIsNotNone(the_storage[PLOT_TYPE_LONG_ODDS]["StreamingTV"])
        self.assertEqual(the_storage[PLOT_TYPE_LONG_ODDS]["StreamingTV"],
                         "../../../resources/Output/LONG_ODDS_PLOT_1.png")

    # negative tests for generate_auc_roc_plot()
    def test_generate_auc_roc_plot_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
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
        self.assertIsInstance(selected_features, np.ndarray)
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

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
                                                                                     FLOAT64_COLUMN_KEY])

        # now we need to get the features we can scale.
        the_f_df_train_to_scale = [i for i in the_f_df_train.columns.to_list() if i in list_of_features_to_scale]

        # now we need to scale the_f_df_train, the_f_df_test only using the columns present in the list
        # the_f_df_train_to_scale
        # Note: The datatype changes from a DataFrame to a numpy.array after the call to fit_tranform().
        the_scaled_f_train_df = the_scalar.fit_transform(the_f_df_train[the_f_df_train_to_scale])
        the_scaled_f_test_df = the_scalar.fit_transform(the_f_df_test[the_f_df_train_to_scale])

        # now we need to cast the_scaled_f_train_df and the_scaled_f_test_df back to DataFrames
        the_scaled_f_train_df = pd.DataFrame(data=the_scaled_f_train_df, columns=the_f_df_train_to_scale)
        the_scaled_f_test_df = pd.DataFrame(data=the_scaled_f_test_df, columns=the_f_df_train_to_scale)

        # re-inject the scaled feature Series back into the_f_df_train and the_f_df_test.
        for the_field in the_f_df_train_to_scale:
            # update the original dataframe
            the_f_df_train[the_field] = the_scaled_f_train_df[the_field].values
            the_f_df_test[the_field] = the_scaled_f_test_df[the_field].values

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the result
        the_knn_result = KNN_Model_Result(the_model=knn,
                                          the_target_variable='Churn',
                                          the_variables_list=the_features_df.columns.to_list(),
                                          the_f_df_train=the_f_df_train,
                                          the_f_df_test=the_f_df_test,
                                          the_t_var_train=the_t_var_train,
                                          the_t_var_test=the_t_var_test,
                                          the_encoded_df=the_features_df,
                                          the_p_values=p_values,
                                          gridsearch=knn_cv,
                                          prepared_data=the_features_df,
                                          cleaned_data=the_df)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # get the model from the_knn_result
        the_knn_model = the_knn_result.get_model()

        # run assertions on the_knn_model
        self.assertIsNotNone(the_knn_model)
        self.assertIsInstance(the_knn_model, KNeighborsClassifier)

        # get the y_predicted
        the_y_predicted = the_knn_result.get_y_predicted()

        # run assertions on the the_y_predicted
        self.assertIsNotNone(the_y_predicted)
        self.assertIsInstance(the_y_predicted, np.ndarray)

        # get the y_scores
        the_y_scores = the_knn_result.get_y_scores()

        # run assertions on the the_y_scores
        self.assertIsNotNone(the_y_scores)
        self.assertIsInstance(the_y_scores, np.ndarray)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # *************************************************************
        # finally, ready to run tests.
        # *************************************************************

        # verify we handle None, None, None, None
        with self.assertRaises(SyntaxError) as context:
            the_pg.generate_auc_roc_plot_knn(the_model=None,
                                             x_test=None,
                                             y_test=None,
                                             the_count=None)

        # validate the error message.
        self.assertEqual("the_model is None or incorrect type.", context.exception.msg)

        # verify we handle the_knn_model, None, None, None
        with self.assertRaises(SyntaxError) as context:
            the_pg.generate_auc_roc_plot_knn(the_model=the_knn_model,
                                             x_test=None,
                                             y_test=None,
                                             the_count=None)

        # validate the error message.
        self.assertEqual("x_test is None or incorrect type.", context.exception.msg)

        # verify we handle the_knn_model, the_t_var_test, None, None
        with self.assertRaises(SyntaxError) as context:
            the_pg.generate_auc_roc_plot_knn(the_model=the_knn_model,
                                             x_test=the_f_df_test,
                                             y_test=None,
                                             the_count=None)

        # validate the error message.
        self.assertEqual("y_test is None or incorrect type.", context.exception.msg)

        # verify we handle the_knn_model, the_t_var_test, the_y_predicted, None
        with self.assertRaises(SyntaxError) as context:
            the_pg.generate_auc_roc_plot_knn(the_model=the_knn_model,
                                             x_test=the_f_df_test,
                                             y_test=the_t_var_test,
                                             the_count=None)

        # validate the error message.
        self.assertEqual("the_count is None or incorrect type.", context.exception.msg)

        # verify we handle the_knn_model, the_t_var_test, the_y_predicted, -2
        with self.assertRaises(ValueError) as context:
            the_pg.generate_auc_roc_plot_knn(the_model=the_knn_model,
                                             x_test=the_f_df_test,
                                             y_test=the_t_var_test,
                                             the_count=-2)

        # validate the error message.
        self.assertEqual("The count argument must be positive.", str(context.exception))

    # tests for generate_auc_roc_plot() method
    def test_generate_auc_roc_plot(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
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
        self.assertIsInstance(selected_features, np.ndarray)
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

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
                                                                                     FLOAT64_COLUMN_KEY])

        # now we need to get the features we can scale.
        the_f_df_train_to_scale = [i for i in the_f_df_train.columns.to_list() if i in list_of_features_to_scale]

        # now we need to scale the_f_df_train, the_f_df_test only using the columns present in the list
        # the_f_df_train_to_scale
        # Note: The datatype changes from a DataFrame to a numpy.array after the call to fit_tranform().
        the_scaled_f_train_df = the_scalar.fit_transform(the_f_df_train[the_f_df_train_to_scale])
        the_scaled_f_test_df = the_scalar.fit_transform(the_f_df_test[the_f_df_train_to_scale])

        # now we need to cast the_scaled_f_train_df and the_scaled_f_test_df back to DataFrames
        the_scaled_f_train_df = pd.DataFrame(data=the_scaled_f_train_df, columns=the_f_df_train_to_scale)
        the_scaled_f_test_df = pd.DataFrame(data=the_scaled_f_test_df, columns=the_f_df_train_to_scale)

        # re-inject the scaled feature Series back into the_f_df_train and the_f_df_test.
        for the_field in the_f_df_train_to_scale:
            # update the original dataframe
            the_f_df_train[the_field] = the_scaled_f_train_df[the_field].values
            the_f_df_test[the_field] = the_scaled_f_test_df[the_field].values

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the result
        the_knn_result = KNN_Model_Result(the_model=knn,
                                          the_target_variable='Churn',
                                          the_variables_list=the_features_df.columns.to_list(),
                                          the_f_df_train=the_f_df_train,
                                          the_f_df_test=the_f_df_test,
                                          the_t_var_train=the_t_var_train,
                                          the_t_var_test=the_t_var_test,
                                          the_encoded_df=the_features_df,
                                          the_p_values=p_values,
                                          gridsearch=knn_cv,
                                          prepared_data=the_features_df,
                                          cleaned_data=the_df)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # get the model from the_knn_result
        the_knn_model = the_knn_result.get_model()

        # run assertions on the_knn_model
        self.assertIsNotNone(the_knn_model)
        self.assertIsInstance(the_knn_model, KNeighborsClassifier)

        # get the y_predicted
        the_y_predicted = the_knn_result.get_y_predicted()

        # run assertions on the the_y_predicted
        self.assertIsNotNone(the_y_predicted)
        self.assertIsInstance(the_y_predicted, np.ndarray)

        # get the y_scores
        the_y_scores = the_knn_result.get_y_scores()

        # run assertions on the the_y_scores
        self.assertIsNotNone(the_y_scores)
        self.assertIsInstance(the_y_scores, np.ndarray)

        # run assertions to make sure we have an StatisticsGenerator
        self.assertIsNotNone(pa.s_gen)

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(pa.analyzer, self.OVERRIDE_PATH)

        # run assertions to make sure we have an PlotGenerator
        self.assertIsNotNone(the_pg)
        self.assertIsInstance(the_pg, PlotGenerator)

        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test

        # make sure the output directory is there.
        self.assertTrue(exists("../../../resources/Output/"))

        # check if the file to be generated already exists
        if exists("../../../resources/Output/ROC_AUC_PLOT_2.png"):
            os.remove("../../../resources/Output/ROC_AUC_PLOT_2.png")

        # make sure the file does not exist
        self.assertFalse(exists("../../../resources/Output/ROC_AUC_PLOT_2.png"))

        # invoke method
        the_pg.generate_auc_roc_plot_knn(the_model=knn, x_test=the_f_df_test, y_test=the_t_var_test, the_count=2)

        # validate the plot was created
        self.assertTrue(exists("../../../resources/Output/ROC_AUC_PLOT_2.png"))

        # validate internal storage
        self.assertTrue("../../../resources/Output/ROC_AUC_PLOT_2.png"
                        in the_pg.plot_storage[GENERAL][PLOT_TYPE_ROC_AUC])
