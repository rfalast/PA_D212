import os
import unittest
from os.path import exists

import numpy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt

from numpy import ndarray
from pandas import Series, DataFrame
from sklearn import metrics
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from model.Project_Assessment import Project_Assessment
from model.analysis.models.Logistic_Model import Logistic_Model
from model.analysis.models.Logistic_Model_Result import Logistic_Model_Result
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_209_CHURN, MT_LOGISTIC_REGRESSION
from model.constants.ModelConstants import LM_FEATURE_NUM, LM_COEFFICIENT, LM_STANDARD_ERROR, LM_T_STATISTIC, \
    LM_LS_CONF_INT, LM_RS_CONF_INT, LM_VIF, LM_PREDICTOR, LM_P_VALUE, LM_FINAL_MODEL, LM_INITIAL_MODEL
from model.constants.ReportConstants import PSEUDO_R_SQUARED_HEADER, MODEL_PRECISION, MODEL_RECALL, MODEL_F1_SCORE, \
    MODEL_ACCURACY, AIC_SCORE, BIC_SCORE, LOG_LIKELIHOOD, NUMBER_OF_OBS, DEGREES_OF_FREEDOM_MODEL, \
    DEGREES_OF_FREEDOM_RESID, MODEL_CONSTANT
from util.CSV_loader import CSV_Loader


class test_Logistic_Model_Result(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_209_CHURN

    # negative test method for __init__
    def test__init__negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a linear model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # verify we handle None, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Logistic_Model_Result(the_regression_wrapper=None, the_target_variable=None,
                                  the_variables_list=None, the_df=None)

            # validate the error message.
            self.assertTrue("the_regression_wrapper is None or incorrect type." in context.exception)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # verify we handle model, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Logistic_Model_Result(the_regression_wrapper=fitted_model, the_target_variable=None,
                                  the_variables_list=None, the_df=None)

            # validate the error message.
            self.assertTrue("the_target_variable is None or incorrect type." in context.exception)

        # verify we handle model, 'Churn', None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Logistic_Model_Result(the_regression_wrapper=fitted_model, the_target_variable='Churn',
                                  the_variables_list=None, the_df=None)

            # validate the error message.
            self.assertTrue("the_regression_wrapper is None or incorrect type." in context.exception)

        # verify we handle model, 'Churn', the_variable_columns, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Logistic_Model_Result(the_regression_wrapper=fitted_model, the_target_variable='Churn',
                                  the_variables_list=the_variable_columns, the_df=None)

            # validate the error message.
            self.assertTrue("the_df is None or incorrect type." in context.exception)

    # test method for __init__
    def test__init__(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a linear model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

    # test method for get_model()
    def test_get_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a linear model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)
        self.assertIsNotNone(the_result.get_model())
        self.assertIsInstance(the_result.get_model(), BinaryResultsWrapper)
        self.assertIsNotNone(the_result.the_target_variable)
        self.assertEqual(the_result.the_target_variable, "Churn")
        self.assertIsNotNone(the_result.the_variables_list)
        self.assertIsInstance(the_result.the_variables_list, list)
        self.assertIsNotNone(the_result.the_df)
        self.assertIsInstance(the_result.the_df, DataFrame)

    # test method for get_the_target_variable()
    def test_get_the_target_variable(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a linear model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke method
        self.assertIsNotNone(the_result.get_the_target_variable())
        self.assertIsInstance(the_result.get_the_target_variable(), str)
        self.assertEqual(the_result.get_the_target_variable(), 'Churn')

    # test method for get_the_variables_list()
    def test_get_the_variables_list(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a linear model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke method
        self.assertIsNotNone(the_result.get_the_variables_list())
        self.assertIsInstance(the_result.get_the_variables_list(), list)
        self.assertEqual(len(the_result.get_the_variables_list()), 47)

        # loop over every column in the_variable_df.columns
        for the_column in the_variable_df.columns:
            self.assertTrue(the_column in the_result.get_the_variables_list())

        # loop over every column in the_result.get_the_variables_list()
        for the_column in the_result.get_the_variables_list():
            self.assertTrue(the_column in the_variable_df.columns)

    # test method for get_the_p_values()
    def test_get_the_p_values(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a linear model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # validate get_the_variables_list()
        self.assertIsNotNone(the_result.get_the_variables_list())
        self.assertIsInstance(the_result.get_the_variables_list(), list)
        self.assertEqual(len(the_result.get_the_variables_list()), 47)

        # loop over every column in the_variable_df.columns
        for the_column in the_variable_df.columns:
            self.assertTrue(the_column in the_result.get_the_variables_list())

        # loop over every column in the_result.get_the_variables_list()
        for the_column in the_result.get_the_variables_list():
            self.assertTrue(the_column in the_variable_df.columns)

        # invoke the method
        the_lm_dict = the_result.get_the_p_values()

        # run final assertions
        self.assertIsNotNone(the_lm_dict)
        self.assertIsInstance(the_lm_dict, dict)
        self.assertEqual(len(the_lm_dict), len(the_variable_df.columns))

        # save this comment
        # print(the_lm_dict)

        self.assertEqual(the_lm_dict['Population'], 0.88773)
        self.assertEqual(the_lm_dict['TimeZone'], 0.76027)
        self.assertEqual(the_lm_dict['Children'], 0.15518)
        self.assertEqual(the_lm_dict['Age'], 0.3091)
        self.assertEqual(the_lm_dict['Income'], 0.77926)
        self.assertEqual(the_lm_dict['Outage_sec_perweek'], 0.73236)
        self.assertEqual(the_lm_dict['Email'], 0.44978)
        self.assertEqual(the_lm_dict['Contacts'], 0.10584)
        self.assertEqual(the_lm_dict['Yearly_equip_failure'], 0.55349)
        self.assertEqual(the_lm_dict['Techie'], 0.0)
        self.assertEqual(the_lm_dict['Port_modem'], 0.06859)
        self.assertEqual(the_lm_dict['Tablet'], 0.53036)
        self.assertEqual(the_lm_dict['Phone'], 0.02584)
        self.assertEqual(the_lm_dict['Multiple'], 0.07326)
        self.assertEqual(the_lm_dict['OnlineSecurity'], 0.35226)
        self.assertEqual(the_lm_dict['OnlineBackup'], 0.78381)

    # test method for get_the_p_values() with argument
    def test_get_the_p_values_with_argument(self):
        # this test verifies that the less_than argument for get_the_p_values() works correctly.

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # validate get_the_variables_list()
        self.assertIsNotNone(the_result.get_the_variables_list())
        self.assertIsInstance(the_result.get_the_variables_list(), list)
        self.assertEqual(len(the_result.get_the_variables_list()), 47)

        # loop over every column in the_variable_df.columns
        for the_column in the_variable_df.columns:
            self.assertTrue(the_column in the_result.get_the_variables_list())

        # loop over every column in the_result.get_the_variables_list()
        for the_column in the_result.get_the_variables_list():
            self.assertTrue(the_column in the_variable_df.columns)

        # invoke the method
        the_lm_dict = the_result.get_the_p_values(less_than=0.50)

        # run final assertions
        self.assertIsNotNone(the_lm_dict)
        self.assertIsInstance(the_lm_dict, dict)
        self.assertEqual(len(the_lm_dict), 28)

        # save this comment
        # print(the_lm_dict)

        self.assertEqual(the_lm_dict['Children'], 0.15518)
        self.assertEqual(the_lm_dict['Age'], 0.3091)
        self.assertEqual(the_lm_dict['Email'], 0.44978)
        self.assertEqual(the_lm_dict['Contacts'], 0.10584)
        self.assertEqual(the_lm_dict['Techie'], 0.0)
        self.assertEqual(the_lm_dict['Port_modem'], 0.06859)
        self.assertEqual(the_lm_dict['Phone'], 0.02584)
        self.assertEqual(the_lm_dict['Multiple'], 0.07326)
        self.assertEqual(the_lm_dict['OnlineSecurity'], 0.35226)
        self.assertEqual(the_lm_dict['TechSupport'], 0.00887)
        self.assertEqual(the_lm_dict['StreamingTV'], 0.0)
        self.assertEqual(the_lm_dict['StreamingMovies'], 0.0)
        self.assertEqual(the_lm_dict['PaperlessBilling'], 0.03199)
        self.assertEqual(the_lm_dict['MonthlyCharge'], 0.0)
        self.assertEqual(the_lm_dict['Bandwidth_GB_Year'], 0.2174)
        self.assertEqual(the_lm_dict['Reliability'], 0.48256)
        self.assertEqual(the_lm_dict['Options'], 0.42267)
        self.assertEqual(the_lm_dict['Marital_Married'], 0.38935)
        self.assertEqual(the_lm_dict['Marital_Separated'], 0.32863)
        self.assertEqual(the_lm_dict['Marital_Widowed'], 0.03363)
        self.assertEqual(the_lm_dict['Gender_Male'], 0.0019)
        self.assertEqual(the_lm_dict['Contract_One year'], 0.0)
        self.assertEqual(the_lm_dict['Contract_Two Year'], 0.0)
        self.assertEqual(the_lm_dict['InternetService_Fiber Optic'], 3e-05)
        self.assertEqual(the_lm_dict['InternetService_No response'], 0.00451)
        self.assertEqual(the_lm_dict['PaymentMethod_Credit Card (automatic)'], 0.07579)
        self.assertEqual(the_lm_dict['PaymentMethod_Electronic Check'], 0.0)
        self.assertEqual(the_lm_dict['PaymentMethod_Mailed Check'], 0.04081)

    # test method for get_aic_bic_for_model()
    def test_get_aic_bic_for_model(self):
        # this test verifies that the less_than argument for get_the_p_values() works correctly.

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # validate get_the_variables_list()
        self.assertIsNotNone(the_result.get_the_variables_list())
        self.assertIsInstance(the_result.get_the_variables_list(), list)
        self.assertEqual(len(the_result.get_the_variables_list()), 47)

        # loop over every column in the_variable_df.columns
        for the_column in the_variable_df.columns:
            self.assertTrue(the_column in the_result.get_the_variables_list())

        # loop over every column in the_result.get_the_variables_list()
        for the_column in the_result.get_the_variables_list():
            self.assertTrue(the_column in the_variable_df.columns)

        # invoke the method
        the_tuple = the_result.get_aic_bic_for_model()

        # run assertions
        self.assertIsNotNone(the_tuple)
        self.assertIsInstance(the_tuple, tuple)
        self.assertEqual(the_tuple[0], ('AIC', 4438.24535171725))
        self.assertEqual(the_tuple[1], ('BIC', 4784.341689572107))

    # negative test method for get_vif_for_model()
    def test_get_vif_for_model_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # even though we constructed the_variable_columns to be all 53 remaining variables, let's override
        # that to a much smaller set.
        the_variable_columns = ['Tenure', 'MonthlyCharge']

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 2)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # validate get_the_variables_list()
        self.assertIsNotNone(the_result.get_the_variables_list())
        self.assertIsInstance(the_result.get_the_variables_list(), list)
        self.assertEqual(len(the_result.get_the_variables_list()), 2)

        # loop over every column in the_variable_df.columns
        for the_column in the_variable_df.columns:
            self.assertTrue(the_column in the_result.get_the_variables_list())

        # loop over every column in the_result.get_the_variables_list()
        for the_column in the_result.get_the_variables_list():
            self.assertTrue(the_column in the_variable_df.columns)

        # invoke get_pseudo_r_squared() to make sure we haven't screwed something up
        self.assertEqual(the_result.get_pseudo_r_squared(), 0.40853)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_result.get_vif_for_model(the_encoded_df=None)

            # validate the error message.
            self.assertTrue("the_encoded_df is None or incorrect type." in context.exception)

        # verify we handle pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_result.get_vif_for_model(the_encoded_df=pd.DataFrame())

            # validate the error message.
            self.assertTrue("[Tenure] not present in the_encoded_df." in context.exception)

    # test method for get_vif_for_model()
    def test_get_vif_for_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # even though we constructed the_variable_columns to be all 47 remaining variables, let's override
        # that to a much smaller set.
        the_variable_columns = ['Tenure', 'MonthlyCharge']

        the_target_series = logistic_model.encoded_df['Churn']
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 2)

        # cast data to int
        x_val = logistic_model.encoded_df[the_variable_columns].astype(int)

        # get the target series
        y_val = logistic_model.encoded_df['Churn'].astype(int)

        # get the
        x_con = sm.add_constant(x_val)

        logistic_regression = sm.Logit(y_val, x_con)

        fitted_model = logistic_regression.fit()

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(the_regression_wrapper=fitted_model,
                                           the_target_variable='Churn',
                                           the_variables_list=the_variable_columns,
                                           the_df=x_val)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # validate get_the_variables_list() returns back the list of features
        self.assertIsNotNone(the_result.get_the_variables_list())
        self.assertIsInstance(the_result.get_the_variables_list(), list)
        self.assertEqual(len(the_result.get_the_variables_list()), 2)

        # loop over every column in the_variable_df.columns
        for the_column in the_variable_df.columns:
            self.assertTrue(the_column in the_result.get_the_variables_list())

        # loop over every column in the_result.get_the_variables_list()
        for the_column in the_result.get_the_variables_list():
            self.assertTrue(the_column in the_variable_df.columns)

        # invoke get_pseudo_r_squared() to make sure we haven't screwed something up
        self.assertEqual(the_result.get_pseudo_r_squared(), 0.40853)

        # invoke the method
        vif_df = the_result.get_vif_for_model(the_encoded_df=x_val)

        # run assertions
        self.assertIsNotNone(vif_df)
        self.assertIsInstance(vif_df, DataFrame)
        self.assertEqual(vif_df.iloc[0]['VIF'], 2.4154295780772395)
        self.assertEqual(vif_df.iloc[1]['VIF'], 2.415429578077241)

    # test method for get_vif_for_model for entire dataset
    def test_get_vif_for_model_full_dataset(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # create the_variable_df
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # fit a model to get a linear_model_result
        the_lm_result = logistic_model.fit_a_model(the_target_column="Churn",
                                                   current_features=the_variable_columns,
                                                   model_type=MT_LOGISTIC_REGRESSION)

        the_variable_df = the_variable_df.astype(float)

        # get the_vif_df
        the_vif_df = the_lm_result.get_vif_for_model(the_variable_df)

        # run assertions
        self.assertIsNotNone(the_vif_df)
        self.assertIsInstance(the_vif_df, DataFrame)

        # keep this commented out in case you change the structure of the dataframe.
        # print(the_vif_df)

        # run assertions
        self.assertEqual(the_vif_df.iloc[0, 0], 'Population')
        self.assertEqual(the_vif_df.iloc[0, 1].round(6), 1.467739)
        self.assertEqual(the_vif_df.iloc[3, 0], 'Age')
        self.assertEqual(the_vif_df.iloc[3, 1].round(6), 80.391741)
        self.assertEqual(the_vif_df.iloc[5, 0], 'Outage_sec_perweek')
        self.assertEqual(the_vif_df.iloc[5, 1].round(6), 12.242200)
        self.assertEqual(the_vif_df.iloc[21, 0], 'Tenure')
        self.assertEqual(the_vif_df.iloc[21, 1].round(6), 28979.155923)
        self.assertEqual(the_vif_df.iloc[22, 0], 'MonthlyCharge')
        self.assertEqual(the_vif_df.iloc[22, 1].round(6), 1506.738469)
        self.assertEqual(the_vif_df.iloc[23, 0], 'Bandwidth_GB_Year')
        self.assertEqual(the_vif_df.iloc[23, 1].round(6), 37112.054430)
        self.assertEqual(the_vif_df.iloc[42, 0], 'InternetService_Fiber Optic')
        self.assertEqual(the_vif_df.iloc[42, 1].round(6), 247.628909)
        self.assertEqual(the_vif_df.iloc[43, 0], 'InternetService_No response')
        self.assertEqual(the_vif_df.iloc[43, 1].round(6), 63.820239)

        # validate feature with the largest VIF
        self.assertEqual(the_vif_df['VIF'].idxmax(), 23)
        self.assertEqual(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'], 'Bandwidth_GB_Year')

        # remove 'Bandwidth_GB_Year' from the_variable_columns
        the_variable_columns.remove(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'])

        # run assertion to make sure the feature is not present
        self.assertFalse('Bandwidth_GB_Year' in the_variable_columns)
        self.assertEqual(len(the_variable_columns), 46)

        # create the_variable_df
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # fit a model to get a linear_model_result
        the_lm_result = logistic_model.fit_a_model(the_target_column="Churn",
                                                   current_features=the_variable_columns,
                                                   model_type=MT_LOGISTIC_REGRESSION)

        # cast to float
        the_variable_df = the_variable_df.astype(float)

        # get the_vif_df
        the_vif_df = the_lm_result.get_vif_for_model(the_variable_df)

        # run assertions
        self.assertIsNotNone(the_vif_df)
        self.assertIsInstance(the_vif_df, DataFrame)

        # keep this commented out in case you change the structure of the dataframe.
        # print(the_vif_df)

        self.assertEqual(the_vif_df['VIF'].idxmax(), 22)
        self.assertEqual(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'], 'MonthlyCharge')

        # remove 'MonthlyCharge' from the_variable_columns
        the_variable_columns.remove(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'])

        # run assertion to make sure the feature is not present
        self.assertFalse('MonthlyCharge' in the_variable_columns)
        self.assertEqual(len(the_variable_columns), 45)

        # create the_variable_df
        the_variable_df = logistic_model.encoded_df[the_variable_columns]

        # fit a model to get a linear_model_result
        the_lm_result = logistic_model.fit_a_model(the_target_column="Churn",
                                                   current_features=the_variable_columns,
                                                   model_type=MT_LOGISTIC_REGRESSION)

        # cast to float
        the_variable_df = the_variable_df.astype(float)

        # get the_vif_df
        the_vif_df = the_lm_result.get_vif_for_model(the_variable_df)

        # run assertions
        self.assertIsNotNone(the_vif_df)
        self.assertIsInstance(the_vif_df, DataFrame)

        # keep this commented out in case you change the structure of the dataframe.
        print(the_vif_df)

        self.assertEqual(the_vif_df['VIF'].idxmax(), 1)
        self.assertEqual(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'], 'TimeZone')

    # test method for get_results_dataframe()
    def test_get_results_dataframe(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                             current_features=current_features,
                                                             model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Logistic_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # DO NOT DELETE THIS....  START
        # print(f"linear_model.model.params-->\n{the_linear_model_result.model.params}")
        # print(f"type(linear_model.model.params)-->\n{type(the_linear_model_result.model.params)}")
        # DO NOT DELETE THIS....  END

        # validate the default status of get_feature_columns()
        self.assertIsNone(the_linear_model_result.get_feature_columns())

        # invoke the test method
        the_result_df = the_linear_model_result.get_results_dataframe()

        # run assertions
        self.assertIsNotNone(the_result_df)
        self.assertIsInstance(the_result_df, DataFrame)

        # need to leave LM_FEATURE_NUM off
        self.assertEqual(list(the_result_df.columns), [LM_FEATURE_NUM, LM_COEFFICIENT, LM_STANDARD_ERROR,
                                                       LM_T_STATISTIC, LM_P_VALUE, LM_LS_CONF_INT, LM_RS_CONF_INT,
                                                       LM_VIF])

        self.assertEqual(the_result_df.index.name, LM_PREDICTOR)

        # do not delete this commented out print line
        # print(the_result_df['p-value'].head(7))

        # run specific p-value tests.
        self.assertEqual(the_result_df.loc['Population', 'p-value'], 0.88773)
        self.assertEqual(the_result_df.loc['TimeZone', 'p-value'], 0.76027)
        self.assertEqual(the_result_df.loc['Children', 'p-value'], 0.15518)
        self.assertEqual(the_result_df.loc['Age', 'p-value'], 0.30910)
        self.assertEqual(the_result_df.loc['Income', 'p-value'], 0.77926)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'p-value'], 0.73236)
        self.assertEqual(the_result_df.loc['Email', 'p-value'], 0.44978)

        # do not delete this commented out print line
        # print(the_result_df['coefficient'].head(7).round(7))

        # run specific coefficient tests.
        self.assertEqual(the_result_df.loc['Population', 'coefficient'].round(7), -4e-07)
        self.assertEqual(the_result_df.loc['TimeZone', 'coefficient'].round(7), -1.190240e-02)
        self.assertEqual(the_result_df.loc['Children', 'coefficient'].round(7), 7.162100e-02)
        self.assertEqual(the_result_df.loc['Age', 'coefficient'].round(7), -5.458600e-03)
        self.assertEqual(the_result_df.loc['Income', 'coefficient'].round(7), 4.000000e-07)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'coefficient'].round(7), -4.447700e-03)
        self.assertEqual(the_result_df.loc['Email', 'coefficient'].round(7), -9.614200e-03)

        # do not delete this commented out print line
        # print(the_result_df['std err'].head(7).round(7))

        # run standard error tests
        self.assertEqual(the_result_df.loc['Population', 'std err'].round(6), 3e-06)
        self.assertEqual(the_result_df.loc['TimeZone', 'std err'].round(6), 0.039008)
        self.assertEqual(the_result_df.loc['Children', 'std err'].round(6), 0.050385)
        self.assertEqual(the_result_df.loc['Age', 'std err'].round(6), 0.005367)
        self.assertEqual(the_result_df.loc['Income', 'std err'].round(6), 0.000001)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'std err'].round(6), 0.013005)
        self.assertEqual(the_result_df.loc['Email', 'std err'].round(6), 0.012721)

        # do not delete this commented out print line
        # print(the_result_df['t-statistic'].head(7).round(7))

        # run t-statistic tests
        self.assertEqual(the_result_df.loc['Population', 't-statistic'].round(6), -0.141175)
        self.assertEqual(the_result_df.loc['TimeZone', 't-statistic'].round(6), -0.305127)
        self.assertEqual(the_result_df.loc['Children', 't-statistic'].round(6), 1.421484)
        self.assertEqual(the_result_df.loc['Age', 't-statistic'].round(6), -1.017118)
        self.assertEqual(the_result_df.loc['Income', 't-statistic'].round(6), 0.280285)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 't-statistic'].round(6), -0.341994)
        self.assertEqual(the_result_df.loc['Email', 't-statistic'].round(6), -0.755776)

        # do not delete this commented out print line
        # print(the_result_df['[0.025'].head(7).round(7))

        # run '[0.025' tests
        self.assertEqual(the_result_df.loc['Population', '[0.025'].round(6), -0.000006)
        self.assertEqual(the_result_df.loc['TimeZone', '[0.025'].round(6), -0.088356)
        self.assertEqual(the_result_df.loc['Children', '[0.025'].round(6), -0.027131)
        self.assertEqual(the_result_df.loc['Age', '[0.025'].round(6), -0.015977)
        self.assertEqual(the_result_df.loc['Income', '[0.025'].round(6), -0.000002)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', '[0.025'].round(6), -0.029937)
        self.assertEqual(the_result_df.loc['Email', '[0.025'].round(6), -0.034547)

        # do not delete this commented out print line
        # print(the_result_df['0.975]'].head(7).round(7))

        # run '0.975]' tests
        self.assertEqual(the_result_df.loc['Population', '0.975]'].round(6), 0.000005)
        self.assertEqual(the_result_df.loc['TimeZone', '0.975]'].round(6), 0.064552)
        self.assertEqual(the_result_df.loc['Children', '0.975]'].round(6), 0.170373)
        self.assertEqual(the_result_df.loc['Age', '0.975]'].round(6), 0.005060)
        self.assertEqual(the_result_df.loc['Income', '0.975]'].round(6), 0.000003)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', '0.975]'].round(6), 0.021042)
        self.assertEqual(the_result_df.loc['Email', '0.975]'].round(6), 0.015318)

        # do not delete this commented out print line
        # print(the_result_df['VIF'].head(14).round(7))

        # run specific VIF tests
        self.assertEqual(the_result_df.loc['Population', 'VIF'].round(6), 1.467507)
        self.assertEqual(the_result_df.loc['TimeZone', 'VIF'].round(6), 33.647461)
        self.assertEqual(the_result_df.loc['Children', 'VIF'].round(6), 9.872871)
        self.assertEqual(the_result_df.loc['Age', 'VIF'].round(6), 36.081682)
        self.assertEqual(the_result_df.loc['Income', 'VIF'].round(6), 2.996354)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'VIF'].round(6), 10.835894)
        self.assertEqual(the_result_df.loc['Email', 'VIF'].round(6), 16.292335)
        self.assertEqual(the_result_df.loc['Contacts', 'VIF'].round(6), 2.015473)
        self.assertEqual(the_result_df.loc['Yearly_equip_failure', 'VIF'].round(6), 1.395444)
        self.assertEqual(the_result_df.loc['Techie', 'VIF'].round(6), 1.205912)
        self.assertEqual(the_result_df.loc['Port_modem', 'VIF'].round(6), 1.935425)
        self.assertEqual(the_result_df.loc['Tablet', 'VIF'].round(6), 1.434380)
        self.assertEqual(the_result_df.loc['Phone', 'VIF'].round(6), 10.534325)
        self.assertEqual(the_result_df.loc['Multiple', 'VIF'].round(6), 8.035171)

        # validate that status of get_feature_columns()
        self.assertIsNotNone(the_linear_model_result.get_feature_columns())
        self.assertIsInstance(the_linear_model_result.get_feature_columns(), list)

        # capture the feature columns list
        feature_column_list = the_linear_model_result.get_feature_columns()

        # run assertions on feature_column_list
        self.assertEqual(len(feature_column_list), 8)
        self.assertTrue(LM_FEATURE_NUM in feature_column_list)
        self.assertTrue(LM_COEFFICIENT in feature_column_list)
        self.assertTrue(LM_STANDARD_ERROR in feature_column_list)
        self.assertTrue(LM_T_STATISTIC in feature_column_list)
        self.assertTrue(LM_P_VALUE in feature_column_list)
        self.assertTrue(LM_LS_CONF_INT in feature_column_list)
        self.assertTrue(LM_RS_CONF_INT in feature_column_list)
        self.assertTrue(LM_VIF in feature_column_list)

    # test to iterate over the results dataframe
    def test_iterate_over_results_dataframe(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                             current_features=current_features,
                                                             model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Logistic_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # invoke the test method
        the_result_df = the_linear_model_result.get_results_dataframe()

        # run assertions
        self.assertIsNotNone(the_result_df)
        self.assertIsInstance(the_result_df, DataFrame)
        # need to leave LM_FEATURE_NUM off
        self.assertEqual(list(the_result_df.columns), [LM_FEATURE_NUM, LM_COEFFICIENT, LM_STANDARD_ERROR,
                                                       LM_T_STATISTIC, LM_P_VALUE, LM_LS_CONF_INT, LM_RS_CONF_INT,
                                                       LM_VIF])
        self.assertEqual(the_result_df.index.name, LM_PREDICTOR)

        # get the list of features
        for the_feature in the_linear_model_result.get_the_variables_list():
            print(f"feature[{the_feature}][{the_result_df[LM_P_VALUE].loc[the_feature]}]")

    # test method for get_constant()
    def test_get_constant(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                             current_features=current_features,
                                                             model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Logistic_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # get the constant
        self.assertEqual(the_linear_model_result.get_constant(), -4.254033191870079)

    # negative test method for are_p_values_above_threshold()
    def test_are_p_values_above_threshold_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                               current_features=current_features,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)
        self.assertIsInstance(the_logistic_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_logistic_model_result.get_the_p_values()), 47)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_logistic_model_result.are_p_values_above_threshold(p_value=None)

            # validate the error message.
            self.assertTrue("p_value was None or incorrect type." in context.exception)

        # verify we handle 2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_logistic_model_result.are_p_values_above_threshold(p_value=2.0)

            # validate the error message.
            self.assertTrue("p_value was greater than 1.0" in context.exception)

    # test method for are_p_values_above_threshold()
    def test_are_p_values_above_threshold(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                               current_features=current_features,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)
        self.assertIsInstance(the_logistic_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_logistic_model_result.get_the_p_values()), 47)

        # invoke the method with default p_value argument of 1.0.  All the p-values should be below
        # the threshold.
        self.assertFalse(the_logistic_model_result.are_p_values_above_threshold())

        # invoke the method with default p_value argument of 0.50.
        self.assertTrue(the_logistic_model_result.are_p_values_above_threshold(p_value=0.50))

        # pick results that are < 0.50
        the_results_df = the_logistic_model_result.get_results_dataframe()

        # run assertions on the_results_df
        self.assertIsNotNone(the_results_df)
        self.assertIsInstance(the_results_df, DataFrame)

        # retrieve rows < 0.50
        smaller_result_df = the_results_df[the_results_df['p-value'] < 0.50]

        # run assertions on smaller_result_df
        self.assertIsNotNone(smaller_result_df)
        self.assertIsInstance(smaller_result_df, DataFrame)
        self.assertTrue(len(smaller_result_df) > 0)
        self.assertTrue(smaller_result_df['p-value'].max() < 0.50)

    # negative test method for identify_parameter_based_on_p_value()
    def test_identify_parameter_based_on_p_value_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on logistic_model to make sure the encoded_df property is setup
        self.assertIsNotNone(logistic_model.encoded_df)
        self.assertIsInstance(logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_logistic_model_result.identify_parameter_based_on_p_value(p_value=None)

            # validate the error message.
            self.assertTrue("p_value was None or incorrect type." in context.exception)

        # verify we handle 2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_logistic_model_result.identify_parameter_based_on_p_value(p_value=2.0)

            # validate the error message.
            self.assertTrue("p_value was greater than 1.0" in context.exception)

    # test method for identify_parameter_based_on_p_value()
    def test_identify_parameter_based_on_p_value(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on logistic_model to make sure the encoded_df property is setup
        self.assertIsNotNone(logistic_model.encoded_df)
        self.assertIsInstance(logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_linear_model_result = logistic_model.fit_a_model(the_target_column='Churn',
                                                             current_features=the_variable_columns,
                                                             model_type=MT_LOGISTIC_REGRESSION)

        # get the next feature to remove
        the_feature = the_linear_model_result.identify_parameter_based_on_p_value(p_value=0.05)

        # run assertions
        self.assertIsNotNone(the_feature)
        self.assertIsInstance(the_feature, str)
        self.assertTrue(the_feature in the_variable_columns)
        self.assertEqual(the_feature, 'Courteous_Exchange')

    # negative test for is_vif_above_threshold()
    def test_is_vif_above_threshold_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on logistic_model to make sure the encoded_df property is setup
        self.assertIsNotNone(logistic_model.encoded_df)
        self.assertIsInstance(logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_logistic_model_result.is_vif_above_threshold(max_allowable_vif=None)

            # validate the error message.
            self.assertTrue("max_allowable_vif was None or incorrect type." in context.exception)

        # verify we handle "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_logistic_model_result.is_vif_above_threshold(max_allowable_vif="foo")

            # validate the error message.
            self.assertTrue("max_allowable_vif was None or incorrect type." in context.exception)

        # verify we handle -1.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_logistic_model_result.is_vif_above_threshold(max_allowable_vif=-1.0)

            # validate the error message.
            self.assertTrue("max_allowable_vif was less than 1.0" in context.exception)

    # test method for is_vif_above_threshold()
    def test_is_vif_above_threshold(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on logistic_model to make sure the encoded_df property is setup
        self.assertIsNotNone(logistic_model.encoded_df)
        self.assertIsInstance(logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(logistic_model.encoded_df.columns), 48)

        # get the list of columns from logistic_model
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Logistic_Model_Result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)

        # get the current list of variables from the the_linear_model_result
        current_variable_columns = the_logistic_model_result.get_the_variables_list()

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in current_variable_columns)

        for the_column in current_variable_columns:
            self.assertTrue(the_column in the_variable_columns)

        # invoke the method
        self.assertTrue(the_logistic_model_result.is_vif_above_threshold(max_allowable_vif=10.0))

        # get the dataframe
        the_results_df = the_logistic_model_result.get_results_dataframe()

        # make sure the columns in the_results_df match current_variable_columns
        for the_column in the_results_df.index.values.tolist():
            self.assertTrue(the_column in current_variable_columns)

        for the_column in current_variable_columns:
            self.assertTrue(the_column in the_results_df.index.values.tolist())

        # print(the_results_df.loc['InternetService_No response']['VIF'].round(6))

        # run verifications that the base VIF results for the first model are correct.
        # these are features with a large VIF
        self.assertEqual(the_results_df.loc['Bandwidth_GB_Year']['VIF'].round(6), 14594.033777)
        self.assertEqual(the_results_df.loc['Tenure']['VIF'].round(6), 11194.512723)
        self.assertEqual(the_results_df.loc['StreamingTV']['VIF'].round(6), 10.958312)
        self.assertEqual(the_results_df.loc['StreamingMovies']['VIF'].round(6), 13.903012)
        self.assertEqual(the_results_df.loc['InternetService_Fiber Optic']['VIF'].round(6), 100.77826)
        self.assertEqual(the_results_df.loc['InternetService_No response']['VIF'].round(6), 25.984759)

        # ****************************************************************
        # remove Bandwidth_GB_Year from the_variable_columns
        the_variable_columns.remove('Bandwidth_GB_Year')

        # run an assertion to prove the remove worked correctly
        self.assertTrue('Bandwidth_GB_Year' not in the_variable_columns)

        # get a Linear_Model_Result
        the_linear_model_result = logistic_model.fit_a_model(the_target_column='Churn',
                                                             current_features=the_variable_columns,
                                                             model_type=MT_LOGISTIC_REGRESSION)

        # invoke the method again
        self.assertTrue(the_linear_model_result.is_vif_above_threshold(max_allowable_vif=10.0))

        # get the the_results_df from the current the_linear_model_result instance
        the_results_df = the_linear_model_result.get_results_dataframe()

        self.assertEqual(the_results_df.loc['Tenure']['VIF'].round(6), 2.653109)
        self.assertEqual(the_results_df.loc['StreamingTV']['VIF'].round(6), 9.854353)
        self.assertEqual(the_results_df.loc['StreamingMovies']['VIF'].round(6), 13.851972)
        self.assertEqual(the_results_df.loc['InternetService_Fiber Optic']['VIF'].round(6), 3.978732)
        self.assertEqual(the_results_df.loc['InternetService_No response']['VIF'].round(6), 1.847708)

        # Thus, this test proves that removing a single variable with the highest VIF can clean up
        # and lower the multi-collinearity issues for a logistics model

    # test method for get_feature_with_max_vif()
    def test_get_feature_with_max_vif(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on logistic_model to make sure the encoded_df property is setup
        self.assertIsNotNone(logistic_model.encoded_df)
        self.assertIsInstance(logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(logistic_model.encoded_df.columns), 48)

        # get the list of columns from logistic_model
        the_variable_columns = logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Logistic_Model_Result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)

        # get the current list of variables from the the_linear_model_result
        current_variable_columns = the_logistic_model_result.get_the_variables_list()

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in current_variable_columns)

        for the_column in current_variable_columns:
            self.assertTrue(the_column in the_variable_columns)

        # check that we actually have a feature above max_allowable_vif
        self.assertTrue(the_logistic_model_result.is_vif_above_threshold(max_allowable_vif=10.0))

        # invoke the method
        the_feature = the_logistic_model_result.get_feature_with_max_vif()

        # run assertions
        self.assertEqual(the_feature, "Bandwidth_GB_Year")

    # proof of concept method for confusion matrix
    def test_proof_of_concept_on_confusion_matrix(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001, the_max_vif=7.0)

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
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # make sure the model result stored is the same.
        self.assertTrue(len(the_lm_result.get_results_dataframe()) > 0)
        self.assertTrue(the_lm_result.get_results_dataframe()['p-value'].max() < 0.001)
        self.assertEqual(len(the_lm_result.get_results_dataframe()), 14)

        # loop over columns of results_df and compare to the stored linear model result
        for the_column in results_df.columns:
            self.assertTrue(the_column in the_logistic_model.get_the_result().get_results_dataframe().columns)

        # this scenario only works if I fully setup the environment to match main(). At this point
        # I am going to assume that pa.build_model() is where the function will ultimately be called.
        # this test case is slightly flawed as the actual call will occur before outliers are removed.
        # another test case will have to be built following this to ensure it actually works prior
        # to the first invocation of pa.build_model().

        # get the actual model from the the_lm_result
        the_model = the_lm_result.get_model()

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, BinaryResultsWrapper)

        # define the_variable_columns
        the_variable_columns = the_lm_result.get_the_variables_list()

        # run assertions on the_variable_columns
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertFalse('Churn' in the_variable_columns)

        # get the underlying dataframe
        the_df = the_logistic_model.encoded_df

        # run assertions on the_df
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df), 10000)
        self.assertTrue('Churn' in the_df.columns)

        # performing predictions on the test dataset
        the_prediction = the_model.predict()

        # run assertions on the prediction
        self.assertIsNotNone(the_prediction)
        self.assertIsInstance(the_prediction, ndarray)

        # convert back to booleans
        predictions_boolean = [False if x < 0.5 else True for x in the_prediction]

        # create confusion matrix
        c_matrix = metrics.confusion_matrix(the_df['Churn'], predictions_boolean)

        # run assertions on c_matrix
        self.assertIsNotNone(c_matrix)
        self.assertIsInstance(c_matrix, numpy.ndarray)

        # print(f"\n**********************************************\n")
        # print(f"type is -->{type(c_matrix)}")

        y_actual = pd.Series(the_df['Churn'], name='Actual')
        y_predicted = pd.Series(predictions_boolean, name='Predicted')

        # create confusion matrix
        # print(f"confusion matrix")
        # print(pd.crosstab(y_actual, y_predicted))
        #
        # # print accuracy of model
        # print(f"accuracy of model")
        # print(metrics.accuracy_score(y_actual, y_predicted))
        #
        # # print precision value of model
        # print(f"precision value of model")
        # print(metrics.precision_score(y_actual, y_predicted))
        #
        # # print recall value of model
        # print(f"recall value of model")
        # print(metrics.recall_score(y_actual, y_predicted))

        # clear everything
        plt.clf()

        group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts = ['{0: 0.0f}'.format(value) for value in c_matrix.flatten()]
        group_percentages = ["{0: .2%}".format(value) for value in c_matrix.flatten() / np.sum(c_matrix)]
        labels = [f'{v1}\n {v2}\n {v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(c_matrix, annot=labels, fmt="", cmap="Blues")

        # add title
        plt.title("Heatmap of confusion matrix")

        # this command forces the full names to show-up.  If you don't invoke this command, longer variable names
        # will be cut off and the general image will be hard to read.
        plt.tight_layout()

        # save the plot to the file system.
        plt.savefig(self.OVERRIDE_PATH + "confusion_matrix.png")
        plt.close()

    # test method for get_confusion_matrix()
    def test_get_confusion_matrix(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # invoke the method
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION,
                       max_p_value=0.001, the_max_vif=7.0)

        # run assertions on the model storage
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_INITIAL_MODEL], Logistic_Model)
        self.assertIsNotNone(pa.analyzer.linear_model_storage[LM_FINAL_MODEL])
        self.assertIsInstance(pa.analyzer.linear_model_storage[LM_FINAL_MODEL], Logistic_Model)

        # get a reference to the final the_logistic_model
        the_logistic_model = pa.analyzer.linear_model_storage[LM_FINAL_MODEL]

        # get the Logistic_Model_Result
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

        # invoke the method
        the_confusion_matrix = the_lm_result.get_confusion_matrix(the_encoded_df=the_df)

        # print(f"the_confusion_matrix_df-->\n{the_confusion_matrix_df}")

        # run assertions
        self.assertIsNotNone(the_confusion_matrix)
        self.assertIsInstance(the_confusion_matrix, ndarray)

        # run assertions on stored accuracy values
        self.assertEqual(the_lm_result.accuracy_score, 0.9038)
        self.assertEqual(the_lm_result.precision_score, 0.8294301327088213)
        self.assertEqual(the_lm_result.recall_score, 0.8018867924528302)
        self.assertEqual(the_lm_result.f1_score, 0.8154259401381427)

    # test method for get_assumptions()
    def test_get_assumptions(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # check what they are
        self.assertTrue(PSEUDO_R_SQUARED_HEADER in the_assumptions)
        self.assertTrue(MODEL_ACCURACY in the_assumptions)
        self.assertTrue(MODEL_PRECISION in the_assumptions)
        self.assertTrue(MODEL_RECALL in the_assumptions)
        self.assertTrue(MODEL_F1_SCORE in the_assumptions)
        self.assertTrue(AIC_SCORE in the_assumptions)
        self.assertTrue(BIC_SCORE in the_assumptions)
        self.assertTrue(LOG_LIKELIHOOD in the_assumptions)
        self.assertTrue(NUMBER_OF_OBS in the_assumptions)
        self.assertTrue(DEGREES_OF_FREEDOM_MODEL in the_assumptions)

        # check the value
        self.assertEqual(the_assumptions[PSEUDO_R_SQUARED_HEADER], 'get_pseudo_r_squared')
        self.assertEqual(the_assumptions[MODEL_ACCURACY], 'get_accuracy_score')
        self.assertEqual(the_assumptions[MODEL_PRECISION], 'get_precision_score')
        self.assertEqual(the_assumptions[MODEL_RECALL], 'get_recall_score')
        self.assertEqual(the_assumptions[MODEL_F1_SCORE], 'get_f1_score')
        self.assertEqual(the_assumptions[AIC_SCORE], 'get_aic_for_model')
        self.assertEqual(the_assumptions[BIC_SCORE], 'get_bic_for_model')
        self.assertEqual(the_assumptions[LOG_LIKELIHOOD], 'get_log_likelihood')
        self.assertEqual(the_assumptions[NUMBER_OF_OBS], 'get_number_of_obs')
        self.assertEqual(the_assumptions[DEGREES_OF_FREEDOM_MODEL], 'get_degrees_of_freedom_for_model')

    # test method for get_aic_for_model()
    def test_get_aic_for_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions on assumptions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # run assertions on method
        self.assertEqual(the_result.get_aic_for_model(), 4438.24535171725)

    # test method for get_bic_for_model()
    def test_get_bic_for_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions on assumptions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # run assertions on method
        self.assertEqual(the_result.get_bic_for_model(), 4784.341689572107)

    # test method for get_log_likelihood()
    def test_get_log_likelihood(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions on assumptions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # run assertions on method
        self.assertEqual(the_result.get_log_likelihood(), -2171.122675858625)

    # test method for get_number_of_obs()
    def test_get_number_of_obs(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions on assumptions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # run assertions on method
        self.assertEqual(the_result.get_number_of_obs(), 10000)

    # test method for get_degrees_of_freedom_for_model()
    def test_get_degrees_of_freedom_for_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions on assumptions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # run assertions on method
        self.assertEqual(the_result.get_degrees_of_freedom_for_model(), 47)

    # test method for get_degrees_of_freedom_for_residuals()
    def test_get_degrees_of_freedom_for_residuals(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                  the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Logistic_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions on assumptions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # run assertions on method
        self.assertEqual(the_result.get_degrees_of_freedom_for_residuals(), 9952)

    # negative test method for get_assumption_result()
    def test_get_assumption_result_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_logistic_model.fit_model(the_target_column='Churn',
                                                     the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # call get_assumptions()
        the_assumptions = the_lm_result.get_assumptions()

        # run assertions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_lm_result.get_assumption_result(the_assumption=None)

            # validate the error message.
            self.assertTrue("the_assumption is None or incorrect type." in context.exception)

        # verify we handle 'foo'
        with self.assertRaises(KeyError) as context:
            # invoke the method
            the_lm_result.get_assumption_result(the_assumption='foo')

            # validate the error message.
            self.assertTrue("the_assumption is not present in storage." in context.exception)

        # add 'foo' to the assumption dictionary
        the_lm_result.assumptions['foo'] = 'get_foo'

        # verify we handle 'foo'
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            the_lm_result.get_assumption_result(the_assumption='foo')

            # validate the error message.
            self.assertTrue("the_assumption is not implemented." in context.exception)

    # test method for get_assumption_result()
    def test_get_assumption_result(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_logistic_model.fit_a_model(the_target_column='Churn',
                                                       current_features=method_results,
                                                       model_type=MT_LOGISTIC_REGRESSION)

        # run basic assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # call get_assumptions()
        the_assumptions = the_lm_result.get_assumptions()

        # run assertions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 12)

        # call the confusion matrix
        the_lm_result.get_confusion_matrix(the_encoded_df=the_logistic_model.encoded_df)

        # run assertions
        self.assertEqual(the_lm_result.get_assumption_result(PSEUDO_R_SQUARED_HEADER), 0.62452)
        self.assertEqual(the_lm_result.get_assumption_result(MODEL_ACCURACY), 0.9043)
        self.assertEqual(the_lm_result.get_assumption_result(MODEL_PRECISION), 0.8282279953470337)
        self.assertEqual(the_lm_result.get_assumption_result(MODEL_RECALL), 0.8060377358490566)
        self.assertEqual(the_lm_result.get_assumption_result(MODEL_F1_SCORE), 0.816982214572576)
        self.assertEqual(the_lm_result.get_assumption_result(AIC_SCORE), 4438.24535171725)
        self.assertEqual(the_lm_result.get_assumption_result(BIC_SCORE), 4784.341689572107)
        self.assertEqual(the_lm_result.get_assumption_result(LOG_LIKELIHOOD), -2171.122675858625)
        self.assertEqual(the_lm_result.get_assumption_result(NUMBER_OF_OBS), 10000)
        self.assertEqual(the_lm_result.get_assumption_result(DEGREES_OF_FREEDOM_MODEL), 47)
        self.assertEqual(the_lm_result.get_assumption_result(DEGREES_OF_FREEDOM_RESID), 9952)
        self.assertEqual(the_lm_result.get_assumption_result(MODEL_CONSTANT), -4.254033191870079)

    # negative tests for has_assumption() method
    def test_has_assumption_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_logistic_model.fit_a_model(the_target_column='Churn',
                                                       current_features=method_results,
                                                       model_type=MT_LOGISTIC_REGRESSION)

        # run basic assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # verify we handle 'None'
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_lm_result.has_assumption(the_assumption=None)

            # validate the error message.
            self.assertTrue("the_assumption is None or incorrect type." in context.exception)

    # test method for get_assumption_result()
    def test_has_assumption(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a Logistic_Model
        the_logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_logistic_model.encoded_df)
        self.assertIsInstance(the_logistic_model.encoded_df, DataFrame)
        self.assertEqual(len(the_logistic_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_logistic_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_logistic_model.get_encoded_variables('Churn')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('Churn' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_logistic_model.fit_a_model(the_target_column='Churn',
                                                       current_features=method_results,
                                                       model_type=MT_LOGISTIC_REGRESSION)

        # run basic assertions on the_lm_result
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Logistic_Model_Result)

        # final assertions
        self.assertTrue(the_lm_result.has_assumption(PSEUDO_R_SQUARED_HEADER))
        self.assertTrue(the_lm_result.has_assumption(MODEL_ACCURACY))
        self.assertTrue(the_lm_result.has_assumption(MODEL_PRECISION))
        self.assertTrue(the_lm_result.has_assumption(MODEL_RECALL))
        self.assertTrue(the_lm_result.has_assumption(MODEL_F1_SCORE))
        self.assertTrue(the_lm_result.has_assumption(AIC_SCORE))
        self.assertTrue(the_lm_result.has_assumption(BIC_SCORE))
        self.assertTrue(the_lm_result.has_assumption(LOG_LIKELIHOOD))
        self.assertTrue(the_lm_result.has_assumption(NUMBER_OF_OBS))

    # test method for get_feature_columns()
    def test_get_feature_columns(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                             current_features=current_features,
                                                             model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Logistic_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # validate that the method returns none prior to invocation of get_feature_columns()
        self.assertIsNone(the_linear_model_result.get_feature_columns())

        # invoke get_results_dataframe()
        the_linear_model_result.get_results_dataframe()

        # validate that status of get_feature_columns()
        self.assertIsNotNone(the_linear_model_result.get_feature_columns())
        self.assertIsInstance(the_linear_model_result.get_feature_columns(), list)

        # capture the feature columns list
        feature_column_list = the_linear_model_result.get_feature_columns()

        # run assertions on feature_column_list
        self.assertEqual(len(feature_column_list), 8)
        self.assertTrue(LM_FEATURE_NUM in feature_column_list)
        self.assertTrue(LM_COEFFICIENT in feature_column_list)
        self.assertTrue(LM_STANDARD_ERROR in feature_column_list)
        self.assertTrue(LM_T_STATISTIC in feature_column_list)
        self.assertTrue(LM_P_VALUE in feature_column_list)
        self.assertTrue(LM_LS_CONF_INT in feature_column_list)
        self.assertTrue(LM_RS_CONF_INT in feature_column_list)
        self.assertTrue(LM_VIF in feature_column_list)

    # negative test method for generate_model_csv_files()
    def test_generate_model_csv_files_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                               current_features=current_features,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)

        # create CSV_Loader
        csv_loader = CSV_Loader()

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            the_logistic_model_result.generate_model_csv_files(csv_loader=None)

        # validate the error message.
        self.assertEqual("csv_loader is None or incorrect type.", context.exception.msg)

    # test method for generate_model_csv_files()
    def test_generate_model_csv_files(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)

        # create a logistic model
        logistic_model = Logistic_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = logistic_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_logistic_model_result = logistic_model.fit_a_model(the_target_column=the_target_column,
                                                               current_features=current_features,
                                                               model_type=MT_LOGISTIC_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_logistic_model_result)
        self.assertIsInstance(the_logistic_model_result, Logistic_Model_Result)

        # check if the previous file is there, and if so, delete it.
        if exists("../../../../resources/Output/churn_cleaned.csv"):
            os.remove("../../../../resources/Output/churn_cleaned.csv")

        if exists("../../../../resources/Output/churn_prepared.csv"):
            os.remove("../../../../resources/Output/churn_prepared.csv")

        # invoke the method
        the_logistic_model_result.generate_model_csv_files(csv_loader=pa.csv_l)

        # run assertions
        self.assertTrue(exists("../../../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_prepared.csv"))
