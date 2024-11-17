import os
import unittest
from os.path import exists

import pandas as pd
import statsmodels.api as sm

from pandas import DataFrame, Series
from statsmodels.regression.linear_model import RegressionResultsWrapper
from model.Project_Assessment import Project_Assessment
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.Linear_Model_Result import Linear_Model_Result
from model.constants.BasicConstants import D_209_CHURN, ANALYZE_DATASET_FULL, MT_LINEAR_REGRESSION
from model.constants.ModelConstants import LM_PREDICTOR, LM_P_VALUE, LM_COEFFICIENT, LM_FEATURE_NUM, \
    LM_STANDARD_ERROR, LM_T_STATISTIC, LM_VIF, LM_LS_CONF_INT, LM_RS_CONF_INT, LM_LAGRANGE_MULTIPLIER_STATISTIC, \
    LM_F_VALUE, LM_F_P_VALUE, LM_JARQUE_BERA_STATISTIC, LM_JARQUE_BERA_PROB, LM_JB_SKEW, LM_JS_KURTOSIS
from model.constants.ReportConstants import R_SQUARED_HEADER, ADJ_R_SQUARED_HEADER, RESIDUAL_STD_ERROR, \
    BREUSCH_PAGAN_P_VALUE, JARQUE_BERA_STATISTIC, F_STATISTIC_HEADER, P_VALUE_F_STATISTIC_HEADER, \
    DURBAN_WATSON_STATISTIC, NUMBER_OF_OBS, DEGREES_OF_FREEDOM_MODEL, DEGREES_OF_FREEDOM_RESID, MODEL_CONSTANT, \
    AIC_SCORE, BIC_SCORE
from util.CSV_loader import CSV_Loader


class test_Linear_Model_Result(unittest.TestCase):
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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # verify we handle None, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Linear_Model_Result(the_regression_wrapper=None, the_target_variable=None,
                                the_variables_list=None, the_df=None)

            # validate the error message.
            self.assertTrue("the_regression_wrapper is None or incorrect type." in context.exception)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # get the dataframe
        the_df = linear_model.encoded_df[the_variable_columns].astype(float)

        # verify we handle model, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Linear_Model_Result(the_regression_wrapper=model, the_target_variable=None,
                                the_variables_list=None, the_df=None)

            # validate the error message.
            self.assertTrue("the_target_variable is None or incorrect type." in context.exception)

        # verify we handle model, 'Churn', None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Linear_Model_Result(the_regression_wrapper=model, the_target_variable='Churn',
                                the_variables_list=None, the_df=None)

            # validate the error message.
            self.assertTrue("the_regression_wrapper is None or incorrect type." in context.exception)

        # verify we handle model, 'Churn', the_variable_columns, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Linear_Model_Result(the_regression_wrapper=model, the_target_variable='Churn',
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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)
        self.assertIsNotNone(the_result.get_model())
        self.assertIsInstance(the_result.get_model(), RegressionResultsWrapper)
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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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

        self.assertEqual(the_lm_dict['Population'], 0.53997)
        self.assertEqual(the_lm_dict['TimeZone'], 0.85799)
        self.assertEqual(the_lm_dict['Children'], 0.36451)
        self.assertEqual(the_lm_dict['Age'], 0.49157)
        self.assertEqual(the_lm_dict['Income'], 0.50276)
        self.assertEqual(the_lm_dict['Outage_sec_perweek'], 0.55094)
        self.assertEqual(the_lm_dict['Email'], 0.71237)
        self.assertEqual(the_lm_dict['Contacts'], 0.19063)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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
        self.assertEqual(len(the_lm_dict), 30)

        # validate the returned results
        self.assertEqual(the_lm_dict['Children'], 0.36451)
        self.assertEqual(the_lm_dict['Age'], 0.49157)
        self.assertEqual(the_lm_dict['Contacts'], 0.19063)
        self.assertEqual(the_lm_dict['Yearly_equip_failure'], 0.2543)
        self.assertEqual(the_lm_dict['Techie'], 0.0)
        self.assertEqual(the_lm_dict['Port_modem'], 0.10361)
        self.assertEqual(the_lm_dict['Tablet'], 0.37769)
        self.assertEqual(the_lm_dict['Phone'], 0.03672)
        self.assertEqual(the_lm_dict['Multiple'], 0.00513)
        self.assertEqual(the_lm_dict['OnlineBackup'], 0.00333)
        self.assertEqual(the_lm_dict['TechSupport'], 0.00039)
        self.assertEqual(the_lm_dict['StreamingTV'], 0.26062)
        self.assertEqual(the_lm_dict['StreamingMovies'], 0.34724)
        self.assertEqual(the_lm_dict['PaperlessBilling'], 0.07855)
        self.assertEqual(the_lm_dict['MonthlyCharge'], 0.0)
        self.assertEqual(the_lm_dict['Bandwidth_GB_Year'], 0.398)
        self.assertEqual(the_lm_dict['Timely_Fixes'], 0.19483)
        self.assertEqual(the_lm_dict['Reliability'], 0.26024)
        self.assertEqual(the_lm_dict['Area_Suburban'], 0.45066)
        self.assertEqual(the_lm_dict['Marital_Separated'], 0.02547)
        self.assertEqual(the_lm_dict['Marital_Widowed'], 0.01404)
        self.assertEqual(the_lm_dict['Gender_Male'], 0.15797)
        self.assertEqual(the_lm_dict['Gender_Nonbinary'], 0.39685)
        self.assertEqual(the_lm_dict['Contract_One year'], 0.0)
        self.assertEqual(the_lm_dict['Contract_Two Year'], 0.0)
        self.assertEqual(the_lm_dict['InternetService_Fiber Optic'], 0.0541)
        self.assertEqual(the_lm_dict['InternetService_No response'], 0.27452)
        self.assertEqual(the_lm_dict['PaymentMethod_Credit Card (automatic)'], 0.07368)
        self.assertEqual(the_lm_dict['PaymentMethod_Mailed Check'], 0.0836)

    # test method for get_adjusted_r_squared()
    def test_get_adjusted_r_squared(self):
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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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
        self.assertEqual(the_result.get_adjusted_r_squared(), 0.48902)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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
        self.assertEqual(the_tuple[0], ('AIC', 5354.266763840464))
        self.assertEqual(the_tuple[1], ('BIC', 5700.36310169532))

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 2)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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

        # invoke get_adjusted_r_squared() to make sure we haven't screwed something up
        self.assertEqual(the_result.get_adjusted_r_squared(), 0.37344)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

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

        the_target_series = linear_model.encoded_df['Churn']
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_target_series)
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_target_series, Series)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 2)

        # cast data to floats
        X = linear_model.encoded_df[the_variable_columns].astype(float)

        # get the target series
        y = the_target_series

        # invoke add_constant
        X2 = sm.add_constant(X)

        # create a model
        model = sm.OLS(y, X2).fit()

        # create Linear_Model_Result
        the_result = Linear_Model_Result(the_regression_wrapper=model,
                                         the_target_variable='Churn',
                                         the_variables_list=the_variable_columns,
                                         the_df=X)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

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

        # invoke get_adjusted_r_squared() to make sure we haven't screwed something up
        self.assertEqual(the_result.get_adjusted_r_squared(), 0.37344)

        # invoke the method
        vif_df = the_result.get_vif_for_model(the_encoded_df=X)

        # run assertions
        self.assertIsNotNone(vif_df)
        self.assertIsInstance(vif_df, DataFrame)
        self.assertEqual(vif_df.iloc[0]['VIF'], 2.4559544274750076)
        self.assertEqual(vif_df.iloc[1]['VIF'], 2.4559544274750067)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('MonthlyCharge')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('MonthlyCharge' in the_variable_columns)

        # create the_variable_df
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # run assertions on dataframes
        self.assertIsNotNone(the_variable_df)
        self.assertIsInstance(the_variable_df, DataFrame)
        self.assertEqual(len(the_variable_df.columns), 47)

        # fit a model to get a linear_model_result
        the_lm_result = linear_model.fit_a_model(the_target_column="MonthlyCharge",
                                                 current_features=the_variable_columns,
                                                 model_type=MT_LINEAR_REGRESSION)

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
        self.assertEqual(the_vif_df.iloc[0, 1], 1.4677113358501654)
        self.assertEqual(the_vif_df.iloc[3, 0], 'Age')
        self.assertEqual(the_vif_df.iloc[3, 1], 18.89942298851312)
        self.assertEqual(the_vif_df.iloc[22, 0], 'Tenure')
        self.assertEqual(the_vif_df.iloc[22, 1], 5401.4237061220465)
        self.assertEqual(the_vif_df.iloc[42, 0], 'InternetService_Fiber Optic')
        self.assertEqual(the_vif_df.iloc[42, 1], 32.3862252988207)

        # validate feature with the largest VIF
        self.assertEqual(the_vif_df['VIF'].idxmax(), 23)
        self.assertEqual(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'], 'Bandwidth_GB_Year')

        # remove 'Bandwidth_GB_Year' from the_variable_columns
        the_variable_columns.remove(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'])

        # run assertion to make sure the feature is not present
        self.assertFalse('Bandwidth_GB_Year' in the_variable_columns)

        # create the_variable_df
        the_variable_df = linear_model.encoded_df[the_variable_columns]

        # fit a model to get a linear_model_result
        the_lm_result = linear_model.fit_a_model(the_target_column="MonthlyCharge",
                                                 current_features=the_variable_columns,
                                                 model_type=MT_LINEAR_REGRESSION)

        # cast to float
        the_variable_df = the_variable_df.astype(float)

        # get the_vif_df
        the_vif_df = the_lm_result.get_vif_for_model(the_variable_df)

        # run assertions
        self.assertIsNotNone(the_vif_df)
        self.assertIsInstance(the_vif_df, DataFrame)

        self.assertEqual(the_vif_df['VIF'].idxmax(), 1)
        self.assertEqual(the_vif_df.iloc[the_vif_df['VIF'].idxmax()]['features'], 'TimeZone')

    # negative test method for get_results_dataframe()
    def test_get_results_dataframe_negative(self):
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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # set up the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model_result.get_results_dataframe(round_value=None)

            # validate the error message.
            self.assertTrue("round_value is None or incorrect type." in context.exception)

        # verify we handle -5
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model_result.get_results_dataframe(round_value=-5)

            # validate the error message.
            self.assertTrue("round_value cannot be below one." in context.exception)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
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

        # run specific p-value tests.
        self.assertEqual(the_result_df.loc['Population', 'p-value'], 0.53997)
        self.assertEqual(the_result_df.loc['TimeZone', 'p-value'], 0.85799)
        self.assertEqual(the_result_df.loc['Children', 'p-value'], 0.36451)
        self.assertEqual(the_result_df.loc['Age', 'p-value'], 0.49157)
        self.assertEqual(the_result_df.loc['Income', 'p-value'], 0.50276)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'p-value'], 0.55094)
        self.assertEqual(the_result_df.loc['Email', 'p-value'], 0.71237)

        # run specific coefficient tests.
        self.assertEqual(the_result_df.loc['Population', 'coefficient'], -1.345e-07)
        self.assertEqual(the_result_df.loc['TimeZone', 'coefficient'], 5.750265e-04)
        self.assertEqual(the_result_df.loc['Children', 'coefficient'], 0.0101101504)
        self.assertEqual(the_result_df.loc['Age', 'coefficient'], -8.161858e-04)
        self.assertEqual(the_result_df.loc['Income', 'coefficient'], 7.51e-08)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'coefficient'], -6.339140e-04)
        self.assertEqual(the_result_df.loc['Email', 'coefficient'], 3.853341e-04)

        # run standard error tests
        self.assertEqual(the_result_df.loc['Population', 'std err'], 2.194000e-07)
        self.assertEqual(the_result_df.loc['TimeZone', 'std err'], 0.0032135191)
        self.assertEqual(the_result_df.loc['Children', 'std err'], 0.0111487244)
        self.assertEqual(the_result_df.loc['Age', 'std err'], 0.0011865876)
        self.assertEqual(the_result_df.loc['Income', 'std err'], 1.121000e-07)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'std err'], 0.0010629577)
        self.assertEqual(the_result_df.loc['Email', 'std err'], 1.045149e-03)

        # run t-statistic tests
        self.assertEqual(the_result_df.loc['Population', 't-statistic'], -0.6128870397)
        self.assertEqual(the_result_df.loc['TimeZone', 't-statistic'], 0.17893982340)
        self.assertEqual(the_result_df.loc['Children', 't-statistic'], 0.9068436931)
        self.assertEqual(the_result_df.loc['Age', 't-statistic'], -0.6878428561)
        self.assertEqual(the_result_df.loc['Income', 't-statistic'], 0.6701778315)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 't-statistic'], -0.5963680469)
        self.assertEqual(the_result_df.loc['Email', 't-statistic'], 0.3686882522)

        # run '[0.025' tests
        self.assertEqual(the_result_df.loc['Population', '[0.025'], -5.646000e-07)
        self.assertEqual(the_result_df.loc['TimeZone', '[0.025'], -0.0057241213)
        self.assertEqual(the_result_df.loc['Children', '[0.025'], -0.0117436057)
        self.assertEqual(the_result_df.loc['Age', '[0.025'], -0.0031421377)
        self.assertEqual(the_result_df.loc['Income', '[0.025'], -1.446000e-07)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', '[0.025'], -0.0027175261)
        self.assertEqual(the_result_df.loc['Email', '[0.025'], -0.0016633693)

        # run '0.975]' tests
        self.assertEqual(the_result_df.loc['Population', '0.975]'], 2.956e-07)
        self.assertEqual(the_result_df.loc['TimeZone', '0.975]'], 0.0068741744)
        self.assertEqual(the_result_df.loc['Children', '0.975]'], 0.0319639066)
        self.assertEqual(the_result_df.loc['Age', '0.975]'], 0.0015097661)
        self.assertEqual(the_result_df.loc['Income', '0.975]'], 2.949000e-07)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', '0.975]'], 0.0014496982)
        self.assertEqual(the_result_df.loc['Email', '0.975]'], 0.0024340376)

        # run specific VIF tests
        self.assertEqual(the_result_df.loc['Population', 'VIF'], 1.0073863747)
        self.assertEqual(the_result_df.loc['TimeZone', 'VIF'], 1.0096719549)
        self.assertEqual(the_result_df.loc['Children', 'VIF'], 57.5669472263)
        self.assertEqual(the_result_df.loc['Age', 'VIF'], 60.5997806827)
        self.assertEqual(the_result_df.loc['Income', 'VIF'], 1.0042853293)
        self.assertEqual(the_result_df.loc['Outage_sec_perweek', 'VIF'], 1.0052661428)
        self.assertEqual(the_result_df.loc['Email', 'VIF'], 1.0047144653)
        self.assertEqual(the_result_df.loc['Contacts', 'VIF'], 1.004121656)
        self.assertEqual(the_result_df.loc['Yearly_equip_failure', 'VIF'], 1.0044778115)
        self.assertEqual(the_result_df.loc['Techie', 'VIF'], 1.0039071535)
        self.assertEqual(the_result_df.loc['Port_modem', 'VIF'], 1.0021968744)
        self.assertEqual(the_result_df.loc['Tablet', 'VIF'], 1.0061991776)
        self.assertEqual(the_result_df.loc['Phone', 'VIF'], 1.0057585782)
        self.assertEqual(the_result_df.loc['Multiple', 'VIF'], 5.988826259)
        self.assertEqual(the_result_df.loc['OnlineSecurity', 'VIF'], 14.8377303543)
        self.assertEqual(the_result_df.loc['OnlineBackup', 'VIF'], 5.3570559371)
        self.assertEqual(the_result_df.loc['DeviceProtection', 'VIF'], 9.1422436449)
        self.assertEqual(the_result_df.loc['TechSupport', 'VIF'], 4.5058082235)
        self.assertEqual(the_result_df.loc['StreamingTV', 'VIF'], 43.4372462796)
        self.assertEqual(the_result_df.loc['StreamingMovies', 'VIF'], 21.2340408596)
        self.assertEqual(the_result_df.loc['PaperlessBilling', 'VIF'], 1.005075186)
        self.assertEqual(the_result_df.loc['Tenure', 'VIF'], 60828.9441972047)
        self.assertEqual(the_result_df.loc['MonthlyCharge', 'VIF'], 222.7078554304)
        self.assertEqual(the_result_df.loc['Bandwidth_GB_Year', 'VIF'], 61882.5725111591)
        self.assertEqual(the_result_df.loc['Timely_Response', 'VIF'], 2.220864434)
        self.assertEqual(the_result_df.loc['Timely_Fixes', 'VIF'], 1.9364656098)
        self.assertEqual(the_result_df.loc['Timely_Replacements', 'VIF'], 1.6087554722)
        self.assertEqual(the_result_df.loc['Reliability', 'VIF'], 1.2798988474)
        self.assertEqual(the_result_df.loc['Options', 'VIF'], 1.3766553563)
        self.assertEqual(the_result_df.loc['Respectful_Response', 'VIF'], 1.4848638897)
        self.assertEqual(the_result_df.loc['Courteous_Exchange', 'VIF'], 1.3154334611)
        self.assertEqual(the_result_df.loc['Active_Listening', 'VIF'], 1.1920006166)
        self.assertEqual(the_result_df.loc['Area_Suburban', 'VIF'], 1.3381317238)
        self.assertEqual(the_result_df.loc['Area_Urban', 'VIF'], 1.3396246618)
        self.assertEqual(the_result_df.loc['Marital_Married', 'VIF'], 1.5538116175)
        self.assertEqual(the_result_df.loc['Marital_Never Married', 'VIF'], 1.5629813003)
        self.assertEqual(the_result_df.loc['Marital_Separated', 'VIF'], 1.5718247458)
        self.assertEqual(the_result_df.loc['Marital_Widowed', 'VIF'], 1.5790515482)
        self.assertEqual(the_result_df.loc['Gender_Male', 'VIF'], 14.6467111046)
        self.assertEqual(the_result_df.loc['Gender_Nonbinary', 'VIF'], 1.1600109871)
        self.assertEqual(the_result_df.loc['Contract_One year', 'VIF'], 1.098604349)
        self.assertEqual(the_result_df.loc['Contract_Two Year', 'VIF'], 1.0987847932)
        self.assertEqual(the_result_df.loc['InternetService_Fiber Optic', 'VIF'], 712.5327208002)
        self.assertEqual(the_result_df.loc['InternetService_No response', 'VIF'], 310.1979479167)
        self.assertEqual(the_result_df.loc['PaymentMethod_Credit Card (automatic)', 'VIF'], 1.5384082782)
        self.assertEqual(the_result_df.loc['PaymentMethod_Electronic Check', 'VIF'], 1.6728698678)
        self.assertEqual(the_result_df.loc['PaymentMethod_Mailed Check', 'VIF'], 1.5714552546)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # get the constant
        self.assertEqual(the_linear_model_result.get_constant(), 0.07474698416202025)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model_result.are_p_values_above_threshold(p_value=None)

            # validate the error message.
            self.assertTrue("p_value was None or incorrect type." in context.exception)

        # verify we handle 2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model_result.are_p_values_above_threshold(p_value=2.0)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
        self.assertIsInstance(the_linear_model_result.get_the_p_values(), dict)
        self.assertEqual(len(the_linear_model_result.get_the_p_values()), 47)

        # invoke the method with default p_value argument of 1.0.  All the p-values should be below
        # the threshold.
        self.assertFalse(the_linear_model_result.are_p_values_above_threshold())

        # invoke the method with default p_value argument of 0.50.
        self.assertTrue(the_linear_model_result.are_p_values_above_threshold(p_value=0.50))

        # pick results that are < 0.50
        the_results_df = the_linear_model_result.get_results_dataframe()

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('Churn')

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
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model_result.identify_parameter_based_on_p_value(p_value=None)

            # validate the error message.
            self.assertTrue("p_value was None or incorrect type." in context.exception)

        # verify we handle 2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model_result.identify_parameter_based_on_p_value(p_value=2.0)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('Churn')

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
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # get the next feature to remove
        the_feature = the_linear_model_result.identify_parameter_based_on_p_value(p_value=0.05)

        # run assertions
        self.assertIsNotNone(the_feature)
        self.assertIsInstance(the_feature, str)
        self.assertTrue(the_feature in the_variable_columns)
        self.assertEqual(the_feature, 'Courteous_Exchange')

    # proof of concept for retrieving the params from the linear model result
    def test_params_proof_of_concept(self):
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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('Churn')

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
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='Churn',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # get the params
        print(the_linear_model_result.model.params)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model_result.is_vif_above_threshold(max_allowable_vif=None)

            # validate the error message.
            self.assertTrue("max_allowable_vif was None or incorrect type." in context.exception)

        # verify we handle "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model_result.is_vif_above_threshold(max_allowable_vif="foo")

            # validate the error message.
            self.assertTrue("max_allowable_vif was None or incorrect type." in context.exception)

        # verify we handle -1.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model_result.is_vif_above_threshold(max_allowable_vif=-1.0)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)

        # get the current list of variables from the the_linear_model_result
        current_variable_columns = the_linear_model_result.get_the_variables_list()

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in current_variable_columns)

        for the_column in current_variable_columns:
            self.assertTrue(the_column in the_variable_columns)

        # invoke the method
        self.assertTrue(the_linear_model_result.is_vif_above_threshold(max_allowable_vif=10.0))

        # get the dataframe
        the_results_df = the_linear_model_result.get_results_dataframe()

        # make sure the columns in the_results_df match current_variable_columns
        for the_column in the_results_df.index.values.tolist():
            self.assertTrue(the_column in current_variable_columns)

        for the_column in current_variable_columns:
            self.assertTrue(the_column in the_results_df.index.values.tolist())

        # run verifications that the base VIF results for the first model are correct.
        # these are features with a large VIF
        self.assertEqual(the_results_df.loc['Bandwidth_GB_Year']['VIF'], 6743.9931365377)
        self.assertEqual(the_results_df.loc['Tenure']['VIF'], 6643.076704641)
        self.assertEqual(the_results_df.loc['StreamingTV']['VIF'], 19.1353559934)
        self.assertEqual(the_results_df.loc['StreamingMovies']['VIF'], 16.3693467654)
        self.assertEqual(the_results_df.loc['InternetService_Fiber Optic']['VIF'], 60.7594002055)
        self.assertEqual(the_results_df.loc['InternetService_No response']['VIF'], 41.6822523449)

        # ****************************************************************
        # remove Bandwidth_GB_Year from the_variable_columns
        the_variable_columns.remove('Bandwidth_GB_Year')

        # run an assertion to prove the remove worked correclty
        self.assertTrue('Bandwidth_GB_Year' not in the_variable_columns)

        # get a Linear_Model_Result
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # invoke the method again
        self.assertFalse(the_linear_model_result.is_vif_above_threshold(max_allowable_vif=10.0))

        # get the the_results_df from the current the_linear_model_result instance
        the_results_df = the_linear_model_result.get_results_dataframe()

        self.assertEqual(the_results_df.loc['Tenure']['VIF'], 1.4536766374)
        self.assertEqual(the_results_df.loc['StreamingTV']['VIF'], 1.1143247147)
        self.assertEqual(the_results_df.loc['StreamingMovies']['VIF'], 1.1626128333)
        self.assertEqual(the_results_df.loc['InternetService_Fiber Optic']['VIF'], 1.2970363178)
        self.assertEqual(the_results_df.loc['InternetService_No response']['VIF'], 1.2904423863)

        # Thus, this test proves that removing a single variable with the highest VIF can clean up
        # all the multicollinearity issues

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_linear_model_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                               current_features=the_variable_columns,
                                                               model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)

        # get the current list of variables from the the_linear_model_result
        current_variable_columns = the_linear_model_result.get_the_variables_list()

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in current_variable_columns)

        for the_column in current_variable_columns:
            self.assertTrue(the_column in the_variable_columns)

        # check that we actually have a feature above max_allowable_vif
        self.assertTrue(the_linear_model_result.is_vif_above_threshold(max_allowable_vif=10.0))

        # invoke the method
        the_feature = the_linear_model_result.get_feature_with_max_vif()

        # run assertions
        self.assertEqual(the_feature, "Bandwidth_GB_Year")

    # test method for get_durbin_watson_statistic_for_residuals()
    def test_get_durbin_watson_statistic_for_residuals(self):
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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # invoke method reduce_a_model() with max p-value
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.80)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertTrue(results_df['p-value'].max() < 0.80)
        self.assertEqual(len(results_df), 39)

        # get a model with max_p_value
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.001,
                                                     max_vif=10.0)

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

        # invoke the method
        db_statistic = the_result.get_durbin_watson_statistic_for_residuals()

        # run assertions
        self.assertIsNotNone(db_statistic)
        self.assertIsInstance(db_statistic, float)
        self.assertEqual(db_statistic, 2.0032527062)

    # test method for get_residual_standard_error() method
    def test_get_residual_standard_error(self):
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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.001,
                                                     max_vif=10.0)

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

        # invoke the method
        the_rse = the_result.get_residual_standard_error()

        # run assertions
        self.assertIsNotNone(the_rse)
        self.assertIsInstance(the_rse, float)
        self.assertEqual(the_rse, 8.7211125709)

    # test method for get_breusch_pagan_statistic()
    def test_get_breusch_pagan_statistic(self):
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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.001,
                                                     max_vif=10.0)

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

        # invoke the method
        the_bps = the_result.get_assumption_result(BREUSCH_PAGAN_P_VALUE)

        # run assertions
        self.assertIsNotNone(the_bps)
        self.assertIsInstance(the_bps, dict)
        self.assertEqual(the_bps[LM_LAGRANGE_MULTIPLIER_STATISTIC], 477.5649515006)
        self.assertEqual(the_bps[LM_P_VALUE], 0)
        self.assertEqual(the_bps[LM_F_VALUE], 38.5241119960)
        self.assertEqual(the_bps[LM_F_P_VALUE], 0)

    # test method for the jarque-bera
    def test_get_jarque_bera_statistic(self):
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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.001,
                                                     max_vif=10.0)

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

        # invoke the method
        the_jbs = the_result.get_jarque_bera_statistic()

        # run assertions
        self.assertIsNotNone(the_jbs)
        self.assertIsInstance(the_jbs, dict)
        self.assertEqual(the_jbs[LM_JARQUE_BERA_STATISTIC], 1546.2362720939)
        self.assertEqual(the_jbs[LM_JARQUE_BERA_PROB], 0)
        self.assertEqual(the_jbs[LM_JB_SKEW], 0.0207223224)
        self.assertEqual(the_jbs[LM_JS_KURTOSIS], 1.0740588289)

        # (1546.2362720938597, 0.0, 0.02072232242803714, 1.0740588289259672)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a model with max_p_value
        the_result = the_linear_model.fit_mlr_model(the_target_column='MonthlyCharge',
                                                    the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # invoke the method
        the_assumptions = the_result.get_assumptions()

        # run assertions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 14)

        # check what they are
        self.assertTrue(R_SQUARED_HEADER in the_assumptions)
        self.assertTrue(ADJ_R_SQUARED_HEADER in the_assumptions)
        self.assertTrue(DURBAN_WATSON_STATISTIC in the_assumptions)
        self.assertTrue(RESIDUAL_STD_ERROR in the_assumptions)
        self.assertTrue(BREUSCH_PAGAN_P_VALUE in the_assumptions)
        self.assertTrue(JARQUE_BERA_STATISTIC in the_assumptions)
        self.assertTrue(F_STATISTIC_HEADER in the_assumptions)
        self.assertTrue(P_VALUE_F_STATISTIC_HEADER in the_assumptions)
        self.assertTrue(NUMBER_OF_OBS in the_assumptions)
        self.assertTrue(DEGREES_OF_FREEDOM_MODEL in the_assumptions)
        self.assertTrue(DEGREES_OF_FREEDOM_RESID in the_assumptions)
        self.assertTrue(MODEL_CONSTANT in the_assumptions)
        self.assertTrue(AIC_SCORE in the_assumptions)
        self.assertTrue(BIC_SCORE in the_assumptions)

        # check the value
        self.assertEqual(the_assumptions[R_SQUARED_HEADER], 'get_r_squared')
        self.assertEqual(the_assumptions[ADJ_R_SQUARED_HEADER], 'get_adjusted_r_squared')
        self.assertEqual(the_assumptions[DURBAN_WATSON_STATISTIC], 'get_durbin_watson_statistic_for_residuals')
        self.assertEqual(the_assumptions[RESIDUAL_STD_ERROR], 'get_residual_standard_error')
        self.assertEqual(the_assumptions[BREUSCH_PAGAN_P_VALUE], 'get_breusch_pagan_statistic')
        self.assertEqual(the_assumptions[JARQUE_BERA_STATISTIC], 'get_jarque_bera_statistic')
        self.assertEqual(the_assumptions[F_STATISTIC_HEADER], 'get_f_value')
        self.assertEqual(the_assumptions[P_VALUE_F_STATISTIC_HEADER], 'get_f_p_value')
        self.assertEqual(the_assumptions[NUMBER_OF_OBS], 'get_number_of_obs')
        self.assertEqual(the_assumptions[DEGREES_OF_FREEDOM_MODEL], 'get_degrees_of_freedom_for_model')
        self.assertEqual(the_assumptions[DEGREES_OF_FREEDOM_RESID], 'get_degrees_of_freedom_for_residuals')
        self.assertEqual(the_assumptions[MODEL_CONSTANT], 'get_constant')
        self.assertEqual(the_assumptions[AIC_SCORE], 'get_aic_for_model')
        self.assertEqual(the_assumptions[BIC_SCORE], 'get_bic_for_model')

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_linear_model.fit_mlr_model(the_target_column='MonthlyCharge',
                                                       the_variable_columns=method_results)

        # run basic assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # call get_assumptions()
        the_assumptions = the_lm_result.get_assumptions()

        # run assertions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 14)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                     current_features=method_results,
                                                     model_type=MT_LINEAR_REGRESSION)

        # run basic assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # call get_assumptions()
        the_assumptions = the_lm_result.get_assumptions()

        # run assertions
        self.assertIsNotNone(the_assumptions)
        self.assertIsInstance(the_assumptions, dict)
        self.assertEqual(len(the_assumptions), 14)

        # run assertions
        self.assertEqual(the_lm_result.get_assumption_result(R_SQUARED_HEADER), 0.99552)
        self.assertEqual(the_lm_result.get_assumption_result(ADJ_R_SQUARED_HEADER), 0.9955)
        self.assertEqual(the_lm_result.get_assumption_result(DURBAN_WATSON_STATISTIC), 2.0097682611)
        self.assertEqual(the_lm_result.get_assumption_result(RESIDUAL_STD_ERROR), 2.8805141553)

        # validate type on BRUESCH PAGAN result
        self.assertIsInstance(the_lm_result.get_assumption_result(BREUSCH_PAGAN_P_VALUE), dict)
        self.assertEqual(
            the_lm_result.get_assumption_result(BREUSCH_PAGAN_P_VALUE)['Lagrange Multiplier Statistic'], 79.8421927769)
        self.assertEqual(
            the_lm_result.get_assumption_result(BREUSCH_PAGAN_P_VALUE)['f p-value'], 0.0019191041)
        self.assertEqual(
            the_lm_result.get_assumption_result(BREUSCH_PAGAN_P_VALUE)['f-value'], 1.7042228517)
        self.assertEqual(
            the_lm_result.get_assumption_result(BREUSCH_PAGAN_P_VALUE)['p-value'], 0.0019712966)

        # validate type on JARQUE BERA result
        self.assertIsInstance(the_lm_result.get_assumption_result(JARQUE_BERA_STATISTIC), dict)
        self.assertEqual(
            the_lm_result.get_assumption_result(JARQUE_BERA_STATISTIC)['Chi^2 two-tail prob'], 0.0)
        self.assertEqual(
            the_lm_result.get_assumption_result(JARQUE_BERA_STATISTIC)['Jarque-Bera Statistic'], 1078.4279865025)
        self.assertEqual(
            the_lm_result.get_assumption_result(JARQUE_BERA_STATISTIC)['Kurtosis'], 1.3921743659)
        self.assertEqual(
            the_lm_result.get_assumption_result(JARQUE_BERA_STATISTIC)['Skew'], -0.027945921)

        self.assertEqual(the_lm_result.get_assumption_result(F_STATISTIC_HEADER), 47066.54003093157)
        self.assertEqual(the_lm_result.get_assumption_result(P_VALUE_F_STATISTIC_HEADER), 0.0)
        self.assertEqual(the_lm_result.get_assumption_result(NUMBER_OF_OBS), 10000)
        self.assertEqual(the_lm_result.get_assumption_result(DEGREES_OF_FREEDOM_MODEL), 47)
        self.assertEqual(the_lm_result.get_assumption_result(DEGREES_OF_FREEDOM_RESID), 9952)
        self.assertEqual(the_lm_result.get_assumption_result(MODEL_CONSTANT), -88.30564306635014)
        self.assertEqual(the_lm_result.get_assumption_result(AIC_SCORE), 49587.03595393858)
        self.assertEqual(the_lm_result.get_assumption_result(BIC_SCORE), 49933.13229179344)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                     current_features=method_results,
                                                     model_type=MT_LINEAR_REGRESSION)

        # run basic assertions
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

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

        # create a linear model
        the_linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions on the_linear_model to make sure the encoded_df property is setup
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('MonthlyCharge' in the_variable_columns)

        # remove MonthlyCharge
        the_variable_columns.remove('MonthlyCharge')

        # call get_encoded_variables()
        method_results = the_linear_model.get_encoded_variables('MonthlyCharge')

        # run assertions
        self.assertIsNotNone(method_results)
        self.assertIsInstance(method_results, list)
        self.assertEqual(len(method_results), 47)
        self.assertFalse('MonthlyCharge' in method_results)

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # get a Linear_Model_Result
        the_lm_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                     current_features=method_results,
                                                     model_type=MT_LINEAR_REGRESSION)

        # run basic assertions on the_lm_result
        self.assertIsNotNone(the_lm_result)
        self.assertIsInstance(the_lm_result, Linear_Model_Result)

        # final assertions
        self.assertTrue(the_lm_result.has_assumption(R_SQUARED_HEADER))
        self.assertTrue(the_lm_result.has_assumption(ADJ_R_SQUARED_HEADER))
        self.assertTrue(the_lm_result.has_assumption(DURBAN_WATSON_STATISTIC))
        self.assertTrue(the_lm_result.has_assumption(RESIDUAL_STD_ERROR))
        self.assertTrue(the_lm_result.has_assumption(BREUSCH_PAGAN_P_VALUE))
        self.assertTrue(the_lm_result.has_assumption(JARQUE_BERA_STATISTIC))
        self.assertTrue(the_lm_result.has_assumption(F_STATISTIC_HEADER))
        self.assertTrue(the_lm_result.has_assumption(P_VALUE_F_STATISTIC_HEADER))

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)
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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)

        # create CSV_Loader
        csv_loader = CSV_Loader()

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            the_linear_model_result.generate_model_csv_files(csv_loader=None)

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

        # create a linear model
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # get the list of columns
        the_variable_columns = linear_model.get_encoded_variables(the_target_column='Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 47)
        self.assertFalse('Churn' in the_variable_columns)

        # setup the variables we will pass to the method fit_a_model()
        the_target_column = 'Churn'
        current_features = the_variable_columns

        # get back the_linear_model_result
        the_linear_model_result = linear_model.fit_a_model(the_target_column=the_target_column,
                                                           current_features=current_features,
                                                           model_type=MT_LINEAR_REGRESSION)

        # run assertions
        self.assertIsNotNone(the_linear_model_result)
        self.assertIsInstance(the_linear_model_result, Linear_Model_Result)

        # check if the previous file is there, and if so, delete it.
        if exists("../../../../resources/Output/churn_cleaned.csv"):
            os.remove("../../../../resources/Output/churn_cleaned.csv")

        if exists("../../../../resources/Output/churn_prepared.csv"):
            os.remove("../../../../resources/Output/churn_prepared.csv")

        # invoke the method
        the_linear_model_result.generate_model_csv_files(csv_loader=pa.csv_l)

        # run assertions
        self.assertTrue(exists("../../../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_prepared.csv"))
