import unittest

from pandas import DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.models.Linear_Model import Linear_Model, order_features_by_r_squared
from model.analysis.models.Linear_Model_Result import Linear_Model_Result
from model.analysis.Variable_Encoder import Variable_Encoder
from model.constants.BasicConstants import D_212_CHURN, ANALYZE_DATASET_FULL, MT_LINEAR_REGRESSION
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_STEP, LM_FINAL_MODEL


class test_Linear_Model(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"
    OVERRIDE_PATH = "../../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    VALID_WB_PATH = "../../../resources/Output/model_dump_test.xlsx"
    CHURN_KEY = D_212_CHURN

    # negative tests for the init() method
    def test_init_negative(self):
        # verify we handle None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            Linear_Model(dataset_analyzer=None)

            # validate the error message.
            self.assertTrue("dataset_analyzer is None or incorrect type." in context.exception)

    # test method for init()
    def test_init(self):
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

        # invoke the method
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # run assertions
        self.assertIsNotNone(linear_model)
        self.assertIsInstance(linear_model, Linear_Model)
        self.assertIsNotNone(linear_model.dataset_analyzer)
        self.assertIsInstance(linear_model.dataset_analyzer, DatasetAnalyzer)
        self.assertIsNotNone(linear_model.model_storage)
        self.assertIsInstance(linear_model.model_storage, dict)
        self.assertIsNotNone(linear_model.variable_encoder)
        self.assertIsInstance(linear_model.variable_encoder, Variable_Encoder)
        self.assertIsNotNone(linear_model.encoded_df)
        self.assertIsInstance(linear_model.encoded_df, DataFrame)

        self.assertEqual(len(linear_model.encoded_df.columns), 48)

        # get the list of columns from the_linear_model
        the_variable_columns = linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # call get_encoded_variables()
        method_results = linear_model.get_encoded_variables('Churn')

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

    # negative test method for build_model_for_single_column()
    def test_build_model_for_single_column_negative(self):
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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # verify we handle None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            linear_model.build_model_for_single_column(the_target_column=None, the_variable_column=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            linear_model.build_model_for_single_column(the_target_column="foo", the_variable_column=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle 'Bandwidth_GB_Year', None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            linear_model.build_model_for_single_column(the_target_column='Bandwidth_GB_Year', the_variable_column=None)

            # validate the error message.
            self.assertTrue("the_variable_column argument is None or incorrect type." in context.exception)

        # verify we handle 'Bandwidth_GB_Year', "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            linear_model.build_model_for_single_column(the_target_column='Bandwidth_GB_Year',
                                                       the_variable_column="foo")

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

    # test method for build_model_for_single_column()
    def test_build_model_for_single_column(self):
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
        linear_model = Linear_Model(dataset_analyzer=pa.analyzer)

        # pick the target as 'Bandwidth_GB_Year' and 'Tenure' as random variable
        the_result = linear_model.build_model_for_single_column(the_target_column='Bandwidth_GB_Year',
                                                                the_variable_column='Tenure')

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)
        self.assertIsInstance(the_result.get_the_p_values(), dict)
        self.assertEqual(the_result.get_the_p_values(), {'Tenure': 0.0})

    # negative tests for fit_mlr_model()
    def test_fit_mlr_model_negative(self):
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

        # verify we handle None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.fit_mlr_model(the_target_column=None, the_variable_columns=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo", None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.fit_mlr_model(the_target_column="foo", the_variable_columns=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle 'Bandwidth_GB_Year', None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.fit_mlr_model(the_target_column='Bandwidth_GB_Year', the_variable_columns=None)

            # validate the error message.
            self.assertTrue("the_variable_columns argument is None or incorrect type." in context.exception)

        # verify we handle 'Bandwidth_GB_Year', "foo"
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.fit_mlr_model(the_target_column='Bandwidth_GB_Year',
                                           the_variable_columns="foo")

            # validate the error message.
            self.assertTrue("the_variable_columns argument is None or incorrect type." in context.exception)

        # verify we handle 'Bandwidth_GB_Year', ['foo', 'Population']
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.fit_mlr_model(the_target_column='Bandwidth_GB_Year',
                                           the_variable_columns=['foo', 'Population'])

            # validate the error message.
            self.assertTrue("an element in the_variable_columns argument is not in dataframe." in context.exception)

    # test method for fit_mlr_model()
    def test_fit_mlr_model(self):
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

        # get the list of columns
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

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

        # invoke the method
        the_result = the_linear_model.fit_mlr_model(the_target_column='Churn',
                                                    the_variable_columns=the_variable_columns)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)
        self.assertIsInstance(the_result.get_model(), RegressionResultsWrapper)
        self.assertEqual(the_result.get_the_target_variable(), 'Churn')
        self.assertIsInstance(the_result.get_the_variables_list(), list)

        # get the list itself
        the_linear_model_results_list = the_result.get_the_variables_list()

        # loop over all the columns in the_linear_model_results_list
        for the_column in the_linear_model_results_list:
            self.assertTrue(the_column in the_variable_columns)

        # loop over all the columns in the_variable_columns
        for the_column in the_variable_columns:
            self.assertTrue(the_column in the_linear_model_results_list)

        # get the p-values dict
        p_values_dict = the_result.get_the_p_values()

        # run assertions
        self.assertIsNotNone(p_values_dict)
        self.assertIsInstance(p_values_dict, dict)
        self.assertEqual(len(p_values_dict), 47)

    # test method using p_values reduction
    def test_fit_mlr_model_p_values_test(self):
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

        # get the list of columns
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

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

        # invoke the method
        the_result = the_linear_model.fit_mlr_model(the_target_column='Churn',
                                                    the_variable_columns=the_variable_columns)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)
        self.assertIsInstance(the_result.get_model(), RegressionResultsWrapper)
        self.assertEqual(the_result.get_the_target_variable(), 'Churn')
        self.assertIsInstance(the_result.get_the_variables_list(), list)

        # get the list itself
        the_linear_model_results_list = the_result.get_the_variables_list()

        # loop over all the columns in the_linear_model_results_list
        for the_column in the_linear_model_results_list:
            self.assertTrue(the_column in the_variable_columns)

        # loop over all the columns in the_variable_columns
        for the_column in the_variable_columns:
            self.assertTrue(the_column in the_linear_model_results_list)

        # get the p-values dict
        p_values_dict = the_result.get_the_p_values()

        # run assertions
        self.assertIsNotNone(p_values_dict)
        self.assertIsInstance(p_values_dict, dict)
        self.assertEqual(len(p_values_dict), 47)

        # validate some p-values for 47 variable model
        self.assertEqual(p_values_dict['Population'], 0.53997)
        self.assertEqual(p_values_dict['TimeZone'], 0.85799)
        self.assertEqual(p_values_dict['Children'], 0.36451)
        self.assertEqual(p_values_dict['Age'], 0.49157)
        self.assertEqual(p_values_dict['Income'], 0.50276)
        self.assertEqual(p_values_dict['Outage_sec_perweek'], 0.55094)

        # validate the adjusted r_squared
        self.assertEqual(the_result.get_adjusted_r_squared(), 0.48902)

        # invoke the method
        p_values_dict = the_result.get_the_p_values(less_than=0.50)

        # run final assertions
        self.assertIsNotNone(p_values_dict)
        self.assertIsInstance(p_values_dict, dict)
        self.assertEqual(len(p_values_dict), 30)

        self.assertEqual(p_values_dict['Children'], 0.36451)
        self.assertEqual(p_values_dict['Age'], 0.49157)
        self.assertEqual(p_values_dict['Contacts'], 0.19063)
        self.assertEqual(p_values_dict['Yearly_equip_failure'], 0.2543)
        self.assertEqual(p_values_dict['Techie'], 0.0)
        self.assertEqual(p_values_dict['Port_modem'], 0.10361)
        self.assertEqual(p_values_dict['Tablet'], 0.37769)
        self.assertEqual(p_values_dict['Phone'], 0.03672)
        self.assertEqual(p_values_dict['Multiple'], 0.00513)
        self.assertEqual(p_values_dict['OnlineBackup'], 0.00333)
        self.assertEqual(p_values_dict['TechSupport'], 0.00039)
        self.assertEqual(p_values_dict['StreamingTV'], 0.26062)
        self.assertEqual(p_values_dict['StreamingMovies'], 0.34724)
        self.assertEqual(p_values_dict['PaperlessBilling'], 0.07855)
        self.assertEqual(p_values_dict['MonthlyCharge'], 0.0)
        self.assertEqual(p_values_dict['Bandwidth_GB_Year'], 0.398)
        self.assertEqual(p_values_dict['Timely_Fixes'], 0.19483)
        self.assertEqual(p_values_dict['Reliability'], 0.26024)
        self.assertEqual(p_values_dict['Area_Suburban'], 0.45066)
        self.assertEqual(p_values_dict['Gender_Nonbinary'], 0.39685)
        self.assertEqual(p_values_dict['Contract_One year'], 0.0)
        self.assertEqual(p_values_dict['Contract_Two Year'], 0.0)
        self.assertEqual(p_values_dict['InternetService_Fiber Optic'], 0.0541)
        self.assertEqual(p_values_dict['PaymentMethod_Credit Card (automatic)'], 0.07368)
        self.assertEqual(p_values_dict['PaymentMethod_Mailed Check'], 0.0836)

        # get the adjusted r_squared
        self.assertEqual(the_result.get_adjusted_r_squared(), 0.48902)

    # negative tests for get_encoded_variables() method
    def test_get_encoded_variables_negative(self):
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

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.get_encoded_variables(the_target_column=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo"
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.get_encoded_variables(the_target_column="foo")

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

    # test method for get_encoded_variables()
    def test_get_encoded_variables(self):
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

        # run assertions on the_linear_model
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # invoke method
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

    # negative tests for build_mlr_for_single_comparison()
    def test_build_mlr_for_single_comparison_negative(self):
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

        # run assertions on the_linear_model
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # invoke method
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

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.get_encoded_variables(the_target_column=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo"
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.get_encoded_variables(the_target_column="foo")

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

    # test method for build_mlr_for_single_comparison()
    def test_build_mlr_for_single_comparison(self):
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

        # run assertions on the_linear_model
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns
        the_variable_columns = the_linear_model.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 48)
        self.assertTrue('Churn' in the_variable_columns)

        # remove Churn
        the_variable_columns.remove('Churn')

        # invoke get_encoded_variables()
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

        # invoke the method
        the_model_results_dict = the_linear_model.build_mlr_for_single_comparison('Churn')

        # run assertions
        self.assertIsNotNone(the_model_results_dict)
        self.assertIsInstance(the_model_results_dict, dict)
        self.assertEqual(len(the_model_results_dict), 47)

        # loop over all the elements in the_model_results_dict
        for the_column in list(the_model_results_dict.keys()):
            # make sure the column is in method_results
            self.assertTrue(the_column in the_model_results_dict)

            # make sure the result returned is the correct type
            self.assertIsInstance(the_model_results_dict[the_column], Linear_Model_Result)

    # negative test method for order_features_by_r_squared()
    def test_order_features_by_r_squared_negative(self):
        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            order_features_by_r_squared(None)

            # validate the error message.
            self.assertTrue("the_dict argument is None or incorrect type." in context.exception)

    # test method for order_features_by_r_squared()
    def test_order_features_by_r_squared(self):
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

        # run assertions on the_linear_model
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns
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

        # get the the_model_results_dict
        the_model_results_dict = the_linear_model.build_mlr_for_single_comparison('Churn')

        # run assertions
        self.assertIsNotNone(the_model_results_dict)
        self.assertIsInstance(the_model_results_dict, dict)
        self.assertEqual(len(the_model_results_dict), 47)

        # loop over all the elements in the_model_results_dict
        for the_column in list(the_model_results_dict.keys()):
            # make sure the column is in method_results
            self.assertTrue(the_column in the_model_results_dict)

            # make sure the result returned is the correct type
            self.assertIsInstance(the_model_results_dict[the_column], Linear_Model_Result)

        # invoke the method
        ordered_df = order_features_by_r_squared(the_model_results_dict)

        # run assertions
        self.assertIsNotNone(ordered_df)
        self.assertIsInstance(ordered_df, DataFrame)
        self.assertEqual(len(ordered_df), 47)

        # test the first few values to make sure the ordering is correct
        self.assertEqual(ordered_df.iloc[0]['predictor'], 'Tenure')
        self.assertEqual(ordered_df.iloc[0]['r-squared'], 0.23561)
        self.assertEqual(ordered_df.iloc[1]['predictor'], 'Bandwidth_GB_Year')
        self.assertEqual(ordered_df.iloc[1]['r-squared'], 0.19499)
        self.assertEqual(ordered_df.iloc[2]['predictor'], 'MonthlyCharge')
        self.assertEqual(ordered_df.iloc[2]['r-squared'], 0.13900)
        self.assertEqual(ordered_df.iloc[3]['predictor'], 'StreamingMovies')
        self.assertEqual(ordered_df.iloc[3]['r-squared'], 0.08358)
        self.assertEqual(ordered_df.iloc[4]['predictor'], 'StreamingTV')
        self.assertEqual(ordered_df.iloc[4]['r-squared'], 0.05287)
        self.assertEqual(ordered_df.iloc[5]['predictor'], 'Contract_Two Year')
        self.assertEqual(ordered_df.iloc[5]['r-squared'], 0.03171)
        self.assertEqual(ordered_df.iloc[6]['predictor'], 'Contract_One year')
        self.assertEqual(ordered_df.iloc[6]['r-squared'], 0.01923)
        self.assertEqual(ordered_df.iloc[7]['predictor'], 'Multiple')
        self.assertEqual(ordered_df.iloc[7]['r-squared'], 0.01727)

    # negative test method for select_next_feature()
    def test_select_next_feature_negative(self):
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

        # run assertions on the_linear_model
        self.assertIsNotNone(the_linear_model.encoded_df)
        self.assertIsInstance(the_linear_model.encoded_df, DataFrame)
        self.assertEqual(len(the_linear_model.encoded_df.columns), 48)

        # get the list of columns
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

        # get the the_model_results_dict
        the_model_results_dict = the_linear_model.build_mlr_for_single_comparison('Churn')

        # run assertions
        self.assertIsNotNone(the_model_results_dict)
        self.assertIsInstance(the_model_results_dict, dict)
        self.assertEqual(len(the_model_results_dict), 47)

        # loop over all the elements in the_model_results_dict
        for the_column in list(the_model_results_dict.keys()):
            # make sure the column is in method_results
            self.assertTrue(the_column in the_model_results_dict)

            # make sure the result returned is the correct type
            self.assertIsInstance(the_model_results_dict[the_column], Linear_Model_Result)

        # invoke the method
        ordered_df = order_features_by_r_squared(the_model_results_dict)

        # run assertions
        self.assertIsNotNone(ordered_df)
        self.assertIsInstance(ordered_df, DataFrame)
        self.assertEqual(len(ordered_df), 47)

        # test the first few values to make sure the ordering is correct
        self.assertEqual(ordered_df.iloc[0]['predictor'], 'Tenure')
        self.assertEqual(ordered_df.iloc[0]['r-squared'], 0.23561)
        self.assertEqual(ordered_df.iloc[1]['predictor'], 'Bandwidth_GB_Year')
        self.assertEqual(ordered_df.iloc[1]['r-squared'], 0.19499)

        # verify we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.select_next_feature(the_target_column=None, current_features=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.select_next_feature(the_target_column="foo", current_features=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle "Churn", "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.select_next_feature(the_target_column="Churn", current_features="foo")

            # validate the error message.
            self.assertTrue("current_features argument is None or incorrect type." in context.exception)

        # verify we handle "Churn", ["foo"]
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.select_next_feature(the_target_column="Churn", current_features=["foo"])

            # validate the error message.
            self.assertTrue("an element in current_features argument is not in dataframe." in context.exception)

    # test method for select_next_features()
    def test_select_next_feature(self):
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

        # get the the_model_results_dict
        the_model_results_dict = the_linear_model.build_mlr_for_single_comparison('Churn')

        # run assertions
        self.assertIsNotNone(the_model_results_dict)
        self.assertIsInstance(the_model_results_dict, dict)
        self.assertEqual(len(the_model_results_dict), 47)

        # loop over all the elements in the_model_results_dict
        for the_column in list(the_model_results_dict.keys()):
            # make sure the column is in method_results
            self.assertTrue(the_column in the_model_results_dict)

            # make sure the result returned is the correct type
            self.assertIsInstance(the_model_results_dict[the_column], Linear_Model_Result)

        # order the features
        ordered_df = order_features_by_r_squared(the_model_results_dict)

        # run assertions
        self.assertIsNotNone(ordered_df)
        self.assertIsInstance(ordered_df, DataFrame)
        self.assertEqual(len(ordered_df), 47)

        # test the first few values to make sure the ordering is correct
        self.assertEqual(ordered_df.iloc[0]['predictor'], 'Tenure')
        self.assertEqual(ordered_df.iloc[0]['r-squared'], 0.23561)
        self.assertEqual(ordered_df.iloc[1]['predictor'], 'Bandwidth_GB_Year')
        self.assertEqual(ordered_df.iloc[1]['r-squared'], 0.19499)

        # create the list of current features
        current_features = ['Tenure']

        # invoke the function
        the_model_results_dict = the_linear_model.select_next_feature(the_target_column='Churn',
                                                                      current_features=current_features)

        # run assertions
        self.assertIsNotNone(the_model_results_dict)
        self.assertIsInstance(the_model_results_dict, dict)
        self.assertEqual(len(the_model_results_dict), 46)

        # order the features
        ordered_df = order_features_by_r_squared(the_model_results_dict)

        # test the first few values to make sure the ordering is correct
        self.assertEqual(ordered_df.iloc[0]['predictor'], 'MonthlyCharge')
        self.assertEqual(ordered_df.iloc[0]['r-squared'], 0.37344)
        self.assertEqual(ordered_df.iloc[1]['predictor'], 'Bandwidth_GB_Year')
        self.assertEqual(ordered_df.iloc[1]['r-squared'], 0.32850)
        self.assertEqual(ordered_df.iloc[2]['predictor'], 'StreamingMovies')
        self.assertEqual(ordered_df.iloc[2]['r-squared'], 0.31850)
        self.assertEqual(ordered_df.iloc[3]['predictor'], 'StreamingTV')
        self.assertEqual(ordered_df.iloc[3]['r-squared'], 0.28906)
        self.assertEqual(ordered_df.iloc[4]['predictor'], 'Contract_Two Year')
        self.assertEqual(ordered_df.iloc[4]['r-squared'], 0.26407)

    # negative test method for fit_a_model()
    def test_fit_a_model_negative(self):
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

        # verify we handle None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.fit_a_model(the_target_column=None, current_features=None, model_type=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle 'foo', None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.fit_a_model(the_target_column='foo', current_features=None, model_type=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle 'MonthlyCharge', None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.fit_a_model(the_target_column='MonthlyCharge', current_features=None, model_type=None)

            # validate the error message.
            self.assertTrue("current_features argument is None or incorrect type." in context.exception)

        # verify we handle 'MonthlyCharge', ['Tenure', 'foo', 'Children'], None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                         current_features=['Tenure', 'foo', 'Children'],
                                         model_type=None)

            # validate the error message.
            self.assertTrue("an element in the_variable_columns argument is not in dataframe." in context.exception)

        # verify we handle 'MonthlyCharge', ['Tenure', 'Children'], None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                         current_features=['Tenure', 'Children'],
                                         model_type=None)

            # validate the error message.
            self.assertTrue("model_type is not a valid value." in context.exception)

    # test method for fit_a_model()
    def test_fit_a_model(self):
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

        # invoke method fit_a_model()
        the_result = the_linear_model.fit_a_model(the_target_column='Churn', current_features=the_variable_columns,
                                                  model_type=MT_LINEAR_REGRESSION)

        # run assertions on Linear_Model_Result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # validate the p-values on the Linear_Model_Result
        the_result_df = the_result.get_results_dataframe()

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

        # make sure the Linear_Model_Result is stored on the Linear_Model
        self.assertIsNotNone(the_linear_model.the_result)
        self.assertIsInstance(the_linear_model.the_result, Linear_Model_Result)

    # negative tests for reduce_a_model()
    def test_reduce_a_model_negative(self):
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

        # verify we handle None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column=None, current_features=None, model_type=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo", None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="foo", current_features=None, model_type=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle "Churn", None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn", current_features=None, model_type=None)

            # validate the error message.
            self.assertTrue("current_features argument is None or incorrect type." in context.exception)

        # verify we handle "Churn", ["foo"], None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn", current_features=["foo"], model_type=None)

            # validate the error message.
            self.assertTrue("an element in the_variable_columns argument is not in dataframe." in context.exception)

        # verify we handle "Churn", ["Population", "foo", "TimeZone"], None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "foo", "TimeZone"], model_type=None)

            # validate the error message.
            self.assertTrue("an element in the_variable_columns argument is not in dataframe." in context.exception)

        # verify we handle "Churn", ["Population", "TimeZone"], None, "0.5"
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "TimeZone"], model_type=None,
                                            max_p_value="0.5")

            # validate the error message.
            self.assertTrue("model_type is not a valid value." in context.exception)

        # verify we handle "Churn", ["Population", "TimeZone"], MT_LINEAR_REGRESSION, "0.5"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "TimeZone"],
                                            model_type=MT_LINEAR_REGRESSION,
                                            max_p_value="0.5")

            # validate the error message.
            self.assertTrue("max_p_value was None or incorrect type." in context.exception)

        # verify we handle "Churn", ["Population", "TimeZone"], MT_LINEAR_REGRESSION, 2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "TimeZone"],
                                            model_type=MT_LINEAR_REGRESSION,
                                            max_p_value=2.0)

            # validate the error message.
            self.assertTrue("max_p_value must be in range (0,1)" in context.exception)

        # verify we handle "Churn", ["Population", "TimeZone"], MT_LINEAR_REGRESSION, -2.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "TimeZone"],
                                            model_type=MT_LINEAR_REGRESSION,
                                            max_p_value=-2.0)

            # validate the error message.
            self.assertTrue("max_p_value must be in range (0,1)" in context.exception)

        # verify we handle "Churn", ["Population", "TimeZone"], MT_LINEAR_REGRESSION, 0.5, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "TimeZone"],
                                            model_type=MT_LINEAR_REGRESSION,
                                            max_p_value=0.5,
                                            max_vif=None)

            # validate the error message.
            self.assertTrue("max_vif was None or incorrect type." in context.exception)

        # verify we handle "Churn", ["Population", "TimeZone"], MT_LINEAR_REGRESSION, 0.5, 4.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.reduce_a_model(the_target_column="Churn",
                                            current_features=["Population", "TimeZone"],
                                            model_type=MT_LINEAR_REGRESSION,
                                            max_p_value=0.5,
                                            max_vif=4.0)

            # validate the error message.
            self.assertTrue("max_vif must be > 5.0" in context.exception)

    # test method for reduce_a_model()
    def test_reduce_a_model(self):
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

        # invoke method fit_a_model() with max p-value
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

        # lower the max_p_value again to .20 and see what we get
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.20)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertTrue(results_df['p-value'].max() < 0.20)
        self.assertEqual(len(results_df), 19)

        # lower the max_p_value again to .08 and see what we get
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.08)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertTrue(results_df['p-value'].max() < 0.08)
        self.assertEqual(len(results_df), 14)

        # lower the max_p_value again to .001 and see what we get
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.001)

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

        print(the_result.get_model().summary())

    # test method for reduce_a_model() that just looks for expected VIF reductions
    def test_reduce_a_model_only_for_VIF(self):
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
        self.assertEqual(len(method_results), 47)  # we should have 48-1
        self.assertFalse('MonthlyCharge' in method_results)  # just for good measure

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # invoke method fit_a_model() with max p-value of 0.999999
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.999999)

        # run basic assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Linear_Model_Result)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run basic assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertTrue(len(results_df) > 0)
        self.assertTrue(results_df['p-value'].max() < 0.999999)
        self.assertEqual(len(results_df), 46)  # only one result should be removed

        # get the storage from the_linear_model
        the_storage_dict = the_linear_model.model_storage

        # run assertions
        self.assertIsNotNone(the_storage_dict)
        self.assertIsInstance(the_storage_dict, dict)
        self.assertEqual(len(the_storage_dict), 1)
        self.assertTrue(1 in the_storage_dict)  # prove we only have one result
        self.assertIsInstance(the_storage_dict[1], tuple)

        # get the tuple from key
        the_tuple = the_storage_dict[1]

        # run assertions on the_tuple
        self.assertEqual(the_tuple[0], "VIF")
        self.assertEqual(the_tuple[1], "Bandwidth_GB_Year")

    # test method for reduce_a_model() that eliminates for VIF, then p-value
    def test_reduce_a_model_for_VIF_then_p_value(self):
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
        self.assertEqual(len(method_results), 47)  # we should have 48-1
        self.assertFalse('MonthlyCharge' in method_results)  # just for good measure

        # validate lists are the same
        for the_column in the_variable_columns:
            self.assertTrue(the_column in method_results)

        for the_column in method_results:
            self.assertTrue(the_column in the_variable_columns)

        # invoke method fit_a_model() with max p-value of 0.80
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
        self.assertEqual(len(results_df), 39)  # only one result should be removed

        # get the storage from the_linear_model
        the_storage_dict = the_linear_model.model_storage

        # run assertions
        self.assertIsNotNone(the_storage_dict)
        self.assertIsInstance(the_storage_dict, dict)
        self.assertEqual(len(the_storage_dict), 8)

        # get the tuple from key
        the_tuple = the_storage_dict[1]
        self.assertEqual(the_tuple[0], "VIF")
        self.assertEqual(the_tuple[1], "Bandwidth_GB_Year")

        the_tuple = the_storage_dict[2]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "Courteous_Exchange")

        the_tuple = the_storage_dict[3]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "Email")

        the_tuple = the_storage_dict[4]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "Marital_Married")

        the_tuple = the_storage_dict[5]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "Area_Urban")

        the_tuple = the_storage_dict[6]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "TimeZone")

        the_tuple = the_storage_dict[7]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "Children")

        the_tuple = the_storage_dict[8]
        self.assertEqual(the_tuple[0], "P_VALUE")
        self.assertEqual(the_tuple[1], "PaymentMethod_Mailed Check")

    # negative tests for solve_for_target()
    def test_solve_for_target_negative(self):
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

        # verify we handle None, None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column=None, model_type=None,
                                                max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is None or incorrect type." in context.exception)

        # verify we handle "foo", None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column="foo", model_type=None,
                                                max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("the_target_column argument is not in dataframe." in context.exception)

        # verify we handle "Churn", None, None, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column="Churn", model_type=None,
                                                max_p_value=None, the_max_vif=None)

            # validate the error message.
            self.assertTrue("model_type is not a valid value." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, 2.0, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column="Churn", model_type=MT_LINEAR_REGRESSION,
                                                max_p_value=2.0, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_p_value was greater than 1.0" in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, -2.0, None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column="Churn", model_type=MT_LINEAR_REGRESSION,
                                                max_p_value=-2.0, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_p_value was greater than 1.0" in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, 0.5, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column="Churn", model_type=MT_LINEAR_REGRESSION,
                                                max_p_value=0.5, the_max_vif=None)

            # validate the error message.
            self.assertTrue("max_vif was None or incorrect type." in context.exception)

        # verify we handle "Churn", MT_LOGISTIC_REGRESSION, -2.0, 4.0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.solve_for_target(the_target_column="Churn", model_type=MT_LINEAR_REGRESSION,
                                                max_p_value=0.5, the_max_vif=4.0)

            # validate the error message.
            self.assertTrue("max_vif must be > 5.0" in context.exception)

    # test method for solve_for_target()
    def test_solve_for_target(self):
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

        # invoke method
        the_linear_model.solve_for_target(the_target_column='Churn', model_type=MT_LINEAR_REGRESSION,
                                          max_p_value=0.001, the_max_vif=10.0)

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
        self.assertEqual(len(results_df), 12)

        # make sure the Linear_Model_Result is stored on the Linear_Model
        self.assertIsNotNone(the_linear_model.get_the_result())
        self.assertIsInstance(the_linear_model.get_the_result(), Linear_Model_Result)

        # make sure the model result stored is the same.
        self.assertTrue(len(the_linear_model.get_the_result().get_results_dataframe()) > 0)
        self.assertTrue(the_linear_model.get_the_result().get_results_dataframe()['p-value'].max() < 0.001)
        self.assertEqual(len(the_linear_model.get_the_result().get_results_dataframe()), 12)

        # loop over columns of results_df and compare to the stored linear model result
        for the_column in results_df.columns:
            self.assertTrue(the_column in the_linear_model.get_the_result().get_results_dataframe().columns)

    # negative test method for log_model_summary_to_console()
    def test_log_model_summary_to_console_negative(self):
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

        # verify we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.log_model_summary_to_console(the_type=None, the_step=None)

            # validate the error message.
            self.assertTrue("the_type is None or incorrect type." in context.exception)

        # verify we handle "foo", None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.log_model_summary_to_console(the_type="foo", the_step=None)

            # validate the error message.
            self.assertTrue("the_type argument is note recognized." in context.exception)

        # verify we handle LM_INITIAL_MODEL, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_linear_model.log_model_summary_to_console(the_type=LM_INITIAL_MODEL, the_step=None)

            # validate the error message.
            self.assertTrue("the_step is None or incorrect type." in context.exception)

        # verify we handle LM_STEP, 0
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.log_model_summary_to_console(the_type=LM_STEP, the_step=0)

            # validate the error message.
            self.assertTrue("the_step argument must be have domain [1, inf)." in context.exception)

        # verify we handle LM_STEP, -1
        with self.assertRaises(ValueError) as context:
            # invoke the method
            the_linear_model.log_model_summary_to_console(the_type=LM_STEP, the_step=-1)

            # validate the error message.
            self.assertTrue("the_step argument must be have domain [1, inf)." in context.exception)

    # test method for log_model_summary_to_console()
    def test_log_model_summary_to_console(self):
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

        # invoke method fit_a_model() with max p-value
        the_result = the_linear_model.reduce_a_model(the_target_column='MonthlyCharge',
                                                     current_features=the_variable_columns,
                                                     model_type=MT_LINEAR_REGRESSION,
                                                     max_p_value=0.80)

        # ***************************************************
        # This test just verifies that we can pass arguments and get the expected outcome.
        # ***************************************************

        # invoke the method for LM_INITIAL_MODEL
        the_linear_model.log_model_summary_to_console(the_type=LM_INITIAL_MODEL)

        # invoke the method for LM_FINAL_MODEL
        the_linear_model.log_model_summary_to_console(the_type=LM_FINAL_MODEL)

        # invoke the method for LM_STEP, step 1
        the_linear_model.log_model_summary_to_console(the_type=LM_STEP, the_step=1)

        # invoke the method for LM_STEP, step 2
        the_linear_model.log_model_summary_to_console(the_type=LM_STEP, the_step=2)

    # test method for get_the_result()
    def test_get_the_result(self):
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

        # call fit_a_model()
        the_result = the_linear_model.fit_a_model(the_target_column='MonthlyCharge',
                                                  current_features=the_variable_columns,
                                                  model_type=MT_LINEAR_REGRESSION)

        # invoke the method
        a_result = the_linear_model.get_the_result()

        # run assertions
        self.assertIsNotNone(a_result)
        self.assertIsInstance(a_result, Linear_Model_Result)
