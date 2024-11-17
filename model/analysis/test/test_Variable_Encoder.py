import unittest
import pandas as pd

from pandas import DataFrame
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Variable_Encoder import Variable_Encoder
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_209_CHURN


class test_Variable_Encoder(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_209_CHURN

    # negative tests for the init() method
    def test_init_negative(self):
        # verify we handle None
        with self.assertRaises(ValueError) as context:
            # invoke the method
            Variable_Encoder(unencoded_df=None)

            # validate the error message.
            self.assertTrue("unencoded_df is None or incorrect type." in context.exception)

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
        self.assertEqual(len(the_df), 10000)

        # invoke the method
        var_encoder = Variable_Encoder(unencoded_df=pa.analyzer.the_df)

        # run assertions
        self.assertIsNotNone(var_encoder)
        self.assertIsInstance(var_encoder, Variable_Encoder)
        self.assertIsNotNone(var_encoder.storage)
        self.assertIsInstance(var_encoder.storage, dict)
        self.assertIsNotNone(var_encoder.original_df)
        self.assertIsInstance(var_encoder.original_df, DataFrame)
        self.assertEqual(len(var_encoder.original_df), 10000)
        self.assertEqual(len(var_encoder.original_df.columns), 39)
        self.assertIsNone(var_encoder.encoded_df)

    # test method for encode_dataframe() with no value for drop_first
    def test_encode_dataframe_drop_first_true(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions about our starting data point
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # create the Variable_Encoder
        var_encoder = Variable_Encoder(unencoded_df=pa.analyzer.the_df)

        # invoke the method. Default argument for drop_first=True
        var_encoder.encode_dataframe()

        # get the original_df and encoded_df from var_encoder
        original_df = var_encoder.original_df
        encoded_df = var_encoder.encoded_df

        # run assertions that original_df didn't get touched
        self.assertIsNotNone(original_df)
        self.assertIsInstance(original_df, DataFrame)
        self.assertEqual(len(original_df.columns), 39)
        self.assertEqual(len(original_df), 10000)

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df.columns), 48)
        self.assertEqual(len(encoded_df), 10000)

        # run assertions that the list of object variables are missing
        self.assertFalse('Area' in encoded_df)
        self.assertFalse('Marital' in encoded_df)
        self.assertFalse('Gender' in encoded_df)
        self.assertFalse('Contract' in encoded_df)
        self.assertFalse('InternetService' in encoded_df)
        self.assertFalse('PaymentMethod' in encoded_df)

        # run assertions that the new encoded columns are present
        self.assertFalse('Area_Rural' in encoded_df)
        self.assertTrue('Area_Suburban' in encoded_df)
        self.assertTrue('Area_Urban' in encoded_df)

        self.assertFalse('Marital_Divorced' in encoded_df)
        self.assertTrue('Marital_Married' in encoded_df)
        self.assertTrue('Marital_Never Married' in encoded_df)
        self.assertTrue('Marital_Separated' in encoded_df)
        self.assertTrue('Marital_Widowed' in encoded_df)

        self.assertFalse('Gender_Female' in encoded_df)
        self.assertTrue('Gender_Male' in encoded_df)
        self.assertTrue('Gender_Nonbinary' in encoded_df)

        self.assertFalse('Contract_Month-to-month' in encoded_df)
        self.assertTrue('Contract_One year' in encoded_df)
        self.assertTrue('Contract_Two Year' in encoded_df)

        self.assertFalse('InternetService_DSL' in encoded_df)
        self.assertTrue('InternetService_Fiber Optic' in encoded_df)
        self.assertTrue('InternetService_No response' in encoded_df)

        self.assertFalse('PaymentMethod_Bank Transfer(automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Electronic Check' in encoded_df)
        self.assertTrue('PaymentMethod_Mailed Check' in encoded_df)

        # validate some of the types, but all of them are bool
        self.assertEqual(encoded_df['InternetService_Fiber Optic'].dtype, bool)
        self.assertEqual(encoded_df['PaymentMethod_Mailed Check'].dtype, bool)
        self.assertEqual(encoded_df['Gender_Male'].dtype, bool)
        self.assertEqual(encoded_df['Marital_Married'].dtype, bool)
        self.assertEqual(encoded_df['Area_Urban'].dtype, bool)

    # test method for encode_dataframe() with drop_first=False
    def test_encode_dataframe_drop_first_false(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions about our starting data point
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # create the Variable_Encoder
        var_encoder = Variable_Encoder(unencoded_df=pa.analyzer.the_df)

        # invoke the method with drop_first = False
        var_encoder.encode_dataframe(drop_first=False)

        # get the original_df and encoded_df from var_encoder
        original_df = var_encoder.original_df
        encoded_df = var_encoder.encoded_df

        # run assertions that original_df didn't get touched
        self.assertIsNotNone(original_df)
        self.assertIsInstance(original_df, DataFrame)
        self.assertEqual(len(original_df.columns), 39)
        self.assertEqual(len(original_df), 10000)

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df.columns), 54)
        self.assertEqual(len(encoded_df), 10000)

        # run assertions that the list of object variables are missing
        self.assertFalse('Area' in encoded_df)
        self.assertFalse('Marital' in encoded_df)
        self.assertFalse('Gender' in encoded_df)
        self.assertFalse('Contract' in encoded_df)
        self.assertFalse('InternetService' in encoded_df)
        self.assertFalse('PaymentMethod' in encoded_df)

        # run assertions that the new encoded columns are present
        self.assertTrue('Area_Rural' in encoded_df)
        self.assertTrue('Area_Suburban' in encoded_df)
        self.assertTrue('Area_Urban' in encoded_df)

        self.assertTrue('Marital_Divorced' in encoded_df)
        self.assertTrue('Marital_Married' in encoded_df)
        self.assertTrue('Marital_Never Married' in encoded_df)
        self.assertTrue('Marital_Separated' in encoded_df)
        self.assertTrue('Marital_Widowed' in encoded_df)

        self.assertTrue('Gender_Female' in encoded_df)
        self.assertTrue('Gender_Male' in encoded_df)
        self.assertTrue('Gender_Nonbinary' in encoded_df)

        self.assertTrue('Contract_Month-to-month' in encoded_df)
        self.assertTrue('Contract_One year' in encoded_df)
        self.assertTrue('Contract_Two Year' in encoded_df)

        self.assertTrue('InternetService_DSL' in encoded_df)
        self.assertTrue('InternetService_Fiber Optic' in encoded_df)
        self.assertTrue('InternetService_No response' in encoded_df)

        self.assertTrue('PaymentMethod_Bank Transfer(automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Electronic Check' in encoded_df)
        self.assertTrue('PaymentMethod_Mailed Check' in encoded_df)

        # validate some of the types, but all of them are bool
        self.assertEqual(encoded_df['InternetService_Fiber Optic'].dtype, bool)
        self.assertEqual(encoded_df['PaymentMethod_Mailed Check'].dtype, bool)
        self.assertEqual(encoded_df['Gender_Male'].dtype, bool)
        self.assertEqual(encoded_df['Marital_Married'].dtype, bool)
        self.assertEqual(encoded_df['Area_Urban'].dtype, bool)

    # test of one hot encoding a single categorical variable 'Area' of type object
    def test_encoding_Area(self):
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

        # run assertions about our starting data point
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        new_df = pd.get_dummies(the_df, columns=['Area'])

        # run basic assertions
        self.assertIsNotNone(new_df)
        self.assertIsInstance(new_df, DataFrame)
        self.assertEqual(len(new_df.columns), 41)
        self.assertEqual(len(new_df), 10000)

        # run assertions for Area's encoding
        self.assertFalse('Area' in new_df.columns)
        self.assertTrue('Area_Rural' in new_df.columns)
        self.assertTrue('Area_Suburban' in new_df.columns)
        self.assertTrue('Area_Urban' in new_df.columns)

    # test of one encoding a single categorical variable 'Churn' of type boolean
    def test_encoding_Churn(self):
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

        # run assertions about our starting data point
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        new_df = pd.get_dummies(the_df, columns=['Churn'])

        # run basic assertions
        self.assertIsNotNone(new_df)
        self.assertIsInstance(new_df, DataFrame)
        self.assertEqual(len(new_df.columns), 40)
        self.assertEqual(len(new_df), 10000)

        # run assertions for Area's encoding
        self.assertFalse('Churn' in new_df.columns)
        self.assertTrue('Churn_False' in new_df.columns)
        self.assertTrue('Churn_True' in new_df.columns)

    # test method for get_encoded_df with no drop_first argument
    def test_get_encoded_df_no_drop_first_argument(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions about our starting data point
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # create the Variable_Encoder
        var_encoder = Variable_Encoder(unencoded_df=pa.analyzer.the_df)

        # get the original_df and encoded_df from var_encoder
        original_df = var_encoder.original_df
        encoded_df = var_encoder.encoded_df

        # run assertions that original_df didn't get touched
        self.assertIsNotNone(original_df)
        self.assertIsInstance(original_df, DataFrame)
        self.assertEqual(len(original_df.columns), 39)
        self.assertEqual(len(original_df), 10000)

        # run assertions on encoded_df
        self.assertIsNone(encoded_df)

        # invoke the method
        encoded_df = var_encoder.get_encoded_dataframe()

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df.columns), 48)
        self.assertEqual(len(encoded_df), 10000)

        # run assertions that the list of object variables are missing
        self.assertFalse('Area' in encoded_df)
        self.assertFalse('Marital' in encoded_df)
        self.assertFalse('Gender' in encoded_df)
        self.assertFalse('Contract' in encoded_df)
        self.assertFalse('InternetService' in encoded_df)
        self.assertFalse('PaymentMethod' in encoded_df)

        # run assertions that the new encoded columns are present
        self.assertFalse('Area_Rural' in encoded_df)  # due to drop_first=True
        self.assertTrue('Area_Suburban' in encoded_df)
        self.assertTrue('Area_Urban' in encoded_df)

        self.assertFalse('Marital_Divorced' in encoded_df)  # due to drop_first=True
        self.assertTrue('Marital_Married' in encoded_df)
        self.assertTrue('Marital_Never Married' in encoded_df)
        self.assertTrue('Marital_Separated' in encoded_df)
        self.assertTrue('Marital_Widowed' in encoded_df)

        self.assertFalse('Gender_Female' in encoded_df)  # due to drop_first=True
        self.assertTrue('Gender_Male' in encoded_df)
        self.assertTrue('Gender_Nonbinary' in encoded_df)

        self.assertFalse('Contract_Month-to-month' in encoded_df)  # due to drop_first=True
        self.assertTrue('Contract_One year' in encoded_df)
        self.assertTrue('Contract_Two Year' in encoded_df)

        self.assertFalse('InternetService_DSL' in encoded_df)  # due to drop_first=True
        self.assertTrue('InternetService_Fiber Optic' in encoded_df)
        self.assertTrue('InternetService_No response' in encoded_df)

        self.assertFalse('PaymentMethod_Bank Transfer(automatic)' in encoded_df)  # due to drop_first=True
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Electronic Check' in encoded_df)
        self.assertTrue('PaymentMethod_Mailed Check' in encoded_df)

        # validate some of the types, but all of them are bool
        self.assertEqual(encoded_df['InternetService_Fiber Optic'].dtype, bool)
        self.assertEqual(encoded_df['PaymentMethod_Mailed Check'].dtype, bool)
        self.assertEqual(encoded_df['Gender_Male'].dtype, bool)
        self.assertEqual(encoded_df['Marital_Married'].dtype, bool)
        self.assertEqual(encoded_df['Area_Urban'].dtype, bool)

    # test method for get_encoded_df with drop_first argument = False
    def test_get_encoded_df_drop_first_argument_false(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # run assertions about our starting data point
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # create the Variable_Encoder
        var_encoder = Variable_Encoder(unencoded_df=pa.analyzer.the_df)

        # get the original_df and encoded_df from var_encoder
        original_df = var_encoder.original_df
        encoded_df = var_encoder.encoded_df

        # run assertions that original_df didn't get touched
        self.assertIsNotNone(original_df)
        self.assertIsInstance(original_df, DataFrame)
        self.assertEqual(len(original_df.columns), 39)
        self.assertEqual(len(original_df), 10000)

        # run assertions on encoded_df
        self.assertIsNone(encoded_df)

        # invoke the method
        encoded_df = var_encoder.get_encoded_dataframe(drop_first=False)

        # run assertions on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(encoded_df, DataFrame)
        self.assertEqual(len(encoded_df.columns), 54)
        self.assertEqual(len(encoded_df), 10000)

        # run assertions that the list of object variables are missing
        self.assertFalse('Area' in encoded_df)
        self.assertFalse('Marital' in encoded_df)
        self.assertFalse('Gender' in encoded_df)
        self.assertFalse('Contract' in encoded_df)
        self.assertFalse('InternetService' in encoded_df)
        self.assertFalse('PaymentMethod' in encoded_df)

        # run assertions that the new encoded columns are present
        self.assertTrue('Area_Rural' in encoded_df)  # due to drop_first=True
        self.assertTrue('Area_Suburban' in encoded_df)
        self.assertTrue('Area_Urban' in encoded_df)

        self.assertTrue('Marital_Divorced' in encoded_df)  # due to drop_first=True
        self.assertTrue('Marital_Married' in encoded_df)
        self.assertTrue('Marital_Never Married' in encoded_df)
        self.assertTrue('Marital_Separated' in encoded_df)
        self.assertTrue('Marital_Widowed' in encoded_df)

        self.assertTrue('Gender_Female' in encoded_df)  # due to drop_first=True
        self.assertTrue('Gender_Male' in encoded_df)
        self.assertTrue('Gender_Nonbinary' in encoded_df)

        self.assertTrue('Contract_Month-to-month' in encoded_df)  # due to drop_first=True
        self.assertTrue('Contract_One year' in encoded_df)
        self.assertTrue('Contract_Two Year' in encoded_df)

        self.assertTrue('InternetService_DSL' in encoded_df)  # due to drop_first=True
        self.assertTrue('InternetService_Fiber Optic' in encoded_df)
        self.assertTrue('InternetService_No response' in encoded_df)

        self.assertTrue('PaymentMethod_Bank Transfer(automatic)' in encoded_df)  # due to drop_first=True
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Electronic Check' in encoded_df)
        self.assertTrue('PaymentMethod_Mailed Check' in encoded_df)

        # validate some of the types, but all of them are bool
        self.assertEqual(encoded_df['InternetService_Fiber Optic'].dtype, bool)
        self.assertEqual(encoded_df['PaymentMethod_Mailed Check'].dtype, bool)
        self.assertEqual(encoded_df['Gender_Male'].dtype, bool)
        self.assertEqual(encoded_df['Marital_Married'].dtype, bool)
        self.assertEqual(encoded_df['Area_Urban'].dtype, bool)