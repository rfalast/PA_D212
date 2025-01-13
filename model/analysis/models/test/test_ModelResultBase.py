import unittest

from pandas import DataFrame, Series

from model.Project_Assessment import Project_Assessment
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_212_CHURN


class test_ModelsBase(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"
    OVERRIDE_PATH = "../../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_212_CHURN

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

        # get the list of columns
        the_variable_columns = linear_model.encoded_df.columns.to_list()

        # verify we handle None, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            ModelResultBase(the_target_variable=None, the_variables_list=None, the_encoded_df=None)

        # validate the error message.
        self.assertTrue("the_target_variable is None or incorrect type." in context.exception.args)

        # verify we handle 'Churn', None, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            ModelResultBase(the_target_variable='Churn', the_variables_list=None, the_encoded_df=None)

        # validate the error message.
        self.assertTrue("the_variables_list is None or incorrect type." in context.exception.args)

        # verify we handle 'Churn', the_variable_columns, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            ModelResultBase(the_target_variable='Churn', the_variables_list=the_variable_columns, the_encoded_df=None)

        # validate the error message.
        self.assertTrue("the_encoded_df is None or incorrect type." in context.exception.args)

    # test init() method
    def test_init(self):
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

        # declare the base
        the_base = None

        # invoke the method
        the_base = ModelResultBase(the_target_variable='Churn',
                                   the_variables_list=the_variable_columns,
                                   the_encoded_df=X)

        # run assertions
        self.assertIsNotNone(the_base)
        self.assertIsInstance(the_base, ModelResultBase)
        self.assertIsNotNone(the_base.the_target_variable)
        self.assertIsNotNone(the_base.the_variables_list)
        self.assertIsNotNone(the_base.the_df)

