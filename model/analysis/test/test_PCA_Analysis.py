import unittest

from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from model.constants.BasicConstants import Z_SCORE
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.PCA_Analysis import PCA_Analysis
from util.CSV_loader import CSV_Loader


# test class for PCA_Analysis
class test_PCA_Analysis(unittest.TestCase):
    # constants
    OVERRIDE_PATH = "../../../resources/Output/"
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    VALID_COLUMN_1 = "Age"
    VALID_COLUMN_2 = "Tenure"
    VALID_COLUMN_INDEX = ["Age", "Tenure", "MonthlyCharge", "Bandwidth_GB_Year", "Yearly_equip_failure",
                          "Contacts", "Outage_sec_perweek", "Children", "Income"]

    VALID_COLUMN_DICT_NORM = {"PC1": "Age", "PC2": "Tenure", "PC3": "MonthlyCharge", "PC4": "Bandwidth_GB_Year",
                              "PC5": "Yearly_equip_failure", "PC6": "Contacts", "PC7": "Outage_sec_perweek",
                              "PC8": "Children", "PC9": "Income"}

    VALID_COLUMN_DICT = {"PC1": "Age", "PC2": "Tenure", "PC3": "MonthlyCharge", "PC4": "Bandwidth_GB_Year",
                         "PC5": "Yearly_equip_failure", "PC6": "Contacts", "PC7": "Outage_sec_perweek",
                         "PC8": "Children", "PC9": "Income"}

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    pca_column_dict = {"PC0": "Population", "PC1": "Children", "PC2": "Age", "PC3": "Income",
                       "PC4": "Outage_sec_perweek", "PC5": "Email", "PC6": "Contacts", "PC7": "Yearly_equip_failure",
                       "PC8": "Port_modem", "PC9": "Tablet", "PC10": "OnlineSecurity", "PC11": "TechSupport",
                       "PC12": "PaperlessBilling", "PC13": "Tenure", "PC14": "Bandwidth_GB_Year",
                       "PC15": "Timely_Response", "PC16": "Timely_Fixes", "PC17": "Timely_Replacements",
                       "PC18": "Reliability", "PC19": "Options", "PC20": "Respectful_Response",
                       "PC21": "Courteous_Exchange", "PC22": "Active_Listening"}

    # negative tests for the init() method
    def test_init_negative(self):
        # variable definition
        pca_analysis = None

        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # test that we can handle None, None
        with self.assertRaises(ValueError) as context:
            # call the constructor
            pca_analysis = PCA_Analysis(None, None)

            # validate the error message.
            self.assertTrue("the DatasetAnalyzer is None or incorrect type." in context.exception)

        # test that we can handle 1, None
        with self.assertRaises(ValueError) as context:
            # call the constructor
            pca_analysis = PCA_Analysis(1, None)

            # validate the error message.
            self.assertTrue("the DatasetAnalyzer is None or incorrect type." in context.exception)

        # test that we can handle dsa, None
        with self.assertRaises(ValueError) as context:
            # call the constructor
            pca_analysis = PCA_Analysis(dsa, None)

            # validate the error message.
            self.assertTrue("the column_dict is None or incorrect type." in context.exception)

    # tests for init() method
    def test_init(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create an instance
        pca_analysis = PCA_Analysis(dsa, self.VALID_COLUMN_DICT)

        # run assertions
        self.assertIsNotNone(pca_analysis.dataset_analyzer)
        self.assertIsNotNone(pca_analysis.column_dict)
        self.assertIsInstance(pca_analysis.dataset_analyzer, DatasetAnalyzer)
        self.assertIsInstance(pca_analysis.column_dict, dict)

        # verify that the original and normalized df are present
        self.assertIsNotNone(pca_analysis.original_df)
        self.assertIsNotNone(pca_analysis.normalized_df)
        self.assertIsInstance(pca_analysis.original_df, DataFrame)
        self.assertIsInstance(pca_analysis.normalized_df, DataFrame)

        # check dimensionality of the two dataframes
        self.assertEqual(len(pca_analysis.original_df.columns), 9)
        self.assertEqual(len(pca_analysis.normalized_df.columns), 9)
        self.assertEqual(len(pca_analysis.original_df), 10000)
        self.assertEqual(len(pca_analysis.normalized_df), 10000)

    # tests for get_original_df() method
    def test_get_original_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create an instance
        pca_analysis = PCA_Analysis(dsa, self.VALID_COLUMN_DICT)

        # get the list of column names on the pca_analysis internal variable original_df
        original_columns = pca_analysis.get_original_df().columns

        # run assertions
        self.assertIsNotNone(pca_analysis.get_original_df())
        self.assertIsNotNone(original_columns)
        self.assertTrue(len(original_columns), len(self.VALID_COLUMN_INDEX))
        self.assertIsInstance(pca_analysis.get_original_df(), DataFrame)

        # verify that the original dataframe on the dataset analyzer has not been touched.
        self.assertEqual(len(dsa.the_df.columns), 49)
        self.assertEqual(len(dsa.the_df), 10000)

        # verify that the normalized dataframe on the dsa has not been touched.
        self.assertEqual(len(dsa.the_normal_df.columns), 23)
        self.assertEqual(len(dsa.the_normal_df), 10000)

        # assertions on pca_analysis.get_original_df()
        self.assertFalse("CaseOrder" in pca_analysis.original_df)
        self.assertTrue("Age" in pca_analysis.original_df)
        self.assertTrue("Tenure" in pca_analysis.original_df)
        self.assertTrue("MonthlyCharge" in pca_analysis.original_df)
        self.assertTrue("Bandwidth_GB_Year" in pca_analysis.original_df)
        self.assertTrue("Yearly_equip_failure" in pca_analysis.original_df)
        self.assertTrue("Contacts" in pca_analysis.original_df)
        self.assertTrue("Outage_sec_perweek" in pca_analysis.original_df)
        self.assertTrue("Children" in pca_analysis.original_df)
        self.assertTrue("Income" in pca_analysis.original_df)

        # assertions on dimensionality
        self.assertEqual(len(pca_analysis.original_df), 10000)
        self.assertEqual(len(pca_analysis.original_df.columns), 9)

    # tests for get_normalized_df() method
    def test_get_normalized_df(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create an instance
        pca_analysis = PCA_Analysis(dsa, self.VALID_COLUMN_DICT)

        # get the list of column names on the pca_analysis internal variable normalized_df
        normalized_columns = pca_analysis.get_normalized_df().columns

        # run assertions
        self.assertIsNotNone(pca_analysis.get_normalized_df())
        self.assertIsInstance(pca_analysis.get_normalized_df(), DataFrame)
        self.assertIsInstance(pca_analysis.get_normalized_df()[self.VALID_COLUMN_1 + Z_SCORE], Series)

        # assess if the columns match expectations
        self.assertIsNotNone(normalized_columns)
        self.assertTrue(len(normalized_columns), len(self.VALID_COLUMN_INDEX))

        # verify that the original dataframe on the dataset analyzer has not been touched.
        self.assertEqual(len(dsa.the_df.columns), 49)
        self.assertEqual(len(dsa.the_df), 10000)

        # verify that the normalized dataframe on the dsa has not been touched.
        self.assertEqual(len(dsa.the_normal_df.columns), 23)
        self.assertEqual(len(dsa.the_normal_df), 10000)

        # make sure that the columns in the pca_analysis.get_original_df() match expected results
        self.assertFalse("CaseOrder" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Age" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Tenure" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("MonthlyCharge" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Bandwidth_GB_Year" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Yearly_equip_failure" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Contacts" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Outage_sec_perweek" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Children" + Z_SCORE in pca_analysis.get_normalized_df())
        self.assertTrue("Income" + Z_SCORE in pca_analysis.get_normalized_df())

        # assertions on dimensionality
        self.assertEqual(len(pca_analysis.get_normalized_df()), 10000)
        self.assertEqual(len(pca_analysis.get_normalized_df().columns), 9)

    # tests for __truncate_dataframe__() method
    def test___truncate_dataframe__orig(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create an instance
        pca_analysis = PCA_Analysis(dsa, self.VALID_COLUMN_DICT)

        # call truncate dataframe
        new_df = pca_analysis.__truncate_dataframe__(pca_analysis.get_original_df())

        # first, test that when you call __truncate_dataframe__, that you don't change the stored objects.
        # This is a little bit of a nuance, but we want to verify there are no inadvertent state changes
        # to the DatasetAnalyzer
        self.assertEqual(len(dsa.the_df.columns), 49)
        self.assertEqual(len(dsa.the_df), 10000)

        # verify that the normalized dataframe on the dsa has not been touched.
        self.assertEqual(len(dsa.the_normal_df.columns), 23)
        self.assertEqual(len(dsa.the_normal_df), 10000)

        # assertions on the new_df
        self.assertIsNotNone(new_df)
        self.assertIsInstance(new_df, DataFrame)

        # test dimensions of the truncated dataframe new_df
        self.assertEqual(len(new_df), 10000)
        self.assertEqual(len(new_df.columns), 9)

        # run one negative test of a column that we KNOW should not be in new_df
        self.assertFalse("CaseOrder" in new_df)

        # assertions on new_df
        self.assertIsNotNone(new_df)
        self.assertIsInstance(new_df, DataFrame)
        self.assertTrue("Age" in new_df)
        self.assertTrue("Tenure" in new_df)
        self.assertTrue("MonthlyCharge" in new_df)
        self.assertTrue("Bandwidth_GB_Year" in new_df)
        self.assertTrue("Yearly_equip_failure" in new_df)
        self.assertTrue("Contacts" in new_df)
        self.assertTrue("Outage_sec_perweek" in new_df)
        self.assertTrue("Children" in new_df)
        self.assertTrue("Income" in new_df)

    # tests for __truncate_dataframe__() method
    def test___truncate_dataframe__norm(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create an instance
        pca_analysis = PCA_Analysis(dsa, self.VALID_COLUMN_DICT)

        # call truncate dataframe on get_normalized_df()
        new_df = pca_analysis.__truncate_dataframe__(pca_analysis.get_normalized_df())

        # first, test that when you call __truncate_dataframe__, that you don't change the stored objects.
        # This is a little bit of a nuance, but we want to verify there are no inadvertent state changes
        # to the DatasetAnalyzer
        self.assertEqual(len(dsa.the_df.columns), 49)
        self.assertEqual(len(dsa.the_df), 10000)

        # verify that the normalized dataframe on the dsa has not been touched.
        self.assertEqual(len(dsa.the_normal_df.columns), 23)
        self.assertEqual(len(dsa.the_normal_df), 10000)

        # assertions on the new_df
        self.assertIsNotNone(new_df)
        self.assertIsInstance(new_df, DataFrame)

        # test dimensions of the truncated dataframe new_df
        self.assertEqual(len(new_df), 10000)
        self.assertEqual(len(new_df.columns), 9)

        # run one negative test of a column that we KNOW should not be in new_df
        self.assertFalse("CaseOrder" in new_df)

        # assertions on new_df
        self.assertIsNotNone(new_df)
        self.assertIsInstance(new_df, DataFrame)
        self.assertTrue("Age" + Z_SCORE in new_df)
        self.assertTrue("Tenure" + Z_SCORE in new_df)
        self.assertTrue("MonthlyCharge" + Z_SCORE in new_df)
        self.assertTrue("Bandwidth_GB_Year" + Z_SCORE in new_df)
        self.assertTrue("Yearly_equip_failure" + Z_SCORE in new_df)
        self.assertTrue("Contacts" + Z_SCORE in new_df)
        self.assertTrue("Outage_sec_perweek" + Z_SCORE in new_df)
        self.assertTrue("Children" + Z_SCORE in new_df)
        self.assertTrue("Income" + Z_SCORE in new_df)

    # tests for perform_analysis() method
    def test_perform_analysis(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run the complete setup.
        dsa.run_complete_setup()

        # normalize dataset
        dsa.normalize_dataset()

        # create an instance
        pca_analysis = PCA_Analysis(dsa, self.VALID_COLUMN_DICT)

        # call perform_analysis()
        pca_analysis.perform_analysis()

        # run assertions on objects we expect to be populated
        self.assertIsNotNone(pca_analysis.the_pca)
        self.assertIsNotNone(pca_analysis.the_pca_df)
        self.assertIsNotNone(pca_analysis.pca_loadings)
        self.assertIsInstance(pca_analysis.the_pca, PCA)
        self.assertIsInstance(pca_analysis.the_pca_df, DataFrame)
        self.assertIsInstance(pca_analysis.pca_loadings, DataFrame)
