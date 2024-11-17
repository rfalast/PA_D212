import unittest
import pandas as pd

from scipy.stats import chi2_contingency, chi2
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.StatisticsGenerator import StatisticsGenerator
from model.constants.BasicConstants import D_209_CHURN, ANALYZE_DATASET_FULL
from model.constants.DatasetConstants import BOOL_COLUMN_KEY, FLOAT64_COLUMN_KEY, INT64_COLUMN_KEY, COLUMN_KEY, \
    OBJECT_COLUMN_KEY, DATA_TYPE_KEYS
from model.constants.StatisticsConstants import CORR_EXCEED_LEVEL, ALL_CORRELATIONS, DIST_NAME, DIST_PARAMETERS
from util.CSV_loader import CSV_Loader
from util.CommonUtils import are_tuples_the_same


class test_Statistics(unittest.TestCase):
    # constants
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']
    DATASET_KEY_NAME = D_209_CHURN

    # test init method
    def test_init_negative(self):
        # make sure we handle None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            StatisticsGenerator(None)

            # validate the error message.
            self.assertTrue("the_dsa was None or incorrect type." in context.exception)

    # test method for init()
    def test_init(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # invoke the method
        the_sg = StatisticsGenerator(dsa)

        # run assertions
        self.assertIsNotNone(the_sg.the_dsa)
        self.assertIsNotNone(the_sg.relevant_corr_storage)
        self.assertIsInstance(the_sg.the_dsa, DatasetAnalyzer)
        self.assertIsInstance(the_sg.relevant_corr_storage, list)

    # test method for find_correlations()
    def test_find_correlations(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke the method
        the_sg.find_correlations()

        # run assertions on relevant_corr_storage
        self.assertIsNotNone(the_sg)
        self.assertIsNotNone(the_sg.relevant_corr_storage)
        self.assertEqual(len(the_sg.relevant_corr_storage), 5)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[1],
                                            ('Tenure', 'Bandwidth_GB_Year', 0.991495)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[2],
                                            ('Timely_Response', 'Timely_Fixes', 0.663069)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[3],
                                            ('Timely_Response', 'Timely_Replacements', 0.578013)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[4],
                                            ('Timely_Fixes', 'Timely_Replacements', 0.520194)))

        # run assertions on all_corr_storage
        self.assertIsNotNone(the_sg)
        self.assertIsNotNone(the_sg.all_corr_storage)
        self.assertEqual(len(the_sg.all_corr_storage), 528)

    # test method for get_list_of_correlations()
    def test_get_list_of_correlations(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # tell the StatisticsGenerator to find all correlations
        the_sg.find_correlations()

        # invoke the method with no argument for the_type.  This should return the same as CORR_EXCEED_LEVEL
        the_list = the_sg.get_list_of_correlations()

        # run assertions
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 5)

        # invoke the method with no argument for the_type=CORR_EXCEED_LEVEL
        the_list2 = the_sg.get_list_of_correlations(the_type=CORR_EXCEED_LEVEL)

        # run assertions
        self.assertIsNotNone(the_list2)
        self.assertIsInstance(the_list2, list)
        self.assertEqual(len(the_list2), 5)
        self.assertEqual(len(the_list), len(the_list2))

        # invoke the method with no argument for the_type=ALL_CORRELATIONS
        the_list = the_sg.get_list_of_correlations(the_type=ALL_CORRELATIONS)

        # run assertions
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 528)

    # negative test method for add_to_storage()
    def test_add_to_storage_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # make sure we handle None, None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.add_to_storage(None, None)

            # validate the error message.
            self.assertTrue("storage_list is None or incorrect type." in context.exception)

        # make sure we handle "foo", None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.add_to_storage("foo", None)

            # validate the error message.
            self.assertTrue("storage_list is None or incorrect type." in context.exception)

        # make sure we handle the_sg.corr_storage, None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.add_to_storage(the_sg.relevant_corr_storage, None)

            # validate the error message.
            self.assertTrue("current_tuple is None or incorrect type." in context.exception)

        # make sure we handle the_sg.corr_storage, "foo"
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.add_to_storage(the_sg.relevant_corr_storage, "foo")

            # validate the error message.
            self.assertTrue("current_tuple is None or incorrect type." in context.exception)

    # test method for add_to_storage()
    def test_add_to_storage(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # create tuples
        tuple_1 = ('StreamingMovies', 'MonthlyCharge', 0.608115)
        tuple_2 = ('MonthlyCharge', 'StreamingMovies', 0.608115)
        tuple_3 = ('Timely_Fixes', 'Timely_Replacements', 0.520194)
        tuple_4 = ('Timely_Replacements', 'Timely_Fixes', 0.520194)
        tuple_5 = ('Area', 'TimeZone', 0.980153)
        tuple_6 = ('TimeZone', 'Area', 0.980153)

        # add tuple_1
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_1)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 1)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))

        # add tuple_1 again
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_1)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 1)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))

        # add tuple_2
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_2)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 1)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))

        # add tuple_3
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_3)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 2)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[1],
                                            ('Timely_Fixes', 'Timely_Replacements', 0.520194)))

        # add tuple_4
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_4)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 2)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[1],
                                            ('Timely_Fixes', 'Timely_Replacements', 0.520194)))

        # add tuple_5
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_5)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 3)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[1],
                                            ('Timely_Fixes', 'Timely_Replacements', 0.520194)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[2],
                                            ('Area', 'TimeZone', 0.980153)))

        # add tuple_6
        the_sg.add_to_storage(the_sg.relevant_corr_storage, tuple_6)

        # run assertions
        self.assertEqual(len(the_sg.relevant_corr_storage), 3)
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[0],
                                            ('StreamingMovies', 'MonthlyCharge', 0.608115)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[1],
                                            ('Timely_Fixes', 'Timely_Replacements', 0.520194)))
        self.assertTrue(are_tuples_the_same(the_sg.relevant_corr_storage[2],
                                            ('Area', 'TimeZone', 0.980153)))

    # test method for get_list_of_categorical_chi_sqrd_results()
    def test_get_list_of_categorical_chi_sqrd_results(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # run assertions
        self.assertEqual(len(the_sg.get_chi_squared_results()), 0)

        # invoke find_if_categorical_var_have_relationships()
        the_sg.find_chi_squared_results()

        # run assertions
        self.assertEqual(len(the_sg.get_chi_squared_results()), 171)
        self.assertIsInstance(the_sg.get_chi_squared_results(), list)

    # negative test method for get_list_of_variable_relationships_of_type
    def test_get_list_of_variable_relationships_of_type_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke find_if_categorical_var_have_relationships()
        the_sg.find_chi_squared_results()

        # make sure we handle None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.get_list_of_variable_relationships_of_type(None)

            # validate the error message.
            self.assertTrue("the_type is None or invalid type." in context.exception)

        # make sure we handle "foo"
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.get_list_of_variable_relationships_of_type("foo")

            # validate the error message.
            self.assertTrue("the_type is invalid." in context.exception)

    # test method for get_list_of_variable_relationships_of_type()
    def test_get_list_of_variable_relationships_of_type(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke find_if_categorical_var_have_relationships()
        the_sg.find_chi_squared_results()

        # get the different types
        valid_type_list = DATA_TYPE_KEYS

        # run assertions
        self.assertIsInstance(the_sg.get_chi_squared_results(), list)
        self.assertEqual(len(the_sg.get_chi_squared_results()), 171)
        self.assertEqual(valid_type_list[3], OBJECT_COLUMN_KEY)
        self.assertEqual(len(the_sg.get_list_of_variable_relationships_of_type(valid_type_list[3])), 15)

    # negative test method for filter_tuples_by_column()
    def test_filter_tuples_by_column_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # populate the correlations
        the_sg.find_correlations()

        # make sure we handle None, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.filter_tuples_by_column(None, None, None)

            # validate the error message.
            self.assertTrue("the_storage is None or invalid type." in context.exception)

        # make sure we handle the_sg.all_corr_storage, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.filter_tuples_by_column(the_sg.all_corr_storage, None, None)

            # validate the error message.
            self.assertTrue("the_column is not present on underlying dataframe." in context.exception)

        # make sure we handle the_sg.all_corr_storage, "foo", None
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.filter_tuples_by_column(the_sg.all_corr_storage, "foo", None)

            # validate the error message.
            self.assertTrue("the_column is not present on underlying dataframe." in context.exception)

        # make sure we handle the_sg.all_corr_storage, "Population", ['a', 'b', 'c']
        with self.assertRaises(AttributeError) as context:
            # invoke generate_hist()
            the_sg.filter_tuples_by_column(the_sg.all_corr_storage, "Population", ['a', 'b', 'c'])

            # validate the error message.
            self.assertTrue("original_list is not all of type tuple." in context.exception)

    # test method for filter_tuples_by_column()
    def test_filter_tuples_by_column(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # populate the correlations
        the_sg.find_correlations()

        # invoke the method
        the_list = the_sg.filter_tuples_by_column(the_sg.all_corr_storage, "Population")

        # run assertions to get all fields with Population.
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 32)
        self.assertEqual(the_list[0], ('Population', 'TimeZone', -0.063363))
        self.assertEqual(the_list[1], ('Population', 'Children', -0.005877))
        self.assertEqual(the_list[2], ('Population', 'Age', 0.010538))
        self.assertEqual(the_list[3], ('Population', 'Income', -0.008639))
        self.assertEqual(the_list[4], ('Population', 'Churn', -0.008533))
        self.assertEqual(the_list[5], ('Population', 'Outage_sec_perweek', 0.005483))
        self.assertEqual(the_list[6], ('Population', 'Email', 0.017962))
        self.assertEqual(the_list[7], ('Population', 'Contacts', 0.004019))
        self.assertEqual(the_list[8], ('Population', 'Yearly_equip_failure', -0.004483))
        self.assertEqual(the_list[9], ('Population', 'Techie', -0.011483))
        self.assertEqual(the_list[10], ('Population', 'Port_modem', 0.008577))
        self.assertEqual(the_list[11], ('Population', 'Tablet', 0.001225))
        self.assertEqual(the_list[12], ('Population', 'Phone', 0.008196))
        self.assertEqual(the_list[13], ('Population', 'Multiple', -0.001241))
        self.assertEqual(the_list[14], ('Population', 'OnlineSecurity', 0.012549))
        self.assertEqual(the_list[15], ('Population', 'OnlineBackup', 0.010352))
        self.assertEqual(the_list[16], ('Population', 'DeviceProtection', -0.003795))
        self.assertEqual(the_list[17], ('Population', 'TechSupport', -0.006606))
        self.assertEqual(the_list[18], ('Population', 'StreamingTV', -0.00659))
        self.assertEqual(the_list[19], ('Population', 'StreamingMovies', -0.005882))
        self.assertEqual(the_list[20], ('Population', 'PaperlessBilling', 0.008656))
        self.assertEqual(the_list[21], ('Population', 'Tenure', -0.003559))
        self.assertEqual(the_list[22], ('Population', 'MonthlyCharge', -0.004778))
        self.assertEqual(the_list[23], ('Population', 'Bandwidth_GB_Year', -0.003902))
        self.assertEqual(the_list[24], ('Population', 'Timely_Response', 0.000618))
        self.assertEqual(the_list[25], ('Population', 'Timely_Fixes', -0.002571))
        self.assertEqual(the_list[26], ('Population', 'Timely_Replacements', 0.00162))
        self.assertEqual(the_list[27], ('Population', 'Reliability', -0.008272))
        self.assertEqual(the_list[28], ('Population', 'Options', 0.00697))
        self.assertEqual(the_list[29], ('Population', 'Respectful_Response', 0.000834))
        self.assertEqual(the_list[30], ('Population', 'Courteous_Exchange', -0.013062))
        self.assertEqual(the_list[31], ('Population', 'Active_Listening', 0.008524))

        # retrieve the list of BOOLEAN columns on key BOOL_COLUMN_KEY
        exclusion_list = pa.analyzer.storage[BOOL_COLUMN_KEY]

        # invoke the method
        the_list = the_sg.filter_tuples_by_column(the_sg.all_corr_storage, "Population", exclusion_list)

        # run assertions to get all fields with Population.
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 19)

        # get the list of INT and FLOAT, make sure it matches the_list size
        numeric_list = pa.analyzer.storage[INT64_COLUMN_KEY] + pa.analyzer.storage[FLOAT64_COLUMN_KEY]

        # make sure the numeric_list has the expected size
        self.assertEqual(len(numeric_list), 20)

        # the numeric_list should be one larger since 'Population' is used for comparision.
        self.assertEqual(len(numeric_list) - 1, len(the_list))

        self.assertEqual(the_list[0], ('Population', 'TimeZone', -0.063363))
        self.assertEqual(the_list[1], ('Population', 'Children', -0.005877))
        self.assertEqual(the_list[2], ('Population', 'Age', 0.010538))
        self.assertEqual(the_list[3], ('Population', 'Income', -0.008639))
        self.assertEqual(the_list[4], ('Population', 'Outage_sec_perweek', 0.005483))
        self.assertEqual(the_list[5], ('Population', 'Email', 0.017962))
        self.assertEqual(the_list[6], ('Population', 'Contacts', 0.004019))
        self.assertEqual(the_list[7], ('Population', 'Yearly_equip_failure', -0.004483))
        self.assertEqual(the_list[8], ('Population', 'Tenure', -0.003559))
        self.assertEqual(the_list[9], ('Population', 'MonthlyCharge', -0.004778))
        self.assertEqual(the_list[10], ('Population', 'Bandwidth_GB_Year', -0.003902))
        self.assertEqual(the_list[11], ('Population', 'Timely_Response', 0.000618))
        self.assertEqual(the_list[12], ('Population', 'Timely_Fixes', -0.002571))
        self.assertEqual(the_list[13], ('Population', 'Timely_Replacements', 0.00162))
        self.assertEqual(the_list[14], ('Population', 'Reliability', -0.008272))
        self.assertEqual(the_list[15], ('Population', 'Options', 0.00697))
        self.assertEqual(the_list[16], ('Population', 'Respectful_Response', 0.000834))
        self.assertEqual(the_list[17], ('Population', 'Courteous_Exchange', -0.013062))
        self.assertEqual(the_list[18], ('Population', 'Active_Listening', 0.008524))

    # test method for find_chi_squared_results()
    def test_find_chi_squared_results(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke the method
        the_sg.find_chi_squared_results()

        # run assertions on overall number
        self.assertIsNotNone(the_sg.chi_square_results)
        self.assertIsInstance(the_sg.chi_square_results, list)
        self.assertEqual(len(the_sg.chi_square_results), 171)

        # get the actual list
        the_list = the_sg.get_chi_squared_results()

        # validate individual items
        self.assertEqual(the_list[0], ('Area', 'Marital', 0.503361))
        self.assertEqual(the_list[1], ('Area', 'Gender', 0.738468))
        self.assertEqual(the_list[2], ('Area', 'Churn', 0.295367))

        # make sure TimeZone isn't used for comparison
        time_zone_list = the_sg.filter_tuples_by_column(the_storage=the_sg.chi_square_results, the_column='TimeZone')

        # make sure the list is empty
        self.assertIsNotNone(time_zone_list)
        self.assertIsInstance(time_zone_list, list)
        self.assertEqual(len(time_zone_list), 0)

    # cross tab test, that the order does not matter.
    def test_cross_tab_assumptions(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # create a contingency table.
        contingency_1 = pd.crosstab(the_df['Churn'], the_df['Population'])
        contingency_2 = pd.crosstab(the_df['Population'], the_df['Churn'])

        # run the chi squared test
        c, p_value_1, dof, expected = chi2_contingency(contingency_1)

        # run the chi squared test again
        c, p_value_2, dof, expected = chi2_contingency(contingency_2)

        # if this test passes, the order of the cross tab does not matter in the p-value caluation.
        self.assertEqual(p_value_1, p_value_2)

    # test chi-squared test
    def test_chi_squared_test(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataframe
        the_df = pa.analyzer.the_df

        # define alpha
        alpha = 0.05

        # churn is boolean, population is integer
        contingency = pd.crosstab(the_df['Churn'], the_df['MonthlyCharge'])

        # run the chi squared test
        c, p_value, dof, expected = chi2_contingency(contingency)

        # calculate the critical statistic
        the_critical = chi2.ppf(alpha, dof)

        # run assertions
        self.assertGreaterEqual(abs(c), the_critical)
        self.assertEqual(the_critical, 686.4947164409323)
        self.assertEqual(p_value, 2.9956582683962194e-270)

    # test method for get_correlation_level()
    def test_get_correlation_level(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # tell the StatisticsGenerator to find all correlations with the_level argument
        the_sg.find_correlations(the_level=0.51)

        # run assertions
        self.assertEqual(the_sg.corr_level, 0.51)
        self.assertEqual(len(the_sg.relevant_corr_storage), 5)  # test to make sure the situation didn't change.

        # tell the StatisticsGenerator to find all correlations with default level
        the_sg.find_correlations()

        # run assertions
        self.assertEqual(the_sg.corr_level, 0.50)
        self.assertEqual(len(the_sg.relevant_corr_storage), 5)  # test to make sure the situation didn't change.

    # test if the method filter_tuples_by_column() returns all churn variables
    def test_filter_tuples_by_column_churn(self):
        # the purpose of this test is to verify that filter_tuples_by_column() can return all
        # the tuples associated with the variable, churn.

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # populate the correlations
        the_sg.find_correlations()

        # perform the chi-squared results
        the_sg.find_chi_squared_results()

        # run assertions
        self.assertIsNotNone(the_sg.chi_square_results)
        self.assertEqual(len(the_sg.chi_square_results), 171)

        # invoke the method
        the_list = the_sg.filter_tuples_by_column(the_sg.chi_square_results, "Churn")

        # run assertions
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 18)

        # test individual elements of the list to make sure it's full of what we expected
        self.assertEqual(the_list[0], ('Area', 'Churn', 0.295367))
        self.assertEqual(the_list[1], ('Marital', 'Churn', 0.234008))
        self.assertEqual(the_list[2], ('Gender', 'Churn', 0.019448))
        self.assertEqual(the_list[3], ('Churn', 'Techie', 0.0))
        self.assertEqual(the_list[4], ('Churn', 'Contract', 0.0))
        self.assertEqual(the_list[5], ('Churn', 'Port_modem', 0.427757))
        self.assertEqual(the_list[6], ('Churn', 'Tablet', 0.800168))
        self.assertEqual(the_list[7], ('Churn', 'InternetService', 0.0))
        self.assertEqual(the_list[8], ('Churn', 'Phone', 0.009578))
        self.assertEqual(the_list[9], ('Churn', 'Multiple', 0.0))
        self.assertEqual(the_list[10], ('Churn', 'OnlineSecurity', 0.183413))
        self.assertEqual(the_list[11], ('Churn', 'OnlineBackup', 0.0))
        self.assertEqual(the_list[12], ('Churn', 'DeviceProtection', 0.0))
        self.assertEqual(the_list[13], ('Churn', 'TechSupport', 0.062824))
        self.assertEqual(the_list[14], ('Churn', 'StreamingTV', 0.0))
        self.assertEqual(the_list[15], ('Churn', 'StreamingMovies', 0.0))
        self.assertEqual(the_list[16], ('Churn', 'PaperlessBilling', 0.496505))
        self.assertEqual(the_list[17], ('Churn', 'PaymentMethod', 0.024007))

    # test if the method filter_tuples_by_column() returns all churn variables, excluding INT and FLOAT
    def test_filter_tuples_by_column_churn_categorical_only(self):
        # the purpose of this test is to verify that filter_tuples_by_column() can return all
        # the tuples associated with the variable, churn.

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # populate the correlations
        the_sg.find_correlations()

        # get a reference to the data_analyzer
        the_da = pa.analyzer

        # create an exclusion list of FLOAT64_COLUMN_KEY and INT64_COLUMN_KEY
        exclusion_list = the_da.storage[FLOAT64_COLUMN_KEY] + the_da.storage[INT64_COLUMN_KEY]

        # run assertions to make sure what we have
        self.assertIsNotNone(exclusion_list)
        self.assertIsInstance(exclusion_list, list)

        # invoke the method
        the_list = the_sg.filter_tuples_by_column(the_storage=the_sg.all_corr_storage,
                                                  the_column="Churn",
                                                  exclusion_list=exclusion_list)

        # run assertions
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 12)

        self.assertEqual(the_list[0], ('Churn', 'Techie', 0.066722))
        self.assertEqual(the_list[1], ('Churn', 'Port_modem', 0.008157))
        self.assertEqual(the_list[2], ('Churn', 'Tablet', -0.002779))
        self.assertEqual(the_list[3], ('Churn', 'Phone', -0.026297))
        self.assertEqual(the_list[4], ('Churn', 'Multiple', 0.131771))
        self.assertEqual(the_list[5], ('Churn', 'OnlineSecurity', -0.01354))
        self.assertEqual(the_list[6], ('Churn', 'OnlineBackup', 0.050508))
        self.assertEqual(the_list[7], ('Churn', 'DeviceProtection', 0.056489))
        self.assertEqual(the_list[8], ('Churn', 'TechSupport', 0.018838))
        self.assertEqual(the_list[9], ('Churn', 'StreamingTV', 0.230151))
        self.assertEqual(the_list[10], ('Churn', 'StreamingMovies', 0.289262))
        self.assertEqual(the_list[11], ('Churn', 'PaperlessBilling', 0.00703))

    # test method to see if filter_tuples_by_column() works correctly with TimeZone as an argument
    def test_filter_tuples_by_column_timezone_verification(self):
        # the purpose of this test is to verify that filter_tuples_by_column() will return an
        # empty list when timezone is passed as an argument on the chi-squared results storage.
        # Since TimeZone is converted to a integer, it should not show up in the chi-squared results.

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # populate the correlations
        the_sg.find_correlations()

        # make sure TimeZone isn't used for comparison
        time_zone_list = the_sg.filter_tuples_by_column(the_storage=the_sg.chi_square_results, the_column='TimeZone')

        # make sure the list is empty
        self.assertIsNotNone(time_zone_list)
        self.assertIsInstance(time_zone_list, list)
        self.assertEqual(len(time_zone_list), 0)

    # test method for removing elements of one list from another list
    def test_list_removal(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # create a list of all columns in the dataset
        all_columns = the_sg.the_dsa.storage[COLUMN_KEY]

        # get the list of int64
        int_list = the_sg.the_dsa.storage[INT64_COLUMN_KEY]

        # get the list of float
        float_list = the_sg.the_dsa.storage[FLOAT64_COLUMN_KEY]

        # get the list of objects
        object_list = the_sg.the_dsa.storage[OBJECT_COLUMN_KEY]

        # get the list of booleans
        boolean_list = the_sg.the_dsa.storage[BOOL_COLUMN_KEY]

        # run assertions
        self.assertEqual(len(all_columns), 39)
        self.assertEqual(len(int_list), 15)
        self.assertEqual(len(float_list), 5)
        self.assertEqual(len(object_list), 6)
        self.assertEqual(len(boolean_list), 13)

        # validate the number of numeric columns
        self.assertEqual(len(int_list) + len(float_list), 15 + 5)

        # validate the number of categorical columns
        self.assertEqual(len(object_list) + len(boolean_list), 6 + 13)

        # remove int_list from all_columns
        all_columns = [i for i in all_columns if i not in int_list]

        # run assertions 39 - 15 = 24
        self.assertEqual(len(all_columns), 24)
        self.assertEqual(len(all_columns), len(float_list) + len(object_list) + len(boolean_list))

        # remove float_list from all_columns
        all_columns = [i for i in all_columns if i not in float_list]

        # run assertions 24 - 5 = 19
        self.assertEqual(len(all_columns), 19)
        self.assertEqual(len(all_columns), len(object_list) + len(boolean_list))

        # ******************************************************************************
        # *                         STATE CHECK
        # ******************************************************************************

        # This test is to make sure we didn't change the state, and repeating the test gives
        # the same results.

        # create a list of all columns in the dataset
        all_columns = the_sg.the_dsa.storage[COLUMN_KEY]

        # run assertions
        self.assertEqual(len(all_columns), 39)

        all_columns = [i for i in all_columns if i not in int_list + float_list]

        # run assertions
        self.assertEqual(len(all_columns), 19)
        self.assertEqual(len(all_columns), len(object_list) + len(boolean_list))

    # negative tests for fit_theoretical_distribution()
    def test_fit_theoretical_distribution_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # make sure we handle None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_sg.fit_theoretical_distribution(None)

            # validate the error message.
            self.assertTrue("the_column is None or incorrect type." in context.exception)

        # make sure we handle "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            the_sg.fit_theoretical_distribution("foo")

            # validate the error message.
            self.assertTrue("the_column is not present in the underlying dataframe." in context.exception)

    # test method for fit_theoretical_distribution()
    def test_fit_theoretical_distribution(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke the method
        the_sg.fit_theoretical_distribution("Age")

        # run assertions
        self.assertIsNotNone(the_sg.distribution)
        self.assertIsInstance(the_sg.distribution, dict)
        self.assertEqual(the_sg.distribution["Age"][DIST_NAME], "beta")
        self.assertEqual(the_sg.distribution["Age"][DIST_PARAMETERS], (0.9507834161575333, 1.0049760804615488,
                                                                       17.999999999999996, 71.01646413274422))

        # invoke the method for Population.  There could be an issue here.
        the_sg.fit_theoretical_distribution("Population")

        # run assertions
        self.assertIsNotNone(the_sg.distribution)
        self.assertIsInstance(the_sg.distribution, dict)
        self.assertEqual(the_sg.distribution["Population"][DIST_NAME], "pareto")
        self.assertEqual(the_sg.distribution["Population"][DIST_PARAMETERS], (0.9643800982631355, -2732.1698244360414,
                                                                              2732.169824436041))

    # test method for fit_theoretical_dist_to_all_columns()
    def test_fit_theoretical_dist_to_all_columns(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.DATASET_KEY_NAME)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # verify that the storage is empty
        self.assertIsNotNone(the_sg.distribution)
        self.assertTrue(len(the_sg.distribution) == 0)

        # invoke the method
        the_sg.fit_theoretical_dist_to_all_columns()

        # verify that the distribution storage is populated
        self.assertIsNotNone(the_sg.distribution)
        self.assertEqual(len(the_sg.distribution), 20)

        # assert each output
        self.assertEqual(the_sg.distribution['Population'][DIST_NAME], "pareto")
        self.assertEqual(the_sg.distribution['Population'][DIST_PARAMETERS], (0.9643800982631355, -2732.1698244360414,
                                                                              2732.169824436041))

        self.assertEqual(the_sg.distribution['Children'][DIST_NAME], "beta")
        self.assertEqual(the_sg.distribution['Children'][DIST_PARAMETERS], (0.4004431584458652, 7.454490526646154,
                                                                            -6.818376297739706e-33, 24.248274350068083))

        self.assertEqual(the_sg.distribution['Age'][DIST_NAME], "beta")
        self.assertEqual(the_sg.distribution['Age'][DIST_PARAMETERS], (0.9507834161575333, 1.0049760804615488,
                                                                       17.999999999999996, 71.01646413274422))

        self.assertEqual(the_sg.distribution['Email'][DIST_NAME], "loggamma")
        self.assertEqual(the_sg.distribution['Email'][DIST_PARAMETERS], (1000.7522468858326, -650.0341887964216,
                                                                         95.8375585745257))

        self.assertEqual(the_sg.distribution['Contacts'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Contacts'][DIST_PARAMETERS], (0.24689230131132867, 1.0000000000000002,
                                                                            1.2866235066933096))

        self.assertEqual(the_sg.distribution['Yearly_equip_failure'][DIST_NAME], "beta")
        self.assertEqual(the_sg.distribution['Yearly_equip_failure'][DIST_PARAMETERS],
                         (0.18170468766406694, 13.095017383848838, -7.325281409460864e-31, 7.654543703366844))

        self.assertEqual(the_sg.distribution['Timely_Response'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Timely_Response'][DIST_PARAMETERS],
                         (1.6709488067693665, 3.498763407139511, 0.9798653823881436))

        self.assertEqual(the_sg.distribution['Timely_Fixes'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Timely_Fixes'][DIST_PARAMETERS], (1.6726288065642265, 3.5018375613637427,
                                                                                0.9772336581739072))

        self.assertEqual(the_sg.distribution['Timely_Replacements'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Timely_Replacements'][DIST_PARAMETERS],
                         (1.6795100050866902, 3.4955728452008836, 0.9724994465320714))

        self.assertEqual(the_sg.distribution['Reliability'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Reliability'][DIST_PARAMETERS], (1.6756190733103455, 3.4984202122918293,
                                                                               0.9694076399035925))

        self.assertEqual(the_sg.distribution['Options'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Options'][DIST_PARAMETERS], (1.6774527505079329, 3.498369633379326,
                                                                           0.9688858462749022))

        self.assertEqual(the_sg.distribution['Respectful_Response'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Respectful_Response'][DIST_PARAMETERS],
                         (1.6866779247450983, 3.50188697627247, 0.9796927394550141))

        self.assertEqual(the_sg.distribution['Courteous_Exchange'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Courteous_Exchange'][DIST_PARAMETERS],
                         (1.6653880163234542, 3.5031844202996183, 0.9694408111960751))

        self.assertEqual(the_sg.distribution['Active_Listening'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Active_Listening'][DIST_PARAMETERS],
                         (1.6740646901799774, 3.4997488902389504, 0.9717044746454877))

        self.assertEqual(the_sg.distribution['Income'][DIST_NAME], "lognorm")
        self.assertEqual(the_sg.distribution['Income'][DIST_PARAMETERS],
                         (0.5829428226812885, -7466.536462025313, 40063.24788277939))

        self.assertEqual(the_sg.distribution['Outage_sec_perweek'][DIST_NAME], "t")
        self.assertEqual(the_sg.distribution['Outage_sec_perweek'][DIST_PARAMETERS],
                         (892.5184217792862, 10.001831564796777, 2.972534635647201))

        self.assertEqual(the_sg.distribution['Tenure'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Tenure'][DIST_PARAMETERS],
                         (4.118497884733575, 34.385843976127305, 27.959013507658874))

        self.assertEqual(the_sg.distribution['MonthlyCharge'][DIST_NAME], "gamma")
        self.assertEqual(the_sg.distribution['MonthlyCharge'][DIST_PARAMETERS],
                         (11.000995358223932, 28.826759347430404, 13.071368128115909))

        self.assertEqual(the_sg.distribution['Bandwidth_GB_Year'][DIST_NAME], "dweibull")
        self.assertEqual(the_sg.distribution['Bandwidth_GB_Year'][DIST_PARAMETERS],
                         (3.5522705023270875, 3399.1078881640897, 2303.2822381176084))
