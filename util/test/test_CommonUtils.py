import math
import unittest

import pandas as pd
from pandas import DataFrame

from model.Project_Assessment import Project_Assessment
from model.analysis.StatisticsGenerator import StatisticsGenerator
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_209_CHURN, D_209_MEDICAL
from model.constants.DatasetConstants import BOOL_COLUMN_KEY
from model.constants.StatisticsConstants import ALL_CORRELATIONS
from util.CommonUtils import CommonUtils, strip_preceding_letter_from_field, strip_trailing_digit_from_field, \
    are_list_elements_hashable, are_tuples_the_same, remove_item_from_list_by_type, \
    get_tuples_from_list_with_specific_field, remove_tuples_from_list, convert_dict_to_str, convert_series_to_dataframe


# test class for CommonUtils
class test_CommonUtils(unittest.TestCase):
    # constants
    STRING_NAN = "nan"
    ACTUAL_NAN = math.nan
    BAD_VALUE = "blue"

    # constants
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    params_dict = {'algorithm': 'auto', 'leaf_size': 2, 'metric': 'minkowski', 'metric_params': None,
                   'n_jobs': None, 'n_neighbors': 4, 'p': 1, 'weights': 'uniform'}

    params_dict_str = ('algorithm: auto, leaf_size: 2, metric: minkowski, metric_params: None, n_jobs: '
                       'None, n_neighbors: 4, p: 1, weights: uniform')

    CHURN_KEY = D_209_CHURN

    # test the init method
    def test_init(self):
        # instantiate a CommonUtils
        util = CommonUtils()

        self.assertIsNotNone(util)
        self.assertIsInstance(util, CommonUtils)

    # test scenarios
    def test_isnan(self):
        # run assertions
        self.assertTrue(CommonUtils.isnan(self.STRING_NAN))
        self.assertTrue(CommonUtils.isnan(self.ACTUAL_NAN))
        self.assertFalse(CommonUtils.isnan(self.BAD_VALUE))

    # test the offset_cell() method.
    def test_strip_trailing_digit_from_field_bad_arguments(self):
        # make sure we throw an SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke strip_trailing_digit_from_field() with None
            strip_trailing_digit_from_field(None)

            # validate the error message.
            self.assertTrue("Argument is None or empty.." in context.exception)

        # make sure we throw an SyntaxError for integer only
        with self.assertRaises(SyntaxError) as context:
            # invoke strip_trailing_digit_from_field() with integer only
            strip_trailing_digit_from_field(1)

            # validate the error message.
            self.assertTrue("Argument is None or empty.." in context.exception)

        # make sure we throw an SyntaxError for empty string
        with self.assertRaises(SyntaxError) as context:
            # invoke strip_trailing_digit_from_field() with integer only
            strip_trailing_digit_from_field("")

            # validate the error message.
            self.assertTrue("Argument is None or empty.." in context.exception)

        # try only digits passed as a string
        the_result = strip_trailing_digit_from_field("1234567")

        # run assertions
        self.assertEqual(the_result, 1234567)

        # try only letter values
        the_result = strip_trailing_digit_from_field("ABCD")
        self.assertIsNone(the_result)

    # test the happy path for strip_preceding_letter_from_field()
    def test_strip_trailing_digit_from_field(self):
        # simple test A5
        the_result = strip_trailing_digit_from_field("A5")
        self.assertEqual(the_result, 5)

        # simple test A125
        the_result = strip_trailing_digit_from_field("A125")
        self.assertEqual(the_result, 125)

        # test 'REGION CALC 9' without the quotes
        the_result = strip_trailing_digit_from_field("REGION CALC 9")
        self.assertEqual(the_result, 9)

        # simple test CD125
        the_result = strip_trailing_digit_from_field("CD125")
        self.assertEqual(the_result, 125)

    # test the strip_preceding_letter_from_field() method.
    def test_strip_preceding_letter_from_field_bad_arguments(self):
        # make sure we throw an SyntaxError for None
        with self.assertRaises(SyntaxError) as context:
            # invoke strip_trailing_digit_from_field() with None
            strip_preceding_letter_from_field(None)

            # validate the error message.
            self.assertTrue("Argument is None or empty." in context.exception)

        # make sure we throw an SyntaxError for integer only
        with self.assertRaises(SyntaxError) as context:
            # invoke strip_trailing_digit_from_field() with integer only
            strip_preceding_letter_from_field(1)

            # validate the error message.
            self.assertTrue("Argument is None or empty." in context.exception)

        # make sure we throw an SyntaxError for empty string
        with self.assertRaises(SyntaxError) as context:
            # invoke strip_trailing_digit_from_field() with integer only
            strip_preceding_letter_from_field("")

            # validate the error message.
            self.assertTrue("Argument is None or empty." in context.exception)

    # test the happy path for strip_preceding_letter_from_field()
    def test_strip_preceding_letter_from_field(self):
        # simple test A5
        the_result = strip_preceding_letter_from_field("A5")
        self.assertEqual(the_result, "A")

        # complicated test ABCDEFG5
        the_result = strip_preceding_letter_from_field("ABCDEFG5")
        self.assertEqual(the_result, "ABCDEFG")

        # complicated test A51476
        the_result = strip_preceding_letter_from_field("A51476")
        self.assertEqual(the_result, "A")

        # complicated test "REGION CALC 9"
        the_result = strip_preceding_letter_from_field("REGION CALC 9")
        self.assertEqual(the_result, "REGION CALC")

        # complicated test "REGION CALC 10"
        the_result = strip_preceding_letter_from_field("REGION CALC 10")
        self.assertEqual(the_result, "REGION CALC")

    # negative test for are_list_elements_hashable() method
    def test_are_list_elements_hashable_negative(self):
        # make sure we throw an AttributeError for None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            are_list_elements_hashable(None)

            # validate the error message.
            self.assertTrue("the_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for foo"
        with self.assertRaises(AttributeError) as context:
            # invoke method
            are_list_elements_hashable("foo")

            # validate the error message.
            self.assertTrue("the_list is None or incorrect type." in context.exception)

    # tests for are_list_elements_hashable() method
    def test_are_list_elements_hashable(self):
        # run assertions
        self.assertTrue(are_list_elements_hashable([('Email', 'Outage_sec_perweek', 0.608125)]))
        self.assertTrue(are_list_elements_hashable([1, 2, 3]))
        self.assertTrue(are_list_elements_hashable(['1', '2', '3']))
        self.assertTrue(are_list_elements_hashable(['red', 'blue', 'green']))
        self.assertTrue(are_list_elements_hashable([None]))

    # tests for are_tuples_the_same() method
    def test_are_tuples_the_same(self):
        # run assertions
        self.assertFalse(are_tuples_the_same((1, 2, 3), (1, 2)))
        self.assertFalse(are_tuples_the_same((1, 2), (1, 2, 3)))
        self.assertFalse(are_tuples_the_same(('1', '2'), (1, 2)))
        self.assertTrue(are_tuples_the_same((1, 2), (1, 2)))
        self.assertTrue(are_tuples_the_same((1, 2), (2, 1)))
        self.assertTrue(are_tuples_the_same(('Outage_sec_perweek', 'Email'), ('Email', 'Outage_sec_perweek')))
        self.assertTrue(are_tuples_the_same(('Outage_sec_perweek', 'Email', 0.608125),
                                            ('Email', 'Outage_sec_perweek', 0.608125)))
        self.assertTrue(are_tuples_the_same(('Area', 'TimeZone', 0.980153), ('TimeZone', 'Area', 0.980153)))

        data = [['rob', 10], ['andree', 15], ['garrett', 19]]

        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns=['Name', 'Age'])

        self.assertTrue(are_tuples_the_same(('Area', 'TimeZone', df), ('TimeZone', 'Area', df)))

    # negative tests for remove_item_from_list()
    def test_remove_item_from_list_negative(self):
        # make sure we throw an AttributeError for None, None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            remove_item_from_list_by_type(None, None)

            # validate the error message.
            self.assertTrue("the_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for "foo", None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            remove_item_from_list_by_type("foo", None)

            # validate the error message.
            self.assertTrue("the_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for [], None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            remove_item_from_list_by_type([], None)

            # validate the error message.
            self.assertTrue("the_type is None." in context.exception)

    # test method for remove_item_from_list()
    def test_remove_item_from_list(self):
        data = [['rob', 10], ['andree', 15], ['garrett', 19]]

        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns=['Name', 'Age'])

        # run assertions
        self.assertIsNotNone(remove_item_from_list_by_type(["a", "b", 1], int))
        self.assertIsInstance(remove_item_from_list_by_type(["a", "b", 1], int), list)
        self.assertEqual(len(remove_item_from_list_by_type(["a", "b", 1], int)), 2)
        self.assertEqual(remove_item_from_list_by_type(["a", "b", 1], int)[0], "a")
        self.assertEqual(remove_item_from_list_by_type(["a", "b", 1], int)[1], "b")
        self.assertIsInstance(remove_item_from_list_by_type(["a", "b", df], DataFrame), list)
        self.assertEqual(remove_item_from_list_by_type(["a", 1, "b"], int)[0], "a")
        self.assertEqual(remove_item_from_list_by_type(["a", 3, "b"], int)[1], "b")
        self.assertEqual(remove_item_from_list_by_type([1, "a", "b"], int)[0], "a")
        self.assertEqual(remove_item_from_list_by_type([3, "a", "b"], int)[1], "b")

    # negative test method for get_tuples_from_list_with_specific_field()
    def test_get_tuples_from_list_with_specific_field_negative(self):
        # make sure we throw an AttributeError for None, None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            get_tuples_from_list_with_specific_field(None, None)

            # validate the error message.
            self.assertTrue("the_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for "foo", None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            get_tuples_from_list_with_specific_field("foo", None)

            # validate the error message.
            self.assertTrue("the_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for [], None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            get_tuples_from_list_with_specific_field([], None)

            # validate the error message.
            self.assertTrue("the_type is None." in context.exception)

        # create a list of tuples
        tuple_1 = ("a", "b", "c")
        tuple_2 = ("d", "e", "f")
        tuple_3 = ("a", "e", "q")

        tuple_list = [tuple_1, tuple_2, tuple_3, "a", "b"]

        # make sure we throw an AttributeError for tuple_list, None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            get_tuples_from_list_with_specific_field(tuple_list, "a")

            # validate the error message.
            self.assertTrue("the_list is not all of type tuple." in context.exception)

    # test method for get_tuples_from_list_with_specific_field()
    def test_get_tuples_from_list_with_specific_field(self):
        # create a list of tuples
        tuple_1 = ("a", "b", "c")
        tuple_2 = ("d", "e", "f")
        tuple_3 = ("a", "e", "q")
        tuple_4 = ("d", "a", "q")
        tuple_5 = ("z", "a")
        tuple_6 = ("j", "k", "l", "n", "a")

        tuple_list_1 = [tuple_1, tuple_2, tuple_3]
        tuple_list_2 = [tuple_1, tuple_2, tuple_3, tuple_4]
        tuple_list_3 = [tuple_1, tuple_2, tuple_3, tuple_4, tuple_5]
        tuple_list_4 = [tuple_1, tuple_2, tuple_3, tuple_4, tuple_5, tuple_6]

        # run assertions
        self.assertIsNotNone(get_tuples_from_list_with_specific_field([tuple_1], "a"))
        self.assertEqual(len(get_tuples_from_list_with_specific_field([tuple_1], "a")), 1)
        self.assertIsNotNone(get_tuples_from_list_with_specific_field([tuple_1], "d"))
        self.assertEqual(len(get_tuples_from_list_with_specific_field([tuple_1], "d")), 0)
        self.assertIsNotNone(get_tuples_from_list_with_specific_field(tuple_list_1, "a"))
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_1, "a")), 2)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_2, "a")), 3)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_2, "q")), 2)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_2, "c")), 1)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_3, "a")), 4)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_3, "e")), 2)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_3, "o")), 0)
        self.assertEqual(len(get_tuples_from_list_with_specific_field(tuple_list_4, "a")), 5)

        # now for a real world test.
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke find_correlations()
        the_sg.find_correlations()

        # get the list of all correlations.  By definition, this does not include OBJECT.
        list_of_tuples = the_sg.get_list_of_correlations(the_type=ALL_CORRELATIONS)

        # retrieve the list of BOOLEAN columns on key BOOL_COLUMN_KEY
        exclude_list = pa.analyzer.storage[BOOL_COLUMN_KEY]

        # make a copy of the list_of_tuples
        the_result = list_of_tuples.copy()

        # loop over the exclude list
        for element in exclude_list:
            # invoke the method
            tuples_to_be_removed_list = get_tuples_from_list_with_specific_field(list_of_tuples, element)

            # remove those from the_result
            for next_tuple in tuples_to_be_removed_list:
                # check if next_tuple is in the_result
                if next_tuple in the_result:
                    # remove it if it is
                    the_result.remove(next_tuple)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, list)
        self.assertNotEqual(len(the_result), len(list_of_tuples))
        self.assertEqual(len(the_result), 190)
        self.assertEqual(the_result[0], ('Population', 'TimeZone', -0.063363))
        self.assertEqual(the_result[1], ('Population', 'Children', -0.005877))
        self.assertEqual(the_result[2], ('Population', 'Age', 0.010538))
        self.assertEqual(the_result[3], ('Population', 'Income', -0.008639))
        self.assertEqual(the_result[4], ('Population', 'Outage_sec_perweek', 0.005483))
        self.assertEqual(the_result[5], ('Population', 'Email', 0.017962))
        self.assertEqual(the_result[6], ('Population', 'Contacts', 0.004019))
        self.assertEqual(the_result[7], ('Population', 'Yearly_equip_failure', -0.004483))
        self.assertEqual(the_result[8], ('Population', 'Tenure', -0.003559))
        self.assertEqual(the_result[9], ('Population', 'MonthlyCharge', -0.004778))
        self.assertEqual(the_result[10], ('Population', 'Bandwidth_GB_Year', -0.003902))
        self.assertEqual(the_result[11], ('Population', 'Timely_Response', 0.000618))
        self.assertEqual(the_result[12], ('Population', 'Timely_Fixes', -0.002571))
        self.assertEqual(the_result[13], ('Population', 'Timely_Replacements', 0.00162))
        self.assertEqual(the_result[14], ('Population', 'Reliability', -0.008272))
        self.assertEqual(the_result[15], ('Population', 'Options', 0.00697))
        self.assertEqual(the_result[16], ('Population', 'Respectful_Response', 0.000834))
        self.assertEqual(the_result[17], ('Population', 'Courteous_Exchange', -0.013062))
        self.assertEqual(the_result[18], ('Population', 'Active_Listening', 0.008524))
        self.assertEqual(the_result[19], ('TimeZone', 'Children', 0.011559))

    # negative test method for remove_tuples_from_list()
    def test_remove_tuples_from_list_negative(self):
        # make sure we throw an AttributeError for None, None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            remove_tuples_from_list(None, None)

            # validate the error message.
            self.assertTrue("original_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for [], None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            remove_tuples_from_list([], None)

            # validate the error message.
            self.assertTrue("exclusion_list is None or incorrect type." in context.exception)

        # make sure we throw an AttributeError for ["a", "b"], []
        with self.assertRaises(AttributeError) as context:
            # invoke method
            remove_tuples_from_list(["a", "b"], [])

            # validate the error message.
            self.assertTrue("original_list is not all of type tuple." in context.exception)

    # test method for remove_tuples_from_list()
    def test_remove_tuples_from_list(self):
        # define tuples
        tuple_1 = ("a", "b")
        tuple_2 = ("c", "d")
        tuple_3 = ("e", "f")
        tuple_4 = ("g", "h")
        tuple_5 = ("i", "k")
        tuple_6 = ("k", "l")
        tuple_7 = ("i", "k", "l")

        original_list_1 = [tuple_1, tuple_2]
        original_list_2 = [tuple_1, tuple_2, tuple_3]
        original_list_3 = [tuple_1, tuple_2, tuple_3, tuple_4]
        original_list_4 = [tuple_1, tuple_2, tuple_3, tuple_4, tuple_5]
        original_list_5 = [tuple_1, tuple_2, tuple_5, tuple_4, tuple_3]
        original_list_6 = [tuple_1, tuple_2, tuple_3, tuple_4, tuple_5, tuple_6]
        original_list_7 = [tuple_1, tuple_2, tuple_3, tuple_7, tuple_5, tuple_6]
        exclusion_list_1 = ["g", "h"]
        exclusion_list_2 = ["i", "j", "k", "l"]
        exclusion_list_3 = ["k", "j", "i", "l"]

        # run assertions
        self.assertIsNotNone(remove_tuples_from_list(original_list_1, exclusion_list_1))
        self.assertIsInstance(remove_tuples_from_list(original_list_1, exclusion_list_1), list)
        self.assertEqual(len(remove_tuples_from_list(original_list_1, exclusion_list_1)), 2)
        self.assertEqual(len(remove_tuples_from_list(original_list_2, exclusion_list_1)), 3)

        self.assertEqual(len(remove_tuples_from_list(original_list_3, exclusion_list_1)), 3)
        self.assertEqual(remove_tuples_from_list(original_list_3, exclusion_list_1)[0], tuple_1)
        self.assertEqual(remove_tuples_from_list(original_list_3, exclusion_list_1)[1], tuple_2)
        self.assertEqual(remove_tuples_from_list(original_list_3, exclusion_list_1)[2], tuple_3)

        self.assertEqual(len(remove_tuples_from_list(original_list_4, exclusion_list_1)), 4)
        self.assertEqual(remove_tuples_from_list(original_list_4, exclusion_list_1)[0], tuple_1)
        self.assertEqual(remove_tuples_from_list(original_list_4, exclusion_list_1)[1], tuple_2)
        self.assertEqual(remove_tuples_from_list(original_list_4, exclusion_list_1)[2], tuple_3)
        self.assertEqual(remove_tuples_from_list(original_list_4, exclusion_list_1)[3], tuple_5)

        self.assertEqual(len(remove_tuples_from_list(original_list_5, exclusion_list_1)), 4)
        self.assertEqual(remove_tuples_from_list(original_list_5, exclusion_list_1)[0], tuple_1)
        self.assertEqual(remove_tuples_from_list(original_list_5, exclusion_list_1)[1], tuple_2)
        self.assertEqual(remove_tuples_from_list(original_list_5, exclusion_list_1)[2], tuple_5)
        self.assertEqual(remove_tuples_from_list(original_list_5, exclusion_list_1)[3], tuple_3)

        self.assertEqual(len(remove_tuples_from_list(original_list_6, exclusion_list_2)), 4)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_2)[0], tuple_1)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_2)[1], tuple_2)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_2)[2], tuple_3)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_2)[3], tuple_4)

        self.assertEqual(len(remove_tuples_from_list(original_list_6, exclusion_list_3)), 4)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_3)[0], tuple_1)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_3)[1], tuple_2)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_3)[2], tuple_3)
        self.assertEqual(remove_tuples_from_list(original_list_6, exclusion_list_3)[3], tuple_4)

        self.assertEqual(len(remove_tuples_from_list(original_list_7, exclusion_list_3)), 3)
        self.assertEqual(remove_tuples_from_list(original_list_7, exclusion_list_3)[0], tuple_1)
        self.assertEqual(remove_tuples_from_list(original_list_7, exclusion_list_3)[1], tuple_2)
        self.assertEqual(remove_tuples_from_list(original_list_7, exclusion_list_3)[2], tuple_3)

        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # create a StatisticsGenerator
        the_sg = StatisticsGenerator(pa.analyzer)

        # invoke find_correlations()
        the_sg.find_correlations()

        # get the list of all correlations.  By definition, this does not include OBJECT.
        list_of_tuples = the_sg.get_list_of_correlations(the_type=ALL_CORRELATIONS)

        # we ultimately want to exclude the list of boolean columns
        exclude_list = pa.analyzer.storage[BOOL_COLUMN_KEY]

        # invoke the method
        the_list = remove_tuples_from_list(list_of_tuples, exclude_list)

        # run assertions
        self.assertIsNotNone(the_list)
        self.assertIsInstance(the_list, list)
        self.assertEqual(len(the_list), 528)
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
        self.assertEqual(the_list[32], ('TimeZone', 'Children', 0.011559))
        self.assertEqual(the_list[33], ('TimeZone', 'Age', 0.004809))
        self.assertEqual(the_list[34], ('TimeZone', 'Income', -0.001933))
        self.assertEqual(the_list[35], ('TimeZone', 'Churn', -0.002378))

    # negative test method for convert_dict_to_str()
    def test_convert_dict_to_str_negative(self):
        # make sure we throw an AttributeError for None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            convert_dict_to_str(the_dict=None)

        # validate the error message.
        self.assertTrue("the_dict is None or incorrect type." in context.exception.args)

    # test method for convert_dict_to_str()
    def test_convert_dict_to_str(self):
        # invoke the method
        self.assertEqual(convert_dict_to_str(the_dict=self.params_dict), self.params_dict_str)

    # negative test method for convert_series_to_dataframe()
    def test_convert_series_to_dataframe_negative(self):
        # make sure we throw an AttributeError for None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            convert_series_to_dataframe(the_series=None)

        # validate the error message.
        self.assertTrue("the_series is None or incorrect type." in context.exception.args)

    # test method for convert_series_to_dataframe()
    def test_convert_series_to_dataframe(self):
        # test 1: verify a series is converted to a DataFrame.

        # create the series
        s = pd.Series(["a", "b", "c"], name="vals")

        # invoke the method
        the_result = convert_series_to_dataframe(the_series=s)

        # run assertions
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, DataFrame)

        # test 2: verify a DataFrame passes through the method the same
        # make a copy of the original dataframe from test 1
        the_copied_result = the_result.copy()

        # run assertions prior to invoking method
        self.assertIsInstance(the_copied_result, DataFrame)
        self.assertTrue('vals' in the_copied_result.columns.to_list())
        self.assertEqual(the_copied_result.shape, (3, 1))
        self.assertEqual(the_copied_result['vals'][0], the_result['vals'][0])
        self.assertEqual(the_copied_result['vals'][1], the_result['vals'][1])
        self.assertEqual(the_copied_result['vals'][2], the_result['vals'][2])

        # invoke the method
        the_df = convert_series_to_dataframe(the_series=the_copied_result)

        # run assertions after to invoking method
        self.assertIsInstance(the_copied_result, DataFrame)
        self.assertTrue('vals' in the_copied_result.columns.to_list())
        self.assertEqual(the_copied_result.shape, (3, 1))
        self.assertEqual(the_copied_result['vals'][0], the_result['vals'][0])
        self.assertEqual(the_copied_result['vals'][1], the_result['vals'][1])
        self.assertEqual(the_copied_result['vals'][2], the_result['vals'][2])

