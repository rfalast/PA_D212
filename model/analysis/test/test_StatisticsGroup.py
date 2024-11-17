import unittest
import numpy as np

from pandas import Series
from model.analysis.StatisticsGroup import StatisticGroup
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from util.CSV_loader import CSV_Loader
from model.constants.StatisticsConstants import NAME, MEAN, MEDIAN, VARIANCE, STD_DEV, COUNT, SKEW, KURTOSIS, \
    STAT_OPTIONS, MEAN_ABS_DEV, NORMALITY_TEST, NORMALITY_TEST_NOT_NORMAL, \
    NORMALITY_TEST_P_VALUE, NORMALITY_TEST_METHOD, EXPONENTIAL_TEST, \
    EXPONENTIAL_TEST_P_VALUE, EXPONENTIAL_TEST_METHOD, KOLMOGOROV_SMIRNOV_TEST, EXPONENTIAL_TEST_NOT_EXP


# test classes for StatisticsGroup
class test_StatisticsGroup(unittest.TestCase):
    # constants
    VALID_CSV_PATH = "../../../resources/Input/churn_raw_data.csv"
    VALID_MESSAGE = "MEAN:[2.0877], MEDIAN:[1.0], STD_DEV:[2.1472004463896], MEAN_ABS_DEV:[2.0877], " \
                    "VARIANCE:[4.610469756975697], SKEW:[1.4426454055922135], KURTOSIS:[2.12194534968734], " \
                    "NAME:[Children], COUNT:[10000], NORMALITY_TEST:[NOT NORMAL], NORMALITY_TEST_P_VALUE:[0.0], " \
                    "NORMALITY_TEST_METHOD:[Kolmogorov-Smirnov]"

    # negative test for init() method
    def test_init_negative(self):
        # verify that we receive AttributeError from None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            the_sg = StatisticGroup(None)

            # validate the error message.
            self.assertTrue("the_series was None or incorrect type." in context.exception)

        # verify that we receive AttributeError from list
        with self.assertRaises(AttributeError) as context:
            # invoke method
            the_sg = StatisticGroup(list)

            # validate the error message.
            self.assertTrue("the_series was None or incorrect type." in context.exception)

    # test method for init()
    def test_init(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run complete setup
        dsa.run_complete_setup()

        # get the dataframe from the dsa
        the_df = dsa.the_df

        # instantiate the SG
        the_sg = StatisticGroup(the_df['Children'])

        # run assertions
        self.assertIsNotNone(the_sg.the_series)
        self.assertIsNotNone(the_sg.storage)
        self.assertIsInstance(the_sg.the_series, Series)
        self.assertIsInstance(the_sg.storage, dict)

    # negative test method for get_statistic()
    def test_get_statistic_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run complete setup
        dsa.run_complete_setup()

        # get the dataframe from the dsa
        the_df = dsa.the_df

        # instantiate the SG
        the_sg = StatisticGroup(the_df['Children'])

        # verify that we receive AttributeError from None
        with self.assertRaises(AttributeError) as context:
            # invoke method
            the_sg.get_statistic(None)

            # validate the error message.
            self.assertTrue("the_name was None or incorrect type." in context.exception)

        # verify that we receive AttributeError from list
        with self.assertRaises(AttributeError) as context:
            # invoke method
            the_sg.get_statistic(list)

            # validate the error message.
            self.assertTrue("the_name was None or incorrect type." in context.exception)

        # verify that we receive AttributeError from foo
        with self.assertRaises(AttributeError) as context:
            # invoke method
            the_sg.get_statistic("foo")

            # validate the error message.
            self.assertTrue("the_name was not a valid option." in context.exception)

    # test method for calculate_statistics()
    def test_calculate_statistics(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run complete setup
        dsa.run_complete_setup()

        # get the dataframe from the dsa
        the_df = dsa.the_df

        # instantiate the SG
        the_sg = StatisticGroup(the_df['Children'])

        # invoke calculate_statistics()
        the_sg.calculate_statistics()

        # run assertions
        self.assertIsNotNone(the_sg.storage[COUNT])
        self.assertIsNotNone(the_sg.storage[MEAN])
        self.assertIsNotNone(the_sg.storage[MEDIAN])
        self.assertIsNotNone(the_sg.storage[VARIANCE])
        self.assertIsNotNone(the_sg.storage[STD_DEV])
        self.assertIsNotNone(the_sg.storage[MEAN_ABS_DEV])
        self.assertIsNotNone(the_sg.storage[SKEW])
        self.assertIsNotNone(the_sg.storage[KURTOSIS])
        self.assertIsNotNone(the_sg.storage[NAME])

        self.assertEqual(the_sg.storage[COUNT], 10000)
        self.assertEqual(the_sg.storage[MEAN], the_df['Children'].mean())
        self.assertEqual(the_sg.storage[MEDIAN], the_df['Children'].median())
        self.assertEqual(the_sg.storage[VARIANCE], the_df['Children'].var())
        self.assertEqual(the_sg.storage[STD_DEV], the_df['Children'].std())
        self.assertEqual(the_sg.storage[MEAN_ABS_DEV], np.mean(np.abs(the_df['Children'])))
        self.assertEqual(the_sg.storage[SKEW], the_df['Children'].skew())
        self.assertEqual(the_sg.storage[KURTOSIS], the_df['Children'].kurtosis())
        self.assertEqual(the_sg.storage[NAME], 'Children')

    # test method for get_series()
    def test_get_series(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run complete setup
        dsa.run_complete_setup()

        # get the dataframe from the dsa
        the_df = dsa.the_df

        # instantiate the SG
        the_sg = StatisticGroup(the_df['Children'])

        # run assertions
        self.assertIsNotNone(the_sg.get_series())
        self.assertIsInstance(the_sg.get_series(), Series)
        self.assertEqual(the_sg.get_series().sum(), the_df['Children'].sum())

    # test method for find_distribution()
    def test_find_distribution(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run complete setup
        dsa.run_complete_setup()

        # get the dataframe from the dsa
        the_df = dsa.the_df

        # instantiate the SG
        the_sg = StatisticGroup(the_df['Children'])

        # run calculate_statistics
        the_sg.calculate_statistics()

        # invoke the method
        the_sg.find_distribution()

        # run assertions

        # normality tests
        self.assertIsNotNone(the_sg.storage[NORMALITY_TEST])
        self.assertIsNotNone(the_sg.storage[NORMALITY_TEST_P_VALUE])
        self.assertIsNotNone(the_sg.storage[NORMALITY_TEST_METHOD])
        self.assertEqual(the_sg.storage[NORMALITY_TEST], NORMALITY_TEST_NOT_NORMAL)
        self.assertEqual(the_sg.storage[NORMALITY_TEST_P_VALUE], 0)
        self.assertEqual(the_sg.storage[NORMALITY_TEST_METHOD], KOLMOGOROV_SMIRNOV_TEST)

        # exponential tests
        self.assertIsNotNone(the_sg.storage[EXPONENTIAL_TEST])
        self.assertIsNotNone(the_sg.storage[EXPONENTIAL_TEST_P_VALUE])
        self.assertIsNotNone(the_sg.storage[EXPONENTIAL_TEST_METHOD])
        self.assertEqual(the_sg.storage[EXPONENTIAL_TEST], EXPONENTIAL_TEST_NOT_EXP)
        self.assertEqual(the_sg.storage[EXPONENTIAL_TEST_P_VALUE], 0)
        self.assertEqual(the_sg.storage[EXPONENTIAL_TEST_METHOD], KOLMOGOROV_SMIRNOV_TEST)

        # instantiate the SG
        the_sg = StatisticGroup(the_df['Email'])

        # run calculate_statistics
        the_sg.calculate_statistics()

        # invoke the method
        the_sg.find_distribution()

        # normality tests
        self.assertIsNotNone(the_sg.storage[NORMALITY_TEST])
        self.assertIsNotNone(the_sg.storage[NORMALITY_TEST_P_VALUE])
        self.assertIsNotNone(the_sg.storage[NORMALITY_TEST_METHOD])
        self.assertEqual(the_sg.storage[NORMALITY_TEST], NORMALITY_TEST_NOT_NORMAL)
        self.assertEqual(the_sg.storage[NORMALITY_TEST_P_VALUE], 0)
        self.assertEqual(the_sg.storage[NORMALITY_TEST_METHOD], KOLMOGOROV_SMIRNOV_TEST)