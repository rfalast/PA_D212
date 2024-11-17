import logging
import numpy as np

from scipy.stats import shapiro, kstest
from model.BaseModel import BaseModel
from pandas import Series
from model.constants.StatisticsConstants import NAME, MEAN, MEDIAN, VARIANCE, STD_DEV, COUNT, SKEW, KURTOSIS, \
    STAT_OPTIONS, NORMALITY_TEST, MEAN_ABS_DEV, NORMALITY_TEST_NOT_NORMAL, NORMALITY_TEST_NORMAL, \
    NORMALITY_TEST_P_VALUE, NORMALITY_TEST_METHOD, SHAPIRO_WILKS_NORMALITY_TEST, \
    EXPONENTIAL_TEST_NOT_EXP, EXPONENTIAL_TEST, EXPONENTIAL_TEST_EXP, EXPONENTIAL_TEST_P_VALUE, EXPONENTIAL_TEST_METHOD, \
    KOLMOGOROV_SMIRNOV_TEST


# The StatisticGroup object represents all the statistics about a specific Series
class StatisticGroup(BaseModel):

    # init() method
    def __init__(self, the_series):
        # call super class
        super().__init__()

        # run validations
        if not isinstance(the_series, Series):
            raise AttributeError("the_series was None or incorrect type.")

        # define logger
        self.logger = logging.getLogger(__name__)

        # variable declaration
        self.the_series = the_series
        self.storage = {}

    # get a specific statistic
    def get_statistic(self, the_name):
        """
        Get the value of the statistic from internal storage.
        :param the_name:
        :return: value of the statistic
        """
        # run validations
        if not isinstance(the_name, str):
            raise AttributeError("the_name was None or incorrect type.")
        elif the_name not in STAT_OPTIONS:
            raise AttributeError("the_name was not a valid option.")
        elif the_name not in self.storage:
            # invoke calculate_statistics()
            self.calculate_statistics()

        # get the value from storage
        return self.storage[the_name]

    # get a string representation of statistics
    def dump_to_string(self):
        # variable declaration
        the_result = ""

        # check if the storage is empty
        if len(self.storage) == 0:
            self.calculate_statistics()

        # loop over values in STAT_OPTIONS
        for the_option in STAT_OPTIONS:
            the_result += the_option + ":[" + self.not_none(self.storage[the_option]) + "], "

        # remove the trailing comma and space
        the_result = the_result[0:-2]

        # return
        return the_result

    # get the actual underlying series
    def get_series(self):
        """
        get the underlying Series for this StatisticGroup
        :return: Series
        """
        return self.the_series

    # run all statistics
    def calculate_statistics(self):
        """
        Calculate statistics and store them on the storage variable.
        :return: None
        """
        # store name
        self.storage[NAME] = self.the_series.name
        # store the count
        self.storage[COUNT] = self.the_series.count()
        # store the mean
        self.storage[MEAN] = self.the_series.mean()
        # store the MEAN_ABS_DEV
        self.storage[MEAN_ABS_DEV] = np.mean(np.abs(self.the_series))
        # store the median
        self.storage[MEDIAN] = self.the_series.median()
        # store the variance
        self.storage[VARIANCE] = self.the_series.var()
        # store the standard deviation
        self.storage[STD_DEV] = self.the_series.std()
        # store the skew
        self.storage[SKEW] = self.the_series.skew()
        # store the kurtosis
        self.storage[KURTOSIS] = self.the_series.kurtosis()

    # determine distribution
    def find_distribution(self, max_sample=100):
        # variable declaration
        normality_test = None

        ##################################################
        # test for Normality
        ##################################################
        if self.storage[COUNT] >= 5000:
            # log what test we're using
            self.logger.debug("Kolmogorov-Smirnov test for normality is being used.")

            # execute the Kolmogorov-Smirnov test
            normality_test = kstest(self.the_series, 'norm')

            # check the p-value
            if normality_test[1] < 0.05:
                self.storage[NORMALITY_TEST] = NORMALITY_TEST_NOT_NORMAL
            else:
                self.storage[NORMALITY_TEST] = NORMALITY_TEST_NORMAL

            # store the p-value
            self.storage[NORMALITY_TEST_P_VALUE] = normality_test[1]

            # store the method used to test normality
            self.storage[NORMALITY_TEST_METHOD] = KOLMOGOROV_SMIRNOV_TEST
        else:
            # log what test we're using
            self.logger.debug("Shapiro-Wilks test for normality is being used.")

            # execute the Shapiro-Wilks test
            normality_test = shapiro(self.the_series)

            # check the p-value
            if normality_test[1] < 0.05:
                self.storage[NORMALITY_TEST] = NORMALITY_TEST_NOT_NORMAL
            else:
                self.storage[NORMALITY_TEST] = NORMALITY_TEST_NORMAL

            # store the p-value
            self.storage[NORMALITY_TEST_P_VALUE] = normality_test[1]

            # store the method used to test normality
            self.storage[NORMALITY_TEST_METHOD] = SHAPIRO_WILKS_NORMALITY_TEST

        ##################################################
        # test for exponential
        ##################################################

        # log what test we're using
        self.logger.debug("Using the Kolmogorov-Smirnov to test if distribution is exponential.")

        # execute the Kolmogorov-Smirnov test
        normality_test = kstest(self.the_series, "expon")

        # check the p-value
        if normality_test[1] < 0.05:
            self.storage[EXPONENTIAL_TEST] = EXPONENTIAL_TEST_NOT_EXP
        else:
            self.storage[EXPONENTIAL_TEST] = EXPONENTIAL_TEST_EXP

        # store the p-value
        self.storage[EXPONENTIAL_TEST_P_VALUE] = normality_test[1]

        # store the method used to test normality
        self.storage[EXPONENTIAL_TEST_METHOD] = KOLMOGOROV_SMIRNOV_TEST

        # test for uniform
        # times is the series
        # stats.kstest(times, stats.uniform(loc=0.0, scale=100.0).cdf)
        #_stats, p = scipy_stats.kstest(self.s, scipy_stats.uniform(loc=low, scale=high - low).cdf)

        # test for log normal
