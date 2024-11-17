import logging
import numpy as np
import pandas as pd

from numpy import int64
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_object_dtype, is_bool_dtype, is_float_dtype, is_numeric_dtype
from model.BaseModel import BaseModel
from model.constants.BasicConstants import TIMEZONE_DICTIONARY, Z_SCORE, BOOL_SWAP_DICTIONARY, NAN_SWAP_DICT, \
    EDUCATION_DICTIONARY
from model.analysis.Detector import Detector


# class definition for Converter utilities
class Converter(BaseModel):
    # constants
    NORMAL_DTYPES = [int64]

    # init() method
    def __init__(self):
        # call superclass
        super().__init__()

        # define logger
        self.logger = logging.getLogger(__name__)

        # log that we are initializing
        self.logger.debug("Initializing Converter.")

    # convert to boolean
    def convert_column_to_boolean(self, the_df, the_column_name):
        # log that we've been called
        self.logger.debug("A request to convert column to boolean has been requested.")

        # run validations
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("The incoming argument was None or incorrect type.")
        elif not isinstance(the_column_name, str):
            raise SyntaxError("The incoming argument was None or incorrect type.")
        elif the_column_name not in the_df:
            raise SyntaxError(f"The incoming column [{the_column_name}] was not present in the DataFrame.")

        # log what we received
        self.logger.debug(f"Attempting to convert [{the_column_name}] to boolean.")

        # perform the mapping
        the_df[the_column_name] = the_df[the_column_name].replace(BOOL_SWAP_DICTIONARY)

        # log the output
        self.logger.debug(f"Final result is \n[{str(the_df[the_column_name].describe())}]")

        # return the result
        return the_df

    # convert all items in a list of columns to a bool type
    def convert_to_boolean(self, the_df, the_list):
        # validate arguments
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("The incoming argument was None or incorrect type.")
        elif not isinstance(the_list, list):
            raise SyntaxError("The incoming argument was None or incorrect type.")

        # log that we've been called
        self.logger.debug(f"A request to convert columns[{the_list}] to boolean has been made.")

        # loop over the items in the list
        for next_item in the_list:
            # invoke convert_column_to_boolean()
            self.convert_column_to_boolean(the_df, next_item)

        return the_df

    # replace Nan values with another value
    def replace_nan_with_value(self, the_df, the_column_name):
        # run validations
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("The incoming argument was None or incorrect type.")
        elif not isinstance(the_column_name, str):
            raise SyntaxError("The incoming argument was None or incorrect type.")
        elif the_column_name not in the_df:
            raise SyntaxError("The incoming column name was not present in the DataFrame.")

        # log what we received
        self.logger.debug(f"Attempting to convert [{the_column_name}] Nan values.")

        # perform the mapping
        the_df[the_column_name] = the_df[the_column_name].replace(NAN_SWAP_DICT)

        # log the output
        self.logger.debug(f"Final result is \n[{str(the_df[the_column_name].describe())}]")

        # return the result
        return the_df

    # convert float data columns into int if they meet criteria.
    def clean_floats_that_should_be_int(self, the_df, the_list):
        # validate arguments
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("The incoming DataFrame was None or incorrect type.")
        elif not isinstance(the_list, list):
            raise SyntaxError("The incoming list was None or incorrect type.")
        elif not len(the_list) > 0:
            raise SyntaxError("The list of columns was empty.")
        elif not set(the_list).issubset(set(the_df.columns)):
            raise SyntaxError("An item in the list was not a column in the DataFrame.")

        # log that we've been called
        self.logger.debug(f"A request to convert columns[{the_list}] to boolean has been made.")

        # variable declaration
        detector = Detector()

        # loop over the list of columns
        for the_column in the_list:
            # invoke detect_when_float_is_int()
            if detector.detect_when_float_is_int(the_df[the_column]):
                # log that the column is actually int.
                self.logger.debug(f"column [{the_column}] was found to be int.")

                # the strategy here is that we want to convert the column to Int64 datatype
                # as this is a datatype that allows null.
                the_df[the_column] = the_df[the_column].astype('Int64')
            else:
                # log that we will not convert the column
                self.logger.debug(f"not going to attempt conversation on [{the_column}]")

    # clean columns with blank name.
    def clean_columns_with_blank_name(self, the_df):
        # validate arguments
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("The incoming argument was None or incorrect type.")

        # log that we've been called
        self.logger.debug("A request to clean blank column names has been made.")

        # variable declaration
        detector = Detector()

        # get the list of blank column names
        columns_with_no_name = detector.get_list_of_blank_column_names(the_df)

        # loop through the column names
        for next_index in columns_with_no_name:
            # log the index to be dropped
            self.logger.debug(f"Dropping column [{next_index}]")

            # drop the column by index id
            the_df.drop(next_index, axis=1, inplace=True)

        # log that we are complete
        self.logger.debug("A request to clean blank column names is complete")

    # get a dataframe that is normalized.
    def get_normalized_df(self, the_dataframe):
        # run validations
        if not self.is_valid(the_dataframe, DataFrame):
            raise SyntaxError("The argument was None or the wrong type.")

        # log what we're about to do
        self.logger.debug("About to normalize DataFrame.")

        # variable declaration
        the_result = DataFrame()

        # loop through all the columns
        for the_column in the_dataframe:
            # log the current column
            self.logger.debug(f"current column is {the_column}]")

            # check if the column is OBJECT or BOOLEAN
            if is_object_dtype(the_dataframe[the_column]) or is_bool_dtype(the_dataframe[the_column]):
                # log that we're skipping normalizing that column
                self.logger.debug(f"cannot normalize column[{the_column}] that is type OBJECT or BOOLEAN.")
            # check if the column is int or float
            elif the_dataframe[the_column].dtype == np.int64 or is_float_dtype(the_dataframe[the_column]):
                # log that we're processing this columns
                self.logger.debug(f"adding column [{the_column}] to normalized ")

                # invoke normalize_series()
                norm_series = self.normalize_series(the_dataframe[the_column])

                # generate name of column
                norm_col_name = the_column + Z_SCORE

                # add series to DataFrame
                the_result[norm_col_name] = norm_series

        # log that we're complete
        self.logger.debug("Complete with normalization of DataFrame.")

        # return normalized df
        return the_result

    # z-score normalize a Series representing a column in a DataFrame and
    def normalize_series(self, the_series):
        # perform validation
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The incoming Series was None or incorrect type.")

        # log that we've been called
        self.logger.debug(f"A request to normalize [{the_series.name}] has been made.")

        # variable declarations
        the_result = None
        the_mean = the_series.mean()
        the_std = the_series.std()

        # generate the result
        the_result = (the_series - the_mean) / the_std

        # log that we are complete
        self.logger.debug(f"z-score normalization is complete for [{the_series.name}].")

        # return the result
        return the_result

    # fill NAN values into numerical Series ONLY
    def fill_nan_values_for_numerical_series(self, the_series):
        # perform validation
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The incoming Series was None or incorrect type.")
        elif not is_numeric_dtype(the_series):
            raise TypeError("The incoming Series was not numeric.")

        # log that we've been called
        self.logger.debug(f"A request to fill NA for [{the_series.name}] has been made.")

        # variable declarations
        the_result = None
        the_detector = Detector()
        the_median = the_series.median()
        the_mean = the_series.mean()

        # check if we have int64 with Nan
        if the_detector.detect_int_with_na(the_series):
            # replace nan values with median
            the_result = the_series.fillna(the_median)
        # has to be float64
        else:
            # replace nan values with mean
            the_result = the_series.fillna(the_mean)

        # log that we are complete
        self.logger.debug(f"A request to fill NA for [{the_series.name}] is complete.")

        # return the result
        return the_result

    # fill NAN values for all numerical series if present
    def fill_nan_values_for_numerical_df(self, the_dataframe):
        # run validations
        if not self.is_valid(the_dataframe, DataFrame):
            raise SyntaxError("The Dataframe argument was None or the wrong type.")

        # loop through all the columns
        for the_column in the_dataframe:
            # log the current column
            self.logger.debug(f"current column is {the_column}]")

            # check if the column is OBJECT or BOOLEAN
            if is_object_dtype(the_dataframe[the_column]) or is_bool_dtype(the_dataframe[the_column]):
                # log that we're skipping filling in data for these types
                self.logger.debug(f"cannot fill NaN values for column[{the_column}] that is type OBJECT or BOOLEAN.")
            # check if the column is int or float
            elif the_dataframe[the_column].dtype == np.int64 or is_float_dtype(the_dataframe[the_column]):
                # log that we're processing this columns
                self.logger.debug(f"filling Nan values for [{the_column}].")

                # grab just a series
                the_series = the_dataframe[the_column]

                # invoke fill_nan_values_for_numerical_series()
                the_dataframe[the_column] = self.fill_nan_values_for_numerical_series(the_series)
            # TODO: add ability to handle datetime

        # return
        return the_dataframe

    # clean timezone data
    def clean_timezone(self, the_series):
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The Series argument was None or the wrong type.")

        # loop through the values in the TIMEZONE_DICTIONARY, replace
        for old_time_zone in TIMEZONE_DICTIONARY.keys():
            # get the value for the timezone
            new_time_zone = TIMEZONE_DICTIONARY[old_time_zone]

            # log the current item to replace
            self.logger.debug(f"replacing [{old_time_zone}] with [{new_time_zone}]")

            # replace values in Series
            the_series.replace(old_time_zone, new_time_zone, inplace=True)

    # clean education data
    def clean_education_level(self, the_series):
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The Series argument was None or the wrong type.")

        # loop through the values in the TIMEZONE_DICTIONARY, replace
        for old_value in EDUCATION_DICTIONARY.keys():
            # get the new value
            new_value = EDUCATION_DICTIONARY[old_value]

            # log the current item to replace
            self.logger.debug(f"replacing [{old_value}] with [{new_value}]")

            # replace values in Series
            the_series.replace(old_value, new_value, inplace=True)

    # remove duplicates from dataframe
    def remove_duplicate_rows(self, the_df) -> DataFrame:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("the_df is None or incorrect type.")

        # log that we've been called
        self.logger.debug("A request to remove all duplicate rows has been made.")

        # remove all duplicates
        the_result = the_df.drop_duplicates()

        # return
        return the_result

    # convert a numpy.array to DataFrame
    @staticmethod
    def convert_array_to_dataframe(the_array: np.ndarray, the_columns: list) -> DataFrame:
        # run validations
        if not isinstance(the_array, np.ndarray):
            raise SyntaxError("the_array is None or incorrect type.")

        # variable definition
        the_result = pd.DataFrame(data=the_array, columns=the_columns)

        return the_result
