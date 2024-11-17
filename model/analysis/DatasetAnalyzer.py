import logging
import math
import numpy as np

from pandas import DataFrame, Int64Dtype
from pandas.core.dtypes.common import is_object_dtype, is_float_dtype, is_bool_dtype, is_datetime64_dtype
from model.BaseModel import BaseModel
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import TIMEZONE_COLUMN, Z_SCORE, DEFAULT_INDEX_NAME, MT_OPTIONS, \
    MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION
from model.analysis.Detector import Detector
from model.constants.DatasetConstants import INT64_COLUMN_KEY, BOOL_COLUMN_KEY, FLOAT64_COLUMN_KEY, OBJECT_COLUMN_KEY, \
    COLUMN_KEY, DATETIME_COLUMN_KEY, COLUMN_COUNT_KEY, UNIQUE_COLUMN_VALUES, COLUMN_TOTAL_COUNT_KEY, COLUMN_NA_COUNT, \
    DATASET_TYPES, UNIQUE_COLUMN_LIST_KEY, UNIQUE_COLUMN_FLAG, UNIQUE_COLUMN_RATIOS, BOOL_COLUMN_COUNT_KEY, \
    OBJECT_COLUMN_COUNT_KEY, BOOL_VALUE_KEY, BOOL_CLEANED_FLAG, INT64_NAN_FLAG, YES_NO_LIST, DATA_TYPE_KEYS, \
    LM_MODEL_TYPES

from model.converters.Converter import Converter
from util.CommonUtils import CommonUtils


# class for performing analysis operations
class DatasetAnalyzer(BaseModel):
    # init method
    def __init__(self, the_dataframe, the_name="default"):
        # call superclass
        super().__init__()

        # initialize the logger
        self.logger = logging.getLogger(__name__)

        # log that we've been invoked
        self.logger.debug("A request to instantiate a DatasetAnalyzer has been made.")

        # validate that we have an argument
        if not self.is_valid(the_dataframe, DataFrame):
            raise SyntaxError("The incoming argument was None or incorrect type.")
        elif not self.is_valid(the_name, str):
            raise SyntaxError("The incoming name argument was None or incorrect type.")
        else:
            # log that we captured the dataframe
            self.logger.debug("Dataframe captured.")

            # store the dataframe internally
            self.the_original_df = the_dataframe
            self.the_df = the_dataframe
            self.the_normal_df = None

            # set the has_boolean flag to False
            self.has_boolean = False

            # set the name of the dataset that this is populated from
            self.dataset_name = the_name

        # additional variable declarations
        self.storage = {}
        self.detector = None
        self.linear_model_storage = {}

    # retrieve_columns from dataframe and store them internally.
    # This is a state based function.
    def retrieve_columns(self):
        # log what we're doing
        logging.debug("A request to retrieve columns from dataframe has been made.")

        # create a dicts to store values
        column_dict = {}
        count_dict = {}
        total_count_dict = {}
        int_list = []
        float_list = []
        object_list = []
        bool_list = []
        unique_column_list = []
        datetime_list = []
        unique_dict = {}
        column_na_count_dict = {}
        data_type_list = []
        item_num = 1

        # create detector
        detector = Detector()

        # retrieve columns and log them
        for column_header in self.the_df.columns:
            # log the columns
            self.logger.debug(f"column[{column_header}] of type[{self.the_df[column_header].dtype}]")

            # capture the name and type in the column dict
            column_dict[column_header] = self.the_df[column_header].dtype

            # capture count or item # on the count_dict
            count_dict[column_header] = item_num

            # calculate the number of NA values for this column
            column_na_count_dict[column_header] = self.the_df[column_header].isna().sum()

            # calculate the number of values for this column
            total_count_dict[column_header] = len(self.the_df[column_header])

            # log the count of NaN's for the current column
            self.logger.debug(f"column[{column_header}] NaN count is [{column_na_count_dict[column_header]}]")

            # now, we need to capture the sub-category to store the columns in the appropriate list

            # INT
            if self.the_df[column_header].dtype == np.int64 or self.the_df[column_header].dtype == Int64Dtype():
                # log that we're adding to the INT list
                self.logger.debug(f"Adding [{column_header}] to int list.")

                # add to the datatype list if it's not there
                if INT64_COLUMN_KEY not in data_type_list:
                    data_type_list.append(INT64_COLUMN_KEY)

                # add to list of INTEGERS
                int_list.append(column_header)

                # check if the column is unique
                if detector.detect_if_series_is_unique(self.the_df[column_header]):
                    # add to the unique column list
                    if column_header not in unique_column_list:
                        unique_column_list.append(column_header)
            # FLOAT 64
            elif is_float_dtype(self.the_df[column_header]):
                # log that we're adding to the FLOAT list
                self.logger.debug(f"Adding [{column_header}] to float list.")

                # add to the datatype list if it's not there
                if FLOAT64_COLUMN_KEY not in data_type_list:
                    data_type_list.append(FLOAT64_COLUMN_KEY)

                # add to list of FLOAT64
                float_list.append(column_header)

                # check if the column is unique
                if detector.detect_if_series_is_unique(self.the_df[column_header]):
                    # add to the unique column list
                    if column_header not in unique_column_list:
                        unique_column_list.append(column_header)
            # OBJECT
            elif is_object_dtype(self.the_df[column_header]):
                # log that we're adding to the OBJECT list
                self.logger.debug(f"Adding [{column_header}] to object list.")

                # add to the datatype list if it's not there
                if OBJECT_COLUMN_KEY not in data_type_list:
                    data_type_list.append(OBJECT_COLUMN_KEY)

                # add to list of objects
                object_list.append(column_header)

                # add the column_header as a key to new dictionary
                unique_dict[column_header] = {}

                # check if the column is unique
                if detector.detect_if_series_is_unique(self.the_df[column_header]):
                    # add to the unique column list
                    if column_header not in unique_column_list:
                        unique_column_list.append(column_header)
            # BOOLEAN
            elif is_bool_dtype(self.the_df[column_header]):
                # log that we're adding to the OBJECT list
                self.logger.debug(f"Adding [{column_header}] to boolean list.")

                # add to the datatype list if it's not there
                if BOOL_COLUMN_KEY not in data_type_list:
                    data_type_list.append(BOOL_COLUMN_KEY)

                # make sure this is flipped, as once we pass through this function
                # the second time, we won't have boolean on the underlying dataframe.
                self.has_boolean = True

                # add to list of objects
                bool_list.append(column_header)

                # Additional note, no boolean can be unique, so no reason to test it.
            # DATE
            elif is_datetime64_dtype(self.the_df[column_header]):
                # log that we're adding to the OBJECT list
                self.logger.debug(f"Adding [{column_header}] to datetime64 list.")

                # add to list of datetime
                datetime_list.append(column_header)

                # check if the column is unique
                if detector.detect_if_series_is_unique(self.the_df[column_header]):
                    # add to the unique column list
                    if column_header not in unique_column_list:
                        unique_column_list.append(column_header)
            # UNKNOWN
            else:
                # log that this shouldn't happen
                self.logger.error(f"column [{column_header}] is not being bucketed.")

                # log additional information about the field
                self.logger.debug(f"The type of [{column_header}] is [{str(self.the_df[column_header].dtype)}]")
                self.logger.debug(f"Result of is_bool_dtype(self.the_df[column_header]) is "
                                  f"[{is_bool_dtype(self.the_df[column_header])}]")

            # increment the item
            item_num = item_num + 1

        # place dictionaries and lists into storage graph.  This happens each pass.
        self.storage[COLUMN_KEY] = column_dict
        self.storage[INT64_COLUMN_KEY] = int_list
        self.storage[FLOAT64_COLUMN_KEY] = float_list
        self.storage[OBJECT_COLUMN_KEY] = object_list
        self.storage[BOOL_COLUMN_KEY] = bool_list
        self.storage[DATETIME_COLUMN_KEY] = datetime_list
        self.storage[COLUMN_COUNT_KEY] = count_dict
        self.storage[UNIQUE_COLUMN_VALUES] = unique_dict
        self.storage[COLUMN_TOTAL_COUNT_KEY] = total_count_dict
        self.storage[COLUMN_NA_COUNT] = column_na_count_dict
        self.storage[DATASET_TYPES] = data_type_list
        self.storage[UNIQUE_COLUMN_LIST_KEY] = unique_column_list

        # these elements of the object graph could have been set in earlier pass.
        self.__update_storage_key__(UNIQUE_COLUMN_FLAG, {})
        self.__update_storage_key__(UNIQUE_COLUMN_RATIOS, {})
        self.__update_storage_key__(BOOL_COLUMN_COUNT_KEY, {})
        self.__update_storage_key__(OBJECT_COLUMN_COUNT_KEY, {})
        self.__update_storage_key__(BOOL_VALUE_KEY, {})

        # STATE BASED VARIABLE DECLARATION.
        # The first time through the function, the storage dictionary will not be initialized.  Thus, we
        # will need to set a key of BOOL_CLEANED_FLAG and INT64_NAN_FLAG with a value of False.

        # check if self.storage contains a key BOOL_CLEANED_FLAG
        if BOOL_CLEANED_FLAG in self.storage:
            # log our state
            self.logger.info(f"BOOL_CLEANED_FLAG value is [{str(self.storage[BOOL_CLEANED_FLAG])}]")
        # since there is no key, we have to initialize it to false
        else:
            # initialize to False
            self.storage[BOOL_CLEANED_FLAG] = False

            # log that we're initializing the key for the first time.
            self.logger.debug("Initializing key BOOL_CLEANED_FLAG")

        # check if self.storage contains a key INT64_NAN_FLAG
        if INT64_NAN_FLAG in self.storage:
            # log our state
            self.logger.info(f"INT64_NAN_FLAG value is [{str(self.storage[INT64_NAN_FLAG])}]")
        # since there is no key, we have to initialize it to false
        else:
            # initialize to False
            self.storage[INT64_NAN_FLAG] = False

            # log that we're initializing the key for the first time.
            self.logger.debug("Initializing key INT64_NAN_FLAG")

        # log what the boolean key is holding
        self.logger.debug(f"At KEY BOOL_COLUMN_KEY [{str(self.storage[BOOL_COLUMN_KEY])}]")

        # log the current state.
        self.logger.debug(f"BOOL_CLEANED_FLAG value is [{str(self.storage[BOOL_CLEANED_FLAG])}]")
        self.logger.debug(f"INT64_NAN_FLAG value is [{str(self.storage[INT64_NAN_FLAG])}]")

    # analyze columns that are OBJECT(s)
    # The state is controlled by self.storage[self.BOOL_COLUMN_KEY] which returns a FALSE if the boolean columns
    # have been cleaned up.  If this is the first pass through the function, the value would be False.
    # If the boolean located at self.storage[self.BOOL_COLUMN_KEY] is TRUE, then the boolean columns have been
    # cleaned, and this is the second pass through the function.
    def analyze_columns(self):
        # log that we've been called
        self.logger.debug("A request to analyze the columns have been made.")

        # log the current state
        self.logger.info(f"At KEY BOOL_COLUMN_KEY [{str(self.storage[BOOL_COLUMN_KEY])}]")

        # variable declaration
        boolean_dict = {}

        # loop over the items in the object list.  This is a list of columns, which is stored in the
        # variable next_value
        for next_value in self.storage[OBJECT_COLUMN_KEY]:

            # log the first retrieved item
            self.logger.debug(f"retrieved column [{next_value}]")

            # get the number of unique values
            unique_value_count = len(self.the_df[next_value].unique())

            # log the number of unique values
            self.logger.debug(f"the number of unique values for column [{next_value}] is [{unique_value_count}]")

            # get unique column ratio
            unique_column_ratio = len(self.the_df[next_value].unique()) / self.the_df[next_value].count()

            # log the ratio
            self.logger.debug(f"column [{next_value}] has a ratio of [{unique_column_ratio}]")

            # capture the ratio
            self.storage[UNIQUE_COLUMN_RATIOS][next_value] = unique_column_ratio

            # if the number of values is two, then capture it as pre-boolean
            if unique_value_count == 2:
                # log that we found a boolean column
                self.logger.debug(f"Boolean column found [{next_value}]")

                # we do not want to populate the BOOL_COLUMN_KEY list until after the passage
                # through run_conversion_for_boolean().

                # capture the column name, and unique values in the boolean_dict
                boolean_dict[next_value] = self.the_df[next_value].unique().tolist()

                # capture values and place them in a list
                self.storage[UNIQUE_COLUMN_VALUES][next_value] = self.the_df[next_value].unique().tolist()

                # log the unique values
                self.logger.debug(f"Placing Boolean list [{self.the_df[next_value].unique().tolist()}] "
                                  f"on UNIQUE_COLUMN_VALUES key [{next_value}]")

                # make sure the flag is flipped
                # this is a questionable state based changed.  Need to investigate what this actually does
                self.has_boolean = True

            # we don't have a boolean, so find out if we want to display the results
            elif unique_value_count <= 100 and unique_column_ratio < .20:
                # log the column
                self.logger.debug(f"Found a column [{next_value}] where we will capture unique values.")

                # capture the list of unique values for a column
                unique_value_list = self.the_df[next_value].unique().tolist()

                # log the unique values
                self.logger.debug(f"VALUES[{self.the_df[next_value].unique().tolist()}]")

                # we next need to capture the count by column.  No boolean fields should hit this branch.
                # Create a dictionary of all the values in the current column.
                temp_dict = {}

                # STATE BASED LOGIC.
                # there is some state based logic here.  In the call to analyze_dataset("INITIAL"), which is the
                # first pass through the analyzer, we want to generate an Initial report to document what object
                # type objects hold NA values.  In the loop below, on the first iteration through, we may have
                # N/A's in the unique values for a specific column.  Thus, our current approach is to remove the
                # N/A from unique_value_list.

                # we need to loop over unique values and get the count.
                for unique_value in unique_value_list:
                    # check if the value is Nan, can only occur during INITIAL state
                    if CommonUtils.isnan(unique_value):
                        # log a message that we found a nana
                        self.logger.debug(f"A Nan value was found on column when getting field count.[{next_value}]")

                        # calculate the count differently
                        the_count = self.the_df[next_value].isna().sum()

                        # change the value stored in unique_value to string
                        unique_value = "nan"
                    else:
                        # get the count using value_counts()
                        the_count = self.the_df[next_value].value_counts().get(key=unique_value)

                    # populate the key as current unique field, value of the count
                    temp_dict[unique_value] = the_count

                    # log what we're adding
                    self.logger.debug(f"placing on OBJECT_COLUMN_COUNT_KEY column [{next_value}] "
                                      f"key [{unique_value}] value [{the_count}]")

                # overwrite the unique_value_list with a new list
                unique_value_list = list()

                # STATE BASED LOGIC
                # If there are NA's present in a column, the lambda function doesn't work.  The lambda function
                # performs an evaluation where there is a < operation.  Since, a < operation cannot be properly
                # evaluated if one of the elements is a N/A or NoneType, Python will blow up.  I fix this
                # not performing the sort.  Since this lambda sort is presentation layer logic, it's not a major
                # loss.

                # thus, we check for nan in the unique_value_list.  The initial time through, this
                # will be True, so skip the lambda sort.
                if math.nan in unique_value_list:
                    # log that it occurred
                    self.logger.debug(f"A nan was found for column [{next_value}]")
                else:
                    # We need to sort the unique values by value count and replace the list of object values
                    # on the key UNIQUE_COLUMN_VALUES
                    for key, value in sorted(temp_dict.items(), key=lambda kv: kv[1], reverse=True):
                        # append the value to the list
                        unique_value_list.append(key)

                # capture the unique values and store them as a list on the storage dict
                self.storage[UNIQUE_COLUMN_VALUES][next_value] = unique_value_list

                # populate the key onto OBJECT_COLUMN_COUNT_KEY
                self.storage[OBJECT_COLUMN_COUNT_KEY][next_value] = temp_dict

                # log the value we just stored
                self.logger.debug(f"Storing on [OBJECT_COLUMN_COUNT_KEY][{next_value}]-->[{temp_dict}]")

            # we possibly have a column with 100% uniqueness
            else:
                # log the column name
                self.logger.info(f"column [{next_value}] unique values will not be captured.")

                # capture the statistics on the unique count
                self.storage[UNIQUE_COLUMN_VALUES][next_value] = ["TOO MANY TO DISPLAY"]

            # now, we need to set the unique flag for this column
            if unique_column_ratio == 1.0:
                # log that we're setting the flag to Y
                self.logger.info(f"Setting column [{next_value}] to Y")

                # set to Y
                self.storage[UNIQUE_COLUMN_FLAG][next_value] = YES_NO_LIST[0]

                # add to the list of unique columns
                if next_value not in self.storage[UNIQUE_COLUMN_LIST_KEY]:
                    self.storage[UNIQUE_COLUMN_LIST_KEY].append(next_value)
            # this isn't a unique column
            else:
                # set to N
                self.storage[UNIQUE_COLUMN_FLAG][next_value] = YES_NO_LIST[1]

        # log the values of the boolean dictionary
        self.logger.info(f"prior to capture of boolean dictionary BOOL_COLUMN_KEY holds "
                         f"[{self.storage[BOOL_COLUMN_KEY]}]")

        # this is state based decision-making.  The first time through this function, we analyze OBJECTS
        # in order to identify which is of type boolean, and store the current column in the key BOOL_COLUMN_KEY.
        # However, the second time through this function, the OBJECT list has already been cleaned of actual
        # boolean values.  Thus, we don't want to overwrite the accurate boolean columns in BOOL_COLUMN_KEY.

        # check if the boolean cleaned flag has been set.
        # if the flag has been set to true, log that we did nothing.
        if self.storage[BOOL_CLEANED_FLAG]:
            # log that this path was taken.
            self.logger.info(f"BOOL_CLEANED_FLAG value is True [{str(self.storage[BOOL_CLEANED_FLAG])}]")
        # if the flag has not been set, this is the first pass through the function
        else:
            # capture the boolean dictionary of column names, values.  Value is Yes / No at this point.
            # self.storage[self.BOOL_COLUMN_KEY] = boolean_dict
            # capture the dirty boolean dictionary.  Key is column name, value is the uncleaned two options
            # that reflect boolean logic.  This needs to be mapped to True/False later.
            self.storage[BOOL_VALUE_KEY] = boolean_dict

            # log that flag was false
            self.logger.info(f"BOOL_CLEANED_FLAG value is False [{str(self.storage[BOOL_CLEANED_FLAG])}]")

        # log what the boolean key is holding
        self.logger.info(f"At KEY BOOL_COLUMN_KEY [{str(self.storage[BOOL_COLUMN_KEY])}]")

    # run boolean variable conversion
    # coming into this function, the BOOL_COLUMN_KEY holds a dictionary.  W
    def run_conversion_for_boolean(self):
        # create a converter
        conv = Converter()

        # log what the BOOL_VALUE_KEY is holding
        self.logger.info(f"At KEY BOOL_VALUE_KEY [{str(self.storage[BOOL_VALUE_KEY])}]")

        # convert the columns in BOOL_VALUE_KEY to True / False
        self.the_df = conv.convert_to_boolean(self.the_df, list(self.storage[BOOL_VALUE_KEY].keys()))

        # log the description
        self.logger.info("After conversion, dataframe is now\n" +
                         str(self.the_df[list(self.storage[BOOL_VALUE_KEY].keys())]))

        # flip the state flag that we've cleaned the booleans
        self.storage[BOOL_CLEANED_FLAG] = True

        # log the value of the flag.
        self.logger.info(f"BOOL_CLEANED_FLAG value set to True [{str(self.storage[BOOL_CLEANED_FLAG])}]")

        # log what the boolean key is holding
        self.logger.info(f"At KEY BOOL_COLUMN_KEY [{str(self.storage[BOOL_COLUMN_KEY])}]")

    # invoke the conversion of float data to int, if it meets the correct criteria.
    def run_conversion_from_float_2int(self):
        # log that method has been called
        self.logger.debug("A request to make convert int data masquerading as floats has been made.")

        # declare variables
        float_list = self.storage[FLOAT64_COLUMN_KEY]
        the_converter = Converter()

        # invoke the function to clean floats into INT64
        the_converter.clean_floats_that_should_be_int(self.the_df, float_list)

        # flip the flag that we have INT64 with nan
        self.storage[INT64_NAN_FLAG] = True

        # call refresh model, this updates all the data structures.
        self.refresh_model()

    # analyze the boolean variables columns, and get the counts for True and False.
    # store the results of this analysis on the key BOOL_COLUMN_COUNT_KEY.  This will
    # ultimately be displayed in the presentation layer code.
    def analyze_boolean_data(self):
        # log that we've entered the function
        self.logger.debug("Starting to perform boolean data analysis and conversion.")

        # log the size
        self.logger.debug(f"The size of the boolean column list is [{len(self.storage[BOOL_COLUMN_KEY])}]")

        # log what the boolean key is holding
        self.logger.info(f"At KEY BOOL_COLUMN_KEY [{str(self.storage[BOOL_COLUMN_KEY])}]")

        # the key BOOL_COLUMN_KEY contains a list of column names.  Loop over the list of boolean column names.
        for next_value in self.storage[BOOL_COLUMN_KEY]:
            # define the temp_dict that captures the count of True and False responses for a specific column.
            temp_dict = {'True': self.the_df[next_value].value_counts()[True],
                         'False': self.the_df[next_value].value_counts()[False]}

            # log what we are creating
            self.logger.debug(f"Column[{next_value}] boolean dict[{temp_dict}]")

            # We are going to store a complete dictionary of boolean column names, and a dictionary of column counts
            self.storage[BOOL_COLUMN_COUNT_KEY][next_value] = temp_dict

            # we also need to repopulate the unique values dict on the key UNIQUE_COLUMN_VALUES.
            # Since this is a state based implementation, it is important to note that the population of the
            # values on key UNIQUE_COLUMN_VALUES occurs during the OBJECT column analysis that occurs in
            # the analyze_columns() method.  After the boolean columns are converted to actual boolean type,
            # we will need to populate the UNIQUE_COLUMN_VALUES dict in this function.

            # populate a list on the key, in this situation the elements in the list is obvious
            self.storage[UNIQUE_COLUMN_VALUES][next_value] = [True, False]

        # log what the boolean key is holding
        self.logger.info(f"At KEY BOOL_COLUMN_KEY [{str(self.storage[BOOL_COLUMN_KEY])}]")

    # getter function to determine if this dataset has boolean data.
    def contains_boolean(self) -> bool:
        return self.has_boolean

    # refresh all the model variables
    def refresh_model(self):
        self.logger.debug("refresh_model() has been called.")

        # retrieve all the columns
        self.retrieve_columns()

        # analyze the columns
        self.analyze_columns()

    # extract boolean data
    def extract_boolean(self):
        # check if there is boolean data
        if self.contains_boolean():
            # run boolean conversion
            self.run_conversion_for_boolean()

            # capture and analyze the columns again, because we've made a conversion
            self.refresh_model()

            # analyze boolean data
            self.analyze_boolean_data()
        # by this point, we've already changed the underlying dataframe and need to make
        # sure the analyze_boolean_data() is called to populate the object graph.
        else:
            # analyze boolean data
            self.analyze_boolean_data()

    # extract int data
    def extract_int_fields(self):
        # lazy instantiation of detector
        if self.detector is None:
            self.detector = Detector()

        # check if there are int columns, mis-identified as float.
        if self.detector.detect_when_float_is_int_for_df(self.the_df):
            # run int conversion, this function refreshes the model.
            self.run_conversion_from_float_2int()

            # capture and analyze the columns again, because we've made a conversion
            self.refresh_model()

            # analyze boolean data
            self.analyze_boolean_data()

    # determine if we have integers being classified as float because there
    # are nan values embedded in the data.
    def has_ints_as_floats(self) -> bool:
        # log that we've been called.
        self.logger.debug("A request to check if we have int data masquerading as type float.")

        # variable declaration
        the_result = False

        # lazy creation of a detector
        if self.detector is None:
            self.detector = Detector()

        # get the list of floats
        float_list = self.storage[FLOAT64_COLUMN_KEY]

        # loop over each float in the list of float columns
        for current_float in float_list:
            # log the current float
            self.logger.debug(f"current_float [{current_float}]")

            # check if the float column is actually and int
            if self.detector.detect_when_float_is_int(self.the_df[current_float]):
                # set the result to True
                the_result = True

                # break out of loop
                break

        # invoke method
        return the_result

    # extract index, if needed
    def extract_index(self):
        # lazy instantiation of detector
        if self.detector is None:
            self.detector = Detector()

        # check if we need to set an index
        if not self.detector.check_if_df_has_named_indexed(self.the_df):
            # log that we need to set an index
            self.logger.debug("There is no index.")

            # get the new index
            new_index = self.detector.detect_index(self.the_df,
                                                   self.storage[UNIQUE_COLUMN_LIST_KEY])

            # set the index
            self.the_df.set_index(new_index, inplace=True)

            # rename the column to INDEX
            self.the_df.index.name = DEFAULT_INDEX_NAME

    # remove NA values, if needed.
    def remove_na_values(self):
        # lazy instantiation of detector
        if self.detector is None:
            self.detector = Detector()

        # log that we've been called
        self.logger.debug("removing nan values.")

        # variable declaration
        converter = Converter()

        # get the dictionary of columns with NA count
        column_dict = self.storage[COLUMN_NA_COUNT]

        # loop over all the column keys
        for the_column in list(column_dict.keys()):
            # log the current column
            self.logger.debug(f"removing nan from column[{the_column}]")

            # check the datatype for INT64 or FLOAT64
            if self.the_df[the_column].dtype == np.int64 or is_float_dtype(self.the_df[the_column]):

                # populate values
                converter.fill_nan_values_for_numerical_df(self.the_df)
            # check if the datatype is OBJECT
            elif is_object_dtype(self.the_df[the_column]):

                # populate values
                converter.replace_nan_with_value(self.the_df, the_column_name=the_column)

    # run full analysis and cleaning
    def run_complete_setup(self):
        # log that we're attempting to completely setup this instance
        self.logger.debug("a request to run complete setup has been made.")

        # variable declarations
        converter = Converter()

        # lazy instantiation of detector
        if self.detector is None:
            self.detector = Detector()

        # refresh the model
        self.refresh_model()

        # extract boolean fields
        self.extract_boolean()

        # extract int fields
        self.extract_int_fields()

        # check if we need to clean NA values
        self.remove_na_values()

        # check if we need to set an index
        if not self.detector.check_if_df_has_named_indexed(self.the_df):
            # log that we need to set an index
            self.logger.debug("There is no index.")

            # get the new index
            new_index = self.detector.detect_index(self.the_df, self.storage[UNIQUE_COLUMN_LIST_KEY])

            # log that we are setting a new index
            self.logger.debug(f"setting index of dataframe to [{new_index}].")

            # set the index
            self.the_df.set_index(new_index, inplace=True)

        # check if the TIMEZONE column is present.
        if TIMEZONE_COLUMN in self.the_df.columns:
            # clean time zone values
            converter.clean_timezone(self.the_df[TIMEZONE_COLUMN])

        # check for duplicates
        if self.detector.detect_if_dataframe_has_duplicates(self.the_df):
            self.the_df = converter.remove_duplicate_rows(self.the_df)

        # capture and analyze the columns again, because we've made a conversion
        self.refresh_model()

        # analyze boolean data
        self.analyze_boolean_data()

    # create a normalized dataset
    def normalize_dataset(self, overriden_df=None, column_drop_list=None):
        # log that we've been called
        self.logger.debug("A request to normalize the dataset has been made.")

        # create a converter
        converter = Converter()

        # check if a value for the_df was passed in.
        if overriden_df is None:
            # normalize the dataframe in the dsa
            self.the_normal_df = converter.get_normalized_df(self.the_df)

            # log that we're normalizing the internal dataframe
            self.logger.debug("Normalizing internal dataframe.")
        else:
            # validate the_df argument is dataframe
            if not isinstance(overriden_df, DataFrame):
                raise SyntaxError("overriden_df argument must be DataFrame.")

            # normalize the passed in dataframe
            self.the_normal_df = converter.get_normalized_df(overriden_df)

            # log that we're normalizing the overloaded argument
            self.logger.debug("Normalizing overloaded argument.")

        # check if the column_drop_list argument is not none
        if column_drop_list is not None:
            for the_column in column_drop_list:
                # append the _z_score field
                the_column = the_column + Z_SCORE

                # log the column we are going to attempt to drop
                self.logger.debug(f"Attempting to drop column [{the_column}]")

                # check if column is in the_normal_df
                if the_column in self.the_normal_df:
                    # log that the column is present
                    self.logger.debug(f"dropping column [{the_column}]")

                    # drop the column
                    self.the_normal_df = self.the_normal_df.drop(the_column, axis=1)

        # log that we are complete
        self.logger.debug("normalization of dataset is complete.")

    # get the name
    def get_name(self) -> str:
        # return the dataset name
        return self.dataset_name

    # add original dataframe reference
    def add_original_df(self, the_df: DataFrame):
        # validate that we have an argument
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The incoming argument was None or incorrect type.")

        # capture the original dataframe
        self.the_original_df = the_df

    # helper function to initialize an element on the internal storage only during the retrieve_columns()
    # function. If the key is present, do not override with empty result
    def __update_storage_key__(self, the_key, the_value):
        if the_key in self.storage:
            pass
        else:
            self.storage[the_key] = the_value

    # validate that a specific field is a member of a type
    def validate_field_type(self, the_field, the_type) -> bool:
        # perform initial validation of the_field
        if not isinstance(the_field, str):
            raise SyntaxError("the_field is None or incorrect type.")
        elif the_field not in self.the_df:
            raise SyntaxError("the_field is not present in DataFrame.")
        elif not self.is_data_type_valid(the_type):
            raise SyntaxError("the_type is None or incorrect type.")

        # variable declaration
        the_result = False

        # validate that the_field is in the_type
        if the_field in self.storage[the_type]:
            the_result = True

        # return
        return the_result

    # check if the data type is valid
    def is_data_type_valid(self, the_type: str) -> bool:
        # validate the_type
        if the_type not in DATA_TYPE_KEYS:
            return False
        else:
            return True

    # get a specific model
    def get_model(self, the_type: str) -> ModelResultBase:
        if the_type not in LM_MODEL_TYPES:
            raise SyntaxError("the_type is None or incorrect option.")

        # log that we've been called
        self.logger.debug(f"A request has been made for linear model [{the_type}].")

        # raise a RuntimeError if
        if the_type not in self.linear_model_storage:
            raise RuntimeError("the linear_model storage is not setup.")

        # return statement
        return self.linear_model_storage[the_type]

    # add model to storage
    def add_model(self, the_type: str, the_model):
        if the_type not in LM_MODEL_TYPES:
            raise SyntaxError("the_type is None or incorrect option.")
        # in order to avoid a circular reference, I cannot include an import for ModelBase
        elif the_model is None:
            raise SyntaxError("the_model is None or incorrect type.")

        if not isinstance(the_model, BaseModel):
            raise SyntaxError("the_model is None or incorrect type.")

        # log what we're storing.
        self.logger.debug(f"A request to store a model of type [{the_type}]")

        # store the model
        self.linear_model_storage[the_type] = the_model

    # clean up outliers
    def clean_up_outliers(self, model_type: str, max_p_value: float):
        # run validation on model type
        if not isinstance(model_type, str):
            raise SyntaxError("model_type is None or incorrect type.")
        elif model_type not in MT_OPTIONS:
            raise ValueError("model_type value is unknown.")
        elif not isinstance(max_p_value, float):
            raise SyntaxError("max_p_value argument is None or incorrect type.")
        elif max_p_value <= 0 or max_p_value >= 1:
            raise ValueError("max_p_value is not in (0,1).")

        # variable declaration
        the_encoded_df = None
        excluded_features_list = None
        detector = Detector()
        variable_encoder = Variable_Encoder(unencoded_df=self.the_df)

        # get the encoded dataframe based on model type
        if model_type == MT_LOGISTIC_REGRESSION or model_type == MT_LINEAR_REGRESSION:
            # encode the dataframe, dropping first
            the_encoded_df = variable_encoder.get_encoded_dataframe(drop_first=True).astype(float)
        # the encoded dataframe is for a MT_KNN_CLASSIFICATION or MT_RF_REGRESSION
        else:
            # encode the dataframe, not dropping anything
            the_encoded_df = variable_encoder.get_encoded_dataframe(drop_first=False).astype(float)

        # check if we detect outliers
        if detector.detect_if_dataset_has_outliers(the_df=the_encoded_df):
            # define excluded_features.  In the future, this will tie to distfit parameters for bimodal
            # distributions, as the MD calculation does not handle a bimodal distribution correctly.
            excluded_features_list = []

            # get outlier index list
            outlier_index_list = detector.detect_outliers_with_mcd(the_df=the_encoded_df,
                                                                   excluded_features=excluded_features_list,
                                                                   max_p_value=max_p_value)

            # log the indices of the outliers we are going to remove.
            self.logger.debug(f"removing [{len(outlier_index_list)}] outliers with index {outlier_index_list}.")

            # ****************************************************************************************
            # It is important to note that the incoming argument is a dataframe we were using for
            # identifying outliers is the encoded_df from the model. The columns on the_encoded_df
            # and self.the_df are NOT the same.  However, the rows are between the_encoded_df and
            # the self.the_df variable.
            # ****************************************************************************************

            # drop the rows on the internal dataframe.
            self.the_df.drop(outlier_index_list, axis=0, inplace=True)
        else:
            # log there were no outliers
            self.logger.debug("No outliers were detected.")

    # retrieve features of a specific type
    def retrieve_features_of_specific_type(self, the_f_type: list, ) -> list:
        # run validations
        if not isinstance(the_f_type, list):
            raise AttributeError("the_f_type is None or incorrect type.")
        for the_data_type in the_f_type:
            if the_data_type not in DATA_TYPE_KEYS:
                raise SyntaxError("a unexpected data type was found in the_f_type list.")

        # variable declaration
        the_result = []

        # loop over the_f_type
        for the_data_type in the_f_type:
            the_result.extend(self.storage[the_data_type])

        # return the list of features
        return the_result
