import copy
import logging
import warnings
import numpy as np
import scipy as sp

from pandas import Series, DataFrame, Int64Dtype
from pandas.core.dtypes.common import is_float_dtype, is_object_dtype
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from model.BaseModel import BaseModel


class Detector(BaseModel):

    # init method()
    def __init__(self):
        super().__init__()

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # log that we've been instantiated
        self.logger.debug("An instance of Detector has been created.")

    # run a test to see if a column of data is float. If it is float, then we need to check for na fields.
    # If there are NA fields in the data, then we need remove those and perform a modulo check to see if
    # the values are actually int.  If column passes the modulo int check, then return true.  Everything else, False.
    def detect_when_float_is_int(self, the_series) -> bool:
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The argument was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to detect when float are actually int has been made.")

        # variable definition
        the_result = False

        # check that the series of data is of type float and has nan values
        if is_float_dtype(the_series) and the_series.isna().sum():
            # drop the NA from the_series
            the_series = the_series.dropna()

            # perform modulo test.  The modulo of any whole number is 0.
            if (the_series % 1 == 0).all():
                # flip result to True
                the_result = True

                # log that the series is actually an int
                logging.debug("The series was found to be int.")

        # return the result
        return the_result

    # determine if dataframe has a float that is an int
    def detect_when_float_is_int_for_df(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to call detect_when_float_is_int() for a DataFrame.")

        # variable definition
        the_result = False
        the_columns = the_df.columns

        # loop over all the columns
        for the_column in the_columns:
            # log the column
            logging.debug(f"the current column is [{the_column}]")

            # invoke detect_when_float_is_int()
            the_result = self.detect_when_float_is_int(the_df[the_column])

            # break out if we find that it's true
            if the_result:
                break

        # return the result
        return the_result

    # detect if a series is an INT64 column, and if so check if it has na values inside it.
    def detect_int_with_na(self, the_series) -> bool:
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The argument was None or the wrong type.")

        # log that the method has been called
        logging.debug(f"running tests on [{the_series.name}]")

        # variable definition
        the_result = False

        # check that the series of data is of type int64 and has nan values
        if the_series.dtype == np.int64:
            # this check has to be inside another if block.
            if the_series.isna().sum():
                # drop the NA from the_series
                the_series = the_series.dropna()

                # perform modulo test.  The modulo of any whole number is 0.
                if (the_series % 1 == 0).all():
                    # flip result to True
                    the_result = True

                    # log that the series is actually an int
                    logging.debug(f"The series [{the_series.name}] was found to be int.")
                else:
                    # log that the series was not an int.
                    logging.info(f"The series [{the_series.name}] was found NOT to be int.")

        # return the result
        return the_result

    # determine if dataframe contains a column where detect_int_with_na is true
    def detect_int_with_na_for_df(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to call detect_int_with_na() for a DataFrame.")

        # variable definition
        the_result = False
        the_columns = the_df.columns

        # loop over all the columns
        for the_column in the_columns:
            # log the column
            logging.debug(f"the current column is [{the_column}]")

            # invoke detect_int_with_na()
            the_result = self.detect_int_with_na(the_df[the_column])

            # break out if we find that it's true
            if the_result:
                break

        # return the result
        return the_result

    # detect if a series is a FLOAT64 column, and if so check if it has na values inside it.
    def detect_float_with_na(self, the_series) -> bool:
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The Series argument was None or the wrong type.")

        # log that the method has been called
        logging.debug(f"running tests on [{the_series.name}]")

        # variable definition
        the_result = False

        # check that we don't explicitly have an INT64. If we've converted a field from
        # FLOAT64 to INT64, is_float_dtype() still returns true
        if not the_series.dtype == np.int64 or the_series.dtype == Int64Dtype():

            # check that the series of data is of type int64 and has nan values
            if is_float_dtype(the_series):

                # to protect against boolean and object, this has to be in another test
                if the_series.isna().sum():
                    # flip result to True
                    the_result = True

                    # log what we found
                    logging.debug(f"we found NaN values in series [{the_series.name}].")

        # return
        return the_result

    # detect if dataframe contains a FLOAT64 column with NaN
    def detect_float_with_na_for_df(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to call detect_float_with_na_for_df() for a DataFrame.")

        # variable definition
        the_result = False
        the_columns = the_df.columns

        # loop over all the columns
        for the_column in the_columns:
            # log the column
            logging.debug(f"the current column is [{the_column}]")

            # invoke detect_int_with_na()
            the_result = self.detect_float_with_na(the_df[the_column])

            # break out if we find that it's true
            if the_result:
                break

        # return the result
        return the_result

    # detect if a specific Series has a blank name
    def is_series_name_blank(self, the_series) -> bool:
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The Series was None or the wrong type.")

        # log that the method has been called
        logging.debug(f"checking if Series [{the_series.name}] has a blank name.")

        # variable declaration
        the_result = False

        # check if the name is None, this spares us from
        # handling errors in len()
        if the_series.name is None:
            # set the result to true
            the_result = True

        # check if we have an empty string, not sure when this would happen
        elif len(the_series.name) == 0:
            # set the result to true
            the_result = True

        # look for 'Unnamed'
        elif the_series.name == "Unnamed" or the_series.name == "Unnamed: 0":
            # set the result to true
            the_result = True

        # return
        return the_result

    # detect when there is a blank column name
    def are_there_blank_column_names(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to call are_there_blank_column_names() for a DataFrame.")

        # variable declaration
        the_result = False
        the_columns = the_df.columns

        # loop over all the columns
        for the_column in the_columns:
            # log the column
            logging.debug(f"the current column is [{the_column}]")

            # invoke is_series_name_blank()
            the_result = self.is_series_name_blank(the_df[the_column])

            # break if necessary
            if the_result:
                break

        # return the_result
        return the_result

    # get a list of indices for blank column names
    def get_list_of_blank_column_names(self, the_df) -> list:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("checking for blank column names.")

        # variable declaration
        the_result = []
        the_columns = the_df.columns

        # check if there are any blanks first
        if self.are_there_blank_column_names(the_df):
            # loop over all the columns
            for the_column in the_columns:
                # check if the series name is blank
                if self.is_series_name_blank(the_df[the_column]):
                    # add the index to the list
                    the_result.append(the_df[the_column].name)

        # log the final result
        logging.debug(f"the list of blank column indices is [{the_result}].")

        # return
        return the_result

    # check if additional cleaning is required.  This function is designed to
    # the business rules.
    def is_additional_cleaning_required(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # variable declaration
        the_result = False

        # log that the method has been called
        logging.debug("checking if additional cleaning is required.")

        # check if there are blank column names
        the_result = self.are_there_blank_column_names(the_df)

        # log what we found
        logging.debug(f"where there blank column names-->[{the_result}]")

        # check if not already true
        if not the_result:
            # check if there are int's with NA
            the_result = self.detect_int_with_na_for_df(the_df)

            # log what we found
            logging.debug(f"where there NaN for INT64 column-->[{the_result}]")

        # check if not already true
        if not the_result:
            # check if there are int's with NA
            the_result = self.detect_float_with_na_for_df(the_df)

            # log what we found
            logging.debug(f"where there NaN for INT64 column-->[{the_result}]")

        # return
        return the_result

    # check if field is suitable as an index
    def detect_index(self, the_df, unique_column_list) -> list:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")
        elif not self.is_valid(unique_column_list, list):
            raise SyntaxError("The unique_column_list was None or the wrong type.")

        # variable declaration
        value_list = None
        valid_column_list = []

        # loop through all the unique columns
        for the_column in unique_column_list:
            # log the current column
            logging.debug(f"current column being examined for index[{the_column}]")

            # check if the column is an integer
            if the_df[the_column].dtype == np.int64 or the_df[the_column].dtype == Int64Dtype():
                # log that we found an integer
                logging.debug(f"column[{the_column}] was found to be an integer")

                # get a list of the values in the column
                value_list = the_df.get(the_column).tolist()

                # get the length
                n = len(value_list) - 1

                # check if the list is consecutively numbered and ordered
                if sum(np.diff(sorted(value_list)) == 1) >= n:
                    # add column to list
                    valid_column_list.append(the_column)

                # look for blank column names
                if self.are_there_blank_column_names(the_df):
                    # get the blank column names
                    blank_columns = self.get_list_of_blank_column_names(the_df)

                    # check if one the elements in the blank column list is in the valid_column_list
                    for blank_column in blank_columns:
                        # check if it is in the valid column list
                        if blank_column in valid_column_list:
                            # dump the contents of valid_column_list
                            valid_column_list.clear()

                            # add the column to the list
                            valid_column_list.append(blank_column)

                            break
            else:
                # log that this is not a good candidate
                logging.debug(f"column [{the_column}] is not a good candidate.")

        # log the results of our check
        logging.debug(f"the list of valid index lists are [{valid_column_list}]")

        # return the results
        return valid_column_list

    # check if list is unique
    def detect_if_series_is_unique(self, the_series) -> bool:
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("The Series was None or the wrong type.")

        # variable declaration
        the_result = False

        # log that the method has been called
        logging.debug(f"checking if Series [{the_series.name}] is unique.")

        # check if the results are unique
        if len(the_series.unique()) == len(the_series):
            # flip to true
            the_result = True

        # return
        return the_result

    # check if dataframe is purposely indexed
    def check_if_df_has_named_indexed(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("checking if DataFrame has explicit index.")

        # variable declarations
        the_result = False

        # get the name of the current index
        the_index_name = the_df.index.name

        # log the name of the current index
        logging.debug(f"the_index_name-->[{the_index_name}]")

        # check that the index is explicitly named
        if the_index_name is not None:
            # log that the index is explicitly named
            logging.debug(f"[{the_index_name}] is an explicit index in columns.")

            # set result to True
            the_result = True
        else:
            # log that the index is not present in columns
            logging.debug(f"[{the_index_name}] is NOT present in columns.")

        # return
        return the_result

    # check if a Series of object has nan.
    def detect_object_with_na(self, the_series) -> bool:
        # run validations
        if not self.is_valid(the_series, Series):
            raise SyntaxError("the_series was None or incorrect type.")

        # log that the method has been called
        logging.debug(f"running tests on [{the_series.name}]")

        # variable definition
        the_result = False

        # check that we don't explicitly have an INT64. If we've converted a field from
        # FLOAT64 to INT64, is_float_dtype() still returns true
        if is_object_dtype(the_series):

            # to protect against boolean and object, this has to be in another test
            if the_series.isna().sum():
                # flip result to True
                the_result = True

                # log what we found
                logging.debug(f"we found NaN values in series [{the_series.name}].")

        # return
        return the_result

    # detect if dataframe contains a OBJECT column with NaN
    def detect_object_with_na_for_df(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to call detect_object_with_na_for_df() for a DataFrame.")

        # variable definition
        the_result = False
        the_columns = the_df.columns

        # loop over all the columns
        for the_column in the_columns:
            # log the column
            logging.debug(f"the current column is [{the_column}]")

            # invoke detect_object_with_na()
            the_result = self.detect_object_with_na(the_df[the_column])

            # break out if we find that it's true
            if the_result:
                break

        # return the result
        return the_result

    # detect if any fields have nan values
    def detect_na_values_for_df(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("The DataFrame was None or the wrong type.")

        # log that the method has been called
        logging.debug("A request to detect nan values for a DataFrame has been made.")

        # variable declaration
        the_result = False

        # check object first
        if self.detect_object_with_na_for_df(the_df):
            # change the result
            the_result = True

        # check for float
        elif self.detect_float_with_na_for_df(the_df):
            # change the result
            the_result = True

        # check for int
        elif self.detect_int_with_na_for_df(the_df):
            # change the result
            the_result = True

        # return
        return the_result

    # detect if a column has duplicates
    def detect_if_dataframe_has_duplicates(self, the_df) -> bool:
        # run validations
        if not self.is_valid(the_df, DataFrame):
            raise SyntaxError("the_df was None or incorrect type.")

        # log that the method has been called
        logging.debug("A request to detect if a DataFrame has duplicate rows.")

        # variable declaration
        the_result = False

        # check if there are duplicates
        if len(the_df[the_df.duplicated()]) > 0:
            the_result = True

        # return
        return the_result

    # detect outliers using Robust Mahalanobis Distance
    def detect_outliers_with_mcd(self, the_df, excluded_features=None, max_p_value=0.001) -> list:
        # run validations
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("the_df argument is None or incorrect type.")
        elif not isinstance(max_p_value, float):
            raise SyntaxError("max_p_value argument is None or incorrect type.")
        elif not (0 < max_p_value < 1.0):
            raise ValueError("max_p_value must fall between (0,1).")

        # default setting because a list is immutable
        if excluded_features is None:
            excluded_features = {}
        elif not isinstance(excluded_features, list):
            raise SyntaxError("excluded_features is incorrect type.")
        else:
            # log the features
            self.logger.debug(f"excluded_features are {excluded_features}")

        # suppress warnings
        warnings.filterwarnings('ignore', 'covariance is not symmetric positive-semidefinite.')

        # make a copy of the the_df
        the_df = copy.deepcopy(the_df)

        # iterate over the excluded_features
        for the_feature in excluded_features:
            # if the_feature is in the the_df, drop it.
            if the_feature in the_df.columns:
                the_df = the_df.drop(the_feature, axis=1)

        # log what we're doing
        self.logger.debug("A request to detect outliers with MCD has been made.")

        # Minimum covariance determinant
        rng = np.random.RandomState(0)

        real_cov = np.cov(the_df.values.T)

        X = rng.multivariate_normal(mean=np.mean(the_df, axis=0), cov=real_cov, size=506)

        cov = MinCovDet(random_state=0).fit(X)

        # robust covariance metric
        mcd = cov.covariance_

        # robust mean
        robust_mean = cov.location_

        # inverse covariance metric
        inv_covmat = sp.linalg.inv(mcd)

        # Robust M-Distance
        x_minus_mu = the_df - robust_mean

        left_term = np.dot(x_minus_mu, inv_covmat)

        mahal = np.dot(left_term, x_minus_mu.T)

        md = np.sqrt(mahal.diagonal())

        # Flag as outlier
        outlier = []

        # degrees of freedom = number of variables
        C = np.sqrt(chi2.ppf((1 - max_p_value), df=the_df.shape[1]))

        for index, value in enumerate(md):
            if value > C:
                outlier.append(index)
            else:
                continue

        return outlier

    # detect outliers using IQR
    def detect_outliers_with_iqr(self, the_df: DataFrame, the_factor: float, excluded_features=None):
        # run validations
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("the_df argument is None or incorrect type.")
        elif not isinstance(the_factor, float):
            raise SyntaxError("the_factor argument is None or incorrect type.")
        elif the_factor <= 0 or the_factor >= 2:
            raise ValueError("the_factor is not in (0,2).")

        # variable declaration
        the_result = []

        # default setting because a list is immutable
        if excluded_features is None:
            excluded_features = {}
        elif not isinstance(excluded_features, list):
            raise SyntaxError("excluded_features is incorrect type.")
        else:
            # log the features
            self.logger.debug(f"excluded_features are {excluded_features}")

            # make a copy of the the_df
            the_df = copy.deepcopy(the_df)

            # iterate over the excluded_features
            for the_feature in excluded_features:
                # if the_feature is in the the_df, drop it.
                if the_feature in the_df.columns:
                    the_df = the_df.drop(the_feature, axis=1)

        # get a list of remaining features
        the_features = the_df.columns.to_list()

        # loop over the features
        for the_feature_name in the_features:
            # log the feature name
            self.logger.debug(f"current feature [{the_feature_name}].")

            # make sure the feature in discrete or continuous
            if the_df[the_feature_name].dtype == np.int64 or the_df[the_feature_name].dtype == Int64Dtype():
                # get a series for the_feature_name
                the_series = the_df[the_feature_name]

                # run calculations
                q1 = the_series.quantile(0.25)
                q3 = the_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - the_factor * iqr
                upper_bound = q3 + the_factor * iqr

                # get the final indices of outliers
                temp_result = the_series[(the_series < lower_bound) | (the_series > upper_bound)].index

                the_result = list(set(temp_result) | set(the_result))
            # not a numerical data type
            else:
                # log that it's not a numerical datatype
                self.logger.debug(f"{the_feature_name} is not numeric and will be skipped.")

        return the_result

    # detect if we need to remove outliers
    def detect_if_dataset_has_outliers(self, the_df) -> bool:
        # run validation
        if not isinstance(the_df, DataFrame):
            raise SyntaxError("the_df argument is None or incorrect type.")

        # log what we're doing
        self.logger.debug("A request to detect if outliers need removal has been made.")

        # variable declaration
        the_result = False
        outlier_list = []

        # TODO: I need to hook the distfit result into this instead of hard coding a list of features.
        # assemble the features not to include in MCD check
        excluded_features = ['Tenure', 'Bandwidth_GB_Year']

        # make a deepcopy of the dataframe
        the_df = copy.deepcopy(the_df)

        # attempt to cast the_df features to float
        the_df = the_df.astype(float)

        # check if any results exceed tolerances of 0.001 with MCD check
        outlier_list = self.detect_outliers_with_mcd(the_df=the_df, excluded_features=excluded_features)

        # check if we received anything back in the list
        if len(outlier_list) > 0:
            # log that we need to remove outliers
            self.logger.debug("outliers were detected.")

            # flip the_result
            the_result = True

        # return statement
        return the_result
