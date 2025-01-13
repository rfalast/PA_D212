import logging
import pandas as pd
import statsmodels.api as sm

from pandas import DataFrame
from model.analysis.models.Logistic_Model_Result import Logistic_Model_Result
from model.analysis.models.ModelBase import ModelBase
from model.constants.BasicConstants import MT_LOGISTIC_REGRESSION
from util.Model_Result_Populator import Model_Result_Populator


# function to order features based on r-squared value
def order_features_by_r_squared(the_dict) -> DataFrame:
    """
    get an ordered data-frame by adjusted r_squared.
    :param the_dict:
    :return: DataFrame
    """
    # run validation
    if not isinstance(the_dict, dict):
        raise SyntaxError("the_dict argument is None or incorrect type.")

    # variable declaration
    variable_dict = {'predictor': [], 'r-squared': []}

    # loop over all the columns in the dictionary
    for the_column in list(the_dict.keys()):
        # add the_column to variable_dict
        variable_dict['predictor'].append(the_column)

        # add the r-squared value
        variable_dict['r-squared'].append(the_dict[the_column].get_pseudo_r_squared())

    # create the result
    the_result = pd.DataFrame(variable_dict).sort_values(by=['r-squared'], ascending=False).reset_index()

    # return
    return the_result


# Logistic Model tools
class Logistic_Model(ModelBase):

    # init method
    def __init__(self, dataset_analyzer):
        # call superclass
        super().__init__(dataset_analyzer)

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # log that we've been instantiated
        self.logger.debug("An instance of Logistic_Model has been created.")

        # set the model type
        self.model_type = MT_LOGISTIC_REGRESSION

    # build model for a single column
    def build_model_for_single_column(self, the_target_column, the_variable_column):
        """
        This function will build a model for the response variable "the_target_column" using the
        variable defined with the_variable_column
        :param the_target_column:
        :param the_variable_column:
        :return: Linear_Model_Result
        """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise ValueError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise SyntaxError("the_target_column argument is not in dataframe.")
        elif not isinstance(the_variable_column, str):
            raise ValueError("the_variable_column argument is None or incorrect type.")
        elif the_variable_column not in self.encoded_df:
            raise SyntaxError("the_variable_column argument is not in dataframe.")

        # log that we've been called
        self.logger.debug(f"building model for target[{the_target_column}] and variable[{the_variable_column}].")

        # invoke fit_mlr_model for only a single variable
        the_result = self.fit_model(the_target_column=the_target_column,
                                    the_variable_columns=[the_variable_column])

        # return the result
        return the_result

    # fit a model to the_target_column based on the_variable_columns
    def fit_model(self, the_target_column, the_variable_columns):
        """
        fit a multiple linear regression model to the response variable "the_target_column" using
        the features defined in the_variable_columns.
        :param the_target_column:
        :param the_variable_columns:
        :return: Linear_Model_Result
        """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise ValueError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise SyntaxError("the_target_column argument is not in dataframe.")
        elif not isinstance(the_variable_columns, list):
            raise ValueError("the_variable_columns argument is None or incorrect type.")

        # make sure everything in the_variable_columns is in self.encoded_df
        for the_column in the_variable_columns:
            if the_column not in self.encoded_df:
                raise SyntaxError("an element in the_variable_columns argument is not in dataframe.")

        # log that we've been called
        self.logger.debug(f"A request was made to build a model for y[{the_target_column}] "
                          f"and x {the_variable_columns}.")

        # first, make sure the_target_column is not in the_variable_columns
        if the_target_column in the_variable_columns:
            # log that we are removing the_target_column from the_variable_columns
            self.logger.info(f"removing [{the_target_column}] from the_variable_columns{the_variable_columns}")

            # remove the_target_column from the_variable_columns
            the_variable_columns.remove(the_target_column)

        # variable declaration
        the_target_df = self.encoded_df[the_target_column].astype(int)
        the_variable_df = self.encoded_df[the_variable_columns].astype(int)
        the_result = None

        # get the constant from the_variable_df
        x_con = sm.add_constant(the_variable_df)

        # get the Logit
        logistic_regression = sm.Logit(the_target_df, x_con)

        # get the fitted model
        fitted_model = logistic_regression.fit()

        # instantiate an instance of Model_Result_Populator
        mrp = Model_Result_Populator()

        # populate storage
        mrp.populate_storage(the_key="the_model", the_item=fitted_model)
        mrp.populate_storage(the_key="the_target_variable", the_item=the_target_column)
        mrp.populate_storage(the_key="the_variables_list", the_item=the_variable_columns)
        mrp.populate_storage(the_key="the_df", the_item=the_variable_df)

        # create Linear_Model_Result
        the_result = Logistic_Model_Result(argument_dict=mrp.get_storage())

        # store the result
        self.the_result = the_result

        # return statement
        return the_result

    # build model for single comparison
    def build_model_for_single_comparison(self, the_target_column) -> dict:
        """
        build a linear regression models using a single feature for the response variable defined
        with the_target_column.  The dictionary returned will have a key of the feature, and the
        value of the dict will be a Linear_Model_Result.
        :param the_target_column:
        :return: dict where key is feature name, and value is Linear_Model_Result
        """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise ValueError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise SyntaxError("the_target_column argument is not in dataframe.")

        # log that we've been called
        self.logger.debug(f"A request to perform pairwise comparison between [{the_target_column}] and dataset.")

        # variable declaration
        the_result = {}
        the_variable_list = self.get_encoded_variables(the_target_column)

        # log the variable list
        self.logger.debug(f"the_variable_list->{the_variable_list}")

        # loop through all the variables
        for the_variable in the_variable_list:
            # add the result to the dictionary
            the_result[the_variable] = self.build_model_for_single_column(the_target_column=the_target_column,
                                                                          the_variable_column=the_variable)

        # return statement
        return the_result

    # select the next feature
    def select_next_feature(self, the_target_column, current_features, ignore_features=None) -> dict:
        """
        select the next feature to use to model the response in "the_target_column"
        :param the_target_column:
        :param current_features:
        :param ignore_features:
        :return: variable_dict with key of feature, and r-squared as value
        """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise SyntaxError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise ValueError("the_target_column argument is not in dataframe.")
        elif not isinstance(current_features, list):
            raise SyntaxError("current_features argument is None or incorrect type.")

        # make sure everything in the_variable_columns is in self.encoded_df
        for the_column in current_features:
            if the_column not in self.encoded_df:
                raise ValueError("an element in current_features argument is not in dataframe.")

        # variable declaration
        the_variable_list = self.get_encoded_variables(the_target_column)
        the_result = {}

        # determine if we received an argument for ignore_features
        if ignore_features is None:
            ignore_features = []

        # loop over every column in the_variable_list
        for the_column in the_variable_list:

            # But only create a model if the feature isn't already selected or ignored
            # make sure the current column isn't in 'current_features' or 'ignore_features'
            if the_column not in (current_features + ignore_features):
                # Add the column name to our dictionary
                the_result[the_column] = self.fit_model(the_target_column=the_target_column,
                                                        the_variable_columns=current_features + [the_column])

        # return
        return the_result

