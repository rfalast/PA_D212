import logging
import pandas as pd

from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor
from model.BaseModel import BaseModel
from model.constants.ModelConstants import LM_PREDICTOR, LM_FEATURE_NUM, LM_COEFFICIENT, LM_STANDARD_ERROR, \
    LM_T_STATISTIC, LM_P_VALUE, LM_LS_CONF_INT, LM_RS_CONF_INT, LM_VIF, LM_CONSTANT
from util.CSV_loader import CSV_Loader


class ModelResultBase(BaseModel):

    # init method
    def __init__(self, the_target_variable: str, the_variables_list: list, the_encoded_df: DataFrame):
        # initialize superclass
        super().__init__()

        # run validations
        if not isinstance(the_target_variable, str):
            raise AttributeError("the_target_variable is None or incorrect type.")
        elif not isinstance(the_variables_list, list):
            raise AttributeError("the_variables_list is None or incorrect type.")
        elif not isinstance(the_encoded_df, DataFrame):
            raise AttributeError("the_encoded_df is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        self.the_target_variable = the_target_variable
        self.the_variables_list = the_variables_list

        # loop through the_encoded_df.columns to make sure the columns in the_variables_list are present
        for the_column in self.the_variables_list:
            if the_column not in the_encoded_df:
                raise ValueError(f"[{the_column}] not present in the_encoded_df.")

        # define internal variables
        self.the_df = the_encoded_df
        self.assumptions = {}
        self.model = None
        self.feature_columns = None

        # check the type, don't call populate_assumptions() if instance of ModelResultBase
        if type(self).__name__ != 'ModelResultBase':
            # for a population of assumptions
            self.populate_assumptions()

    # getter method for the_target_variable property
    def get_the_target_variable(self) -> str:
        return self.the_target_variable

    # getter method for the the_variables_list property
    def get_the_variables_list(self) -> list:
        return self.the_variables_list

    # get p-values
    def get_the_p_values(self, less_than=1.00) -> dict:
        # validate arguments
        if less_than <= 0 or less_than > 1.00:
            raise ValueError("less_than must be in range (0, 1.0]")

        # create a result dictionary
        the_result = {}

        # get a series
        p_values = round(self.model.pvalues[self.the_variables_list].astype(float), 5)

        # loop over all the variables
        for the_column in self.the_variables_list:
            # make sure the p_value is <= less_than argument
            if p_values.loc[the_column] <= less_than:
                # add to results
                the_result[the_column] = p_values.loc[the_column]

        # loop through the
        return the_result

    # get the AIC / BIC values
    def get_aic_bic_for_model(self) -> tuple:
        the_result = (('AIC', self.model.aic), ('BIC', self.model.bic))

        # return statement
        return the_result

    # get the VIF for current variables
    def get_vif_for_model(self, the_encoded_df: DataFrame) -> DataFrame:
        # run validations
        if not isinstance(the_encoded_df, DataFrame):
            raise SyntaxError("the_encoded_df is None or incorrect type.")

        # loop through the_encoded_df.columns to make sure the columns in the_variables_list are present
        for the_column in self.the_variables_list:
            if the_column not in the_encoded_df:
                raise ValueError("["+the_column+"] not present in the_encoded_df.")

        # Define an empty dataframe to capture the VIF scores
        vif_2 = pd.DataFrame()

        # Label the scores with their related columns
        vif_2["features"] = the_encoded_df.columns

        # For each column, run a variance_inflation_factor against all other columns to get a VIF Factor score
        vif_2["VIF"] = [variance_inflation_factor(the_encoded_df.values, i) for i in range(len(the_encoded_df.columns))]

        return vif_2

    # get results dataframe
    def get_results_dataframe(self, round_value=10) -> DataFrame:
        # run validation of arguments
        if not isinstance(round_value, int):
            raise SyntaxError("round_value is None or incorrect type.")
        elif round_value < 1:
            raise ValueError("round_value cannot be below one.")

        # variable declaration
        the_dict = self.get_the_p_values()          # get a dict of p-values
        predictor_list = self.the_variables_list    # get a list of all the predictor names
        p_value_list = []                           # create the p-value list
        coefficient_list = []                       # create the coefficient list
        counter_list = []                           # create a list of feature count, for display purposes
        standard_error_list = []                    # create a list of standard error for each coefficient
        t_statistic_list = []                       # create a list of t-statistic for each coefficient
        fitted_model_left_list = []                 # create a list of left confidence intervals for each coefficient
        fitted_model_right_list = []                # create a list of left confidence intervals for each coefficient
        vif_list = []                               # list of VIF results
        the_results_dict = {LM_FEATURE_NUM: [], LM_PREDICTOR: [], LM_COEFFICIENT: [], LM_STANDARD_ERROR: [],
                            LM_T_STATISTIC: [], LM_P_VALUE: [], LM_LS_CONF_INT: [], LM_RS_CONF_INT: [], LM_VIF: []}

        counter = 1
        the_vif_df = self.get_vif_for_model(the_encoded_df=self.the_df)

        # get a list of all the p-values
        for the_predictor in predictor_list:
            # add the p-value to the p_value_list
            p_value_list.append(the_dict[the_predictor])

            # add the coefficient to the coefficient_list
            coefficient_list.append(self.model.params.loc[the_predictor].round(round_value))

            # add the standard error to the standard_error_list
            standard_error_list.append(float(self.model.bse.loc[the_predictor].round(round_value)))

            # add the t-statistic to the t_statistic_list
            t_statistic_list.append(self.model.tvalues.loc[the_predictor].round(round_value))

            # add the '[0.025' values
            fitted_model_left_list.append(self.model.conf_int()[0].loc[the_predictor].round(round_value))

            # add the '0.975]' values
            fitted_model_right_list.append(self.model.conf_int()[1].loc[the_predictor].round(round_value))

            # add the VIF to the vif_list
            vif_list.append(the_vif_df.loc[the_vif_df['features'] == the_predictor]['VIF'].iloc[0].round(round_value))

            # add the counter to the counter_list
            counter_list.append(counter)

            # increment counter
            counter = counter + 1

        # add the lists to the the_results_dict
        the_results_dict[LM_FEATURE_NUM] = counter_list
        the_results_dict[LM_PREDICTOR] = predictor_list
        the_results_dict[LM_COEFFICIENT] = coefficient_list             # the index
        the_results_dict[LM_STANDARD_ERROR] = standard_error_list
        the_results_dict[LM_T_STATISTIC] = t_statistic_list
        the_results_dict[LM_P_VALUE] = p_value_list
        the_results_dict[LM_LS_CONF_INT] = fitted_model_left_list
        the_results_dict[LM_RS_CONF_INT] = fitted_model_right_list
        the_results_dict[LM_VIF] = vif_list

        # create the final dataframe
        the_result = pd.DataFrame(the_results_dict)

        # set the index to be 'predictor'
        the_result.set_index(LM_PREDICTOR, inplace=True)

        # capture the columns
        self.feature_columns = the_result.columns.to_list()

        # return
        return the_result

    # get the constant
    def get_constant(self) -> float:
        return self.model.params[LM_CONSTANT]

    # check if p-value(s) are above a level
    def are_p_values_above_threshold(self, p_value=1.0) -> bool:
        if not isinstance(p_value, float):
            raise SyntaxError("p_value was None or incorrect type.")
        elif p_value > 1.0:
            raise ValueError("p_value was greater than 1.0")

        # variable declaration
        the_result = False
        the_result_df = self.get_results_dataframe()
        the_feature_list = self.the_variables_list

        # loop over all the features
        for the_feature in the_feature_list:
            # check if the feature is above the p_value
            if the_result_df.loc[the_feature, LM_P_VALUE] > p_value:
                # set the_result to True
                the_result = True

                # log the feature
                self.logger.debug(f"the_feature{the_feature}] has p_value["
                                  f"{the_result_df.loc[the_feature, LM_P_VALUE]}] above [{p_value}]")

                # break out of for loop
                break

        # return
        return the_result

    # check if VIF is above a threshold
    def is_vif_above_threshold(self, max_allowable_vif=5.0) -> bool:
        if not isinstance(max_allowable_vif, float):
            raise SyntaxError("max_allowable_vif was None or incorrect type.")
        elif max_allowable_vif < 1.0:
            raise ValueError("max_allowable_vif was less than 1.0")

        # variable declaration
        the_result = False
        the_result_df = self.get_results_dataframe()
        the_feature_list = self.the_variables_list

        # loop over all the features
        for the_feature in the_feature_list:
            # check if the feature is above the p_value
            if the_result_df.loc[the_feature, LM_VIF] > max_allowable_vif:
                # set the_result to True
                the_result = True

                # break out of for loop
                break

        # return
        return the_result

    # identify a parameter to remove
    def identify_parameter_based_on_p_value(self, p_value=1.0) -> str:
        if not isinstance(p_value, float):
            raise SyntaxError("p_value was None or incorrect type.")
        elif p_value > 1.0:
            raise ValueError("p_value was greater than 1.0")

        # variable declaration
        the_result = None
        the_result_df = self.get_results_dataframe()

        # select the maximum p-value
        the_result = the_result_df.sort_values(LM_P_VALUE, ascending=False).index[0]

        # return statement
        return the_result

    # get the feature with the max VIF
    def get_feature_with_max_vif(self) -> str:
        # log that we've been called
        self.logger.debug("A request to retrieve the feature with the maximum VIF has been made.")

        # variable declaration
        the_result = None

        # get the results dataframe
        the_results_df = self.get_results_dataframe()

        # get the name of the feature with the largest VIF
        the_result = the_results_df['VIF'].idxmax()

        # log the result
        self.logger.debug(f"the feature with the largest VIF is [{the_result}]")

        # return
        return the_result

    # get the AIC for model.
    def get_aic_for_model(self) -> float:
        return self.get_aic_bic_for_model()[0][1]

    # get the BIC for model.
    def get_bic_for_model(self) -> float:
        return self.get_aic_bic_for_model()[1][1]

    # get the number of observations
    def get_number_of_obs(self) -> int:
        return self.model.nobs

    # get the degrees of freedom for model
    def get_degrees_of_freedom_for_model(self) -> int:
        return self.model.df_model

    # get the degrees of freedom for residuals
    def get_degrees_of_freedom_for_residuals(self) -> int:
        return self.model.df_resid

    # get assumptions
    def get_assumptions(self) -> dict:
        # return
        return self.assumptions

    # populate assumptions. Implemented in subclass
    def populate_assumptions(self):
        raise NotImplementedError

    # method to get assumption result
    def get_assumption_result(self, the_assumption: str):
        # argument validation
        if not isinstance(the_assumption, str):
            raise SyntaxError("the_assumption is None or incorrect type.")
        elif the_assumption not in self.assumptions:
            raise KeyError("the_assumption is not present in storage.")

        # check if the_assumption is implemented.
        try:
            # check if the function is present
            the_method = getattr(self, self.assumptions[the_assumption])

            # check if it is callable in this context.
            if callable(the_method):
                the_result = the_method()
            else:
                raise NotImplementedError("the_assumption is not implemented.")
        except NotImplementedError as err:
            raise err

        # return
        return the_result

    # method to check if a model result has an assumption
    def has_assumption(self, the_assumption: str) -> bool:
        # argument validation
        if not isinstance(the_assumption, str):
            raise SyntaxError("the_assumption is None or incorrect type.")

        # define the_result
        the_result = False

        # check if the model result has the assumption
        if the_assumption in self.assumptions:
            the_result = True

        # return
        return the_result

    # get accuracy_score
    def get_accuracy_score(self) -> float:
        raise NotImplementedError

    # get feature columns
    def get_feature_columns(self) -> list:
        return self.feature_columns

    # persist CSV files for this model.  This is implemented in child classes.
    def generate_model_csv_files(self, csv_loader: CSV_Loader):
        raise NotImplementedError
