import logging
import numpy
import pandas as pd

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import CHURN_FINAL, D_212_CHURN, CHURN_X_TRAIN, CHURN_X_TEST, CHURN_Y_TRAIN, \
    CHURN_Y_TEST, CHURN_PREP
from model.constants.ModelConstants import LM_FEATURE_NUM, LM_PREDICTOR, LM_P_VALUE
from model.constants.ReportConstants import NUMBER_OF_OBS, MODEL_Y_PRED, MODEL_BEST_SCORE, \
    MODEL_BEST_PARAMS, MODEL_MEAN_ABSOLUTE_ERROR, MODEL_MEAN_SQUARED_ERROR, MODEL_ROOT_MEAN_SQUARED_ERROR, \
    R_SQUARED_HEADER
from util.CSV_loader import CSV_Loader
from util.CommonUtils import convert_dict_to_str, convert_series_to_dataframe


# Random Forest Model result
class Random_Forest_Model_Result(ModelResultBase):

    # init method. Note to self; I need to change the method interface to take a complex object or
    # dictionary with defined keys. The current exposed signature is a complete mess.
    def __init__(self, argument_dict: dict):
        # run initial validation
        if not isinstance(argument_dict, dict):
            raise AttributeError("argument_dict is None or incorrect type.")
        elif "the_model" not in argument_dict:
            raise AttributeError("the_model is missing.")
        elif "the_target_variable" not in argument_dict:
            raise AttributeError("the_target_variable is missing.")
        elif "the_variables_list" not in argument_dict:
            raise AttributeError("the_variables_list is missing.")
        elif "the_f_df_train" not in argument_dict:
            raise AttributeError("the_f_df_train is missing.")
        elif "the_f_df_test" not in argument_dict:
            raise AttributeError("the_f_df_test is missing.")
        elif "the_t_var_train" not in argument_dict:
            raise AttributeError("the_t_var_train is missing.")
        elif "the_t_var_test" not in argument_dict:
            raise AttributeError("the_t_var_test is missing.")
        elif "the_encoded_df" not in argument_dict:
            raise AttributeError("the_encoded_df is missing.")
        elif "the_p_values" not in argument_dict:
            raise AttributeError("the_p_values is missing.")
        elif "gridsearch" not in argument_dict:
            raise AttributeError("gridsearch is missing.")
        elif "prepared_data" not in argument_dict:
            raise AttributeError("prepared_data is missing.")
        elif "cleaned_data" not in argument_dict:
            raise AttributeError("cleaned_data is missing.")

        # call super class
        super().__init__(the_target_variable=argument_dict['the_target_variable'],
                         the_variables_list=argument_dict['the_variables_list'],
                         the_encoded_df=argument_dict['the_encoded_df'])

        # run validations not handled by super-class.
        if not isinstance(argument_dict['the_model'], RandomForestRegressor):
            raise AttributeError("the_model is None or incorrect type.")
        elif not isinstance(argument_dict['the_f_df_train'], DataFrame):
            raise AttributeError("the_f_df_train is None or incorrect type.")
        elif not isinstance(argument_dict['the_f_df_test'], DataFrame):
            raise AttributeError("the_f_df_test is None or incorrect type.")
        elif not isinstance(argument_dict['the_t_var_train'], Series):
            raise AttributeError("the_t_var_train is None or incorrect type.")
        elif not isinstance(argument_dict['the_t_var_test'], Series):
            raise AttributeError("the_t_var_test is None or incorrect type.")
        elif not isinstance(argument_dict['the_p_values'], DataFrame):
            raise AttributeError("the_p_values is None or incorrect type.")
        elif not isinstance(argument_dict['gridsearch'], GridSearchCV):
            raise AttributeError("gridsearch is None or incorrect type.")
        elif not isinstance(argument_dict['prepared_data'], DataFrame):
            raise AttributeError("prepared_data is None or incorrect type.")
        elif not isinstance(argument_dict['cleaned_data'], DataFrame):
            raise AttributeError("cleaned_data is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # define  internal variables
        self.model = argument_dict['the_model']
        self.y_pred = self.model.predict(argument_dict['the_f_df_test'])
        self.the_f_df_test = argument_dict['the_f_df_test']                      # X_test
        self.the_f_df_train = argument_dict['the_f_df_train']                    # X_train
        self.the_t_var_test = argument_dict['the_t_var_test']                    # Y_test
        self.the_t_var_train = argument_dict['the_t_var_train']                  # Y_train
        self.the_p_values = argument_dict['the_p_values']
        self.grid_search = argument_dict['gridsearch']
        self.prepared_data = argument_dict['prepared_data']
        self.cleaned_data = argument_dict['cleaned_data']

    # get p-values in the form of a dict.  The key is the name of the feature, the value is the
    # actual p_value associated with the feature.  If the argument less_than is used, you will end
    # up only with result set with of features where the p_value is strictly < the value of less_than.
    def get_the_p_values(self, less_than=1.00) -> dict:
        # validate arguments
        if less_than <= 0 or less_than > 1.00:
            raise ValueError("less_than argument must be in range (0, 1.0]")

        # create a result dictionary
        the_result = {}

        # the initial data structure we have captured in self.the_p_values consists of
        # 1) the index set to the original feature number
        # 2) the feature name
        # 3) the un-rounded p_value

        # loop through all the elements of 'Feature'
        for i, v in self.the_p_values['Feature'].items():
            # make sure the p_value is <= less_than argument
            if self.the_p_values['p_value'][i] <= less_than:
                # add to results
                the_result[v] = round(self.the_p_values['p_value'][i], 6)

        # return the_result
        return the_result

    # getter method for model property
    def get_model(self) -> RandomForestRegressor:
        return self.model

    # get results dataframe
    def get_results_dataframe(self, round_value=10) -> DataFrame:
        # run validation of arguments
        if not isinstance(round_value, int):
            raise SyntaxError("round_value is None or incorrect type.")
        elif round_value < 1:
            raise ValueError("round_value cannot be below one.")

        # the purpose of this function is to create a dataframe that matches what we expect to see.

        # variable declaration
        the_dict = self.get_the_p_values()  # get a dict of p-values
        predictor_list = self.the_variables_list  # get a list of all the predictor names
        p_value_list = []  # create the p-value list
        counter_list = []  # create a list of feature count, for display purposes
        the_results_dict = {LM_FEATURE_NUM: [], LM_PREDICTOR: [], LM_P_VALUE: []}

        counter = 1

        # get a list of all the p-values
        for the_predictor in predictor_list:
            # add the p-value to the p_value_list
            p_value_list.append(the_dict[the_predictor])

            # add the counter to the counter_list
            counter_list.append(counter)

            # increment counter
            counter = counter + 1

        # add the lists to the the_results_dict
        the_results_dict[LM_FEATURE_NUM] = counter_list
        the_results_dict[LM_PREDICTOR] = predictor_list
        the_results_dict[LM_P_VALUE] = p_value_list

        # create the final dataframe
        the_result = pd.DataFrame(the_results_dict)

        # set the index to be 'predictor'
        the_result.set_index(LM_PREDICTOR, inplace=True)

        # capture the columns
        self.feature_columns = the_result.columns.to_list()

        # return
        return the_result

    # the model best score
    def get_model_best_score(self) -> float:
        return -self.grid_search.best_score_

    # get the best params.
    def get_best_params(self) -> str:
        # retrieve the params dict from the
        return convert_dict_to_str(self.grid_search.best_params_)

    # get the mean absolute error
    def get_mean_absolute_error(self) -> float:
        return metrics.mean_absolute_error(self.the_t_var_test, self.y_pred)

    def get_mean_squared_error(self) -> float:
        return metrics.mean_squared_error(self.the_t_var_test, self.y_pred)

    def get_root_mean_squared_error(self) -> float:
        return metrics.root_mean_squared_error(self.the_t_var_test, self.y_pred)

    def get_r_squared(self) -> float:
        return metrics.r2_score(self.the_t_var_test, self.y_pred)

    # populate assumptions
    def populate_assumptions(self):
        # add keys to assumptions dict
        self.assumptions[MODEL_MEAN_ABSOLUTE_ERROR] = 'get_mean_absolute_error'
        self.assumptions[MODEL_MEAN_SQUARED_ERROR] = 'get_mean_squared_error'
        self.assumptions[MODEL_ROOT_MEAN_SQUARED_ERROR] = 'get_root_mean_squared_error'
        self.assumptions[R_SQUARED_HEADER] = 'get_r_squared'
        self.assumptions[MODEL_Y_PRED] = 'get_y_predicted'
        self.assumptions[NUMBER_OF_OBS] = 'get_number_of_obs'
        self.assumptions[MODEL_BEST_SCORE] = 'get_model_best_score'
        self.assumptions[MODEL_BEST_PARAMS] = 'get_best_params'

    # get the predicted values for y
    def get_y_predicted(self) -> numpy.ndarray:
        return self.y_pred

    # get the number of observations
    def get_number_of_obs(self) -> int:
        return len(self.the_f_df_test)

    # get a confusion matrix
    def get_confusion_matrix(self) -> ndarray:
        # log that we've been called
        self.logger.debug("A request to generate a confusion matrix has been made.")

        # declare the_result
        c_matrix = None

        if isinstance(self.model, RandomForestRegressor):
            # log that we can't build a confusion matrix
            self.logger.debug("Cannot build confusion matrix for non-classification model.")
        else:
            # log that we're building a confusion matrix
            self.logger.debug(f"building confusion matrix for {type(self.model)}.")

            # create the confusion matrix
            c_matrix = metrics.confusion_matrix(y_true=self.the_t_var_test, y_pred=self.y_pred)

        # return
        return c_matrix

    # persist CSV files for this model.
    def generate_model_csv_files(self, csv_loader: CSV_Loader):
        # log what we are doing
        self.logger.debug("exporting CSV files for KNN model.")

        # validate arguments
        if not isinstance(csv_loader, CSV_Loader):
            raise SyntaxError("csv_loader is None or incorrect type.")

        # there are CSV files that this model will generate
        # CHURN_PREP -> "resources/Output/churn_prepared.csv"
        # CHURN_FINAL -> "resources/Output/churn_cleaned.csv"
        # CHURN_X_TRAIN -> "resources/Output/churn_X_train.csv"
        # CHURN_X_TEST -> "resources/Output/churn_X_test.csv"
        # CHURN_Y_TRAIN -> "resources/Output/churn_Y_train.csv"
        # CHURN_Y_TEST -> "resources/Output/churn_Y_test.csv"

        # write out CHURN_FINAL, the cleaned data set
        csv_loader.generate_output_file(data_set=D_212_CHURN, option=CHURN_FINAL, the_dataframe=self.cleaned_data)

        # write out CHURN_PREP, the smaller dataset after feature selection and scaling
        csv_loader.generate_output_file(data_set=D_212_CHURN, option=CHURN_PREP, the_dataframe=self.prepared_data)

        # write out CHURN_X_TRAIN, converting the np.ndarray back to a dataframe.
        csv_loader.generate_output_file(data_set=D_212_CHURN, option=CHURN_X_TRAIN, the_dataframe=self.the_f_df_train)

        # write out CHURN_X_TRAIN, converting the np.ndarray back to a dataframe.
        csv_loader.generate_output_file(data_set=D_212_CHURN, option=CHURN_X_TEST, the_dataframe=self.the_f_df_test)

        # write out CHURN_Y_TRAIN  the_t_var_train
        csv_loader.generate_output_file(data_set=D_212_CHURN,
                                        option=CHURN_Y_TRAIN,
                                        the_dataframe=convert_series_to_dataframe(the_series=self.the_t_var_train))

        # write out CHURN_Y_TEST  the_t_var_test
        csv_loader.generate_output_file(data_set=D_212_CHURN,
                                        option=CHURN_Y_TEST, 
                                        the_dataframe=convert_series_to_dataframe(the_series=self.the_t_var_test))
