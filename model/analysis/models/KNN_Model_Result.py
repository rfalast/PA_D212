import logging
import numpy
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import CHURN_FINAL, D_209_CHURN, CHURN_X_TRAIN, CHURN_X_TEST, CHURN_Y_TRAIN, \
    CHURN_Y_TEST, CHURN_PREP
from model.constants.ModelConstants import LM_FEATURE_NUM, LM_PREDICTOR, LM_P_VALUE
from model.constants.ReportConstants import NUMBER_OF_OBS, MODEL_ACCURACY, MODEL_PRECISION, MODEL_RECALL, \
    MODEL_F1_SCORE, MODEL_AVG_PRECISION, MODEL_ROC_SCORE, MODEL_Y_PRED, MODEL_Y_SCORES, MODEL_BEST_SCORE, \
    MODEL_BEST_PARAMS
from model.converters.Converter import Converter
from util.CSV_loader import CSV_Loader
from util.CommonUtils import convert_dict_to_str


# KNN Model result
class KNN_Model_Result(ModelResultBase):

    # init method. Note to self; I need to change the method interface to take a complex object or
    # dictionary with defined keys. The current exposed signature is a complete mess.
    def __init__(self, the_model: KNeighborsClassifier, the_target_variable: str, the_variables_list: list,
                 the_f_df_train: DataFrame, the_f_df_test: DataFrame, the_t_var_train: DataFrame,
                 the_t_var_test: DataFrame, the_encoded_df: DataFrame, the_p_values: DataFrame,
                 gridsearch: GridSearchCV, prepared_data: DataFrame, cleaned_data: DataFrame):
        # call super class
        super().__init__(the_target_variable=the_target_variable,
                         the_variables_list=the_variables_list,
                         the_encoded_df=the_encoded_df)

        # run validations not handled by super-class.
        if not isinstance(the_model, KNeighborsClassifier):
            raise SyntaxError("the_model is None or incorrect type.")
        elif not isinstance(the_f_df_train, DataFrame):
            raise SyntaxError("the_f_df_train is None or incorrect type.")
        elif not isinstance(the_f_df_test, DataFrame):
            raise SyntaxError("the_f_df_test is None or incorrect type.")
        elif not isinstance(the_t_var_train, DataFrame):
            raise SyntaxError("the_t_var_train is None or incorrect type.")
        elif not isinstance(the_t_var_test, DataFrame):
            raise SyntaxError("the_t_var_test is None or incorrect type.")
        elif not isinstance(the_p_values, DataFrame):
            raise SyntaxError("the_p_values is None or incorrect type.")
        elif not isinstance(gridsearch, GridSearchCV):
            raise SyntaxError("gridsearch is None or incorrect type.")
        elif not isinstance(prepared_data, DataFrame):
            raise SyntaxError("prepared_data is None or incorrect type.")
        elif not isinstance(cleaned_data, DataFrame):
            raise SyntaxError("cleaned_data is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # define  internal variables
        self.model = the_model
        self.y_pred = the_model.predict(the_f_df_test)
        self.y_scores = the_model.predict_proba(the_f_df_test)
        self.the_f_df_test = the_f_df_test                      # X_test
        self.the_f_df_train = the_f_df_train                    # X_train
        self.the_t_var_test = the_t_var_test                    # Y_test
        self.the_t_var_train = the_t_var_train                  # Y_train
        self.the_p_values = the_p_values
        self.grid_search = gridsearch
        self.prepared_data = prepared_data
        self.cleaned_data = cleaned_data

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
    def get_model(self) -> KNeighborsClassifier:
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

    # get the accuracy for model
    def get_accuracy(self) -> float:
        return metrics.accuracy_score(self.the_t_var_test.values.ravel(), self.y_pred)

    # get the average precision
    def get_avg_precision(self) -> float:
        return metrics.average_precision_score(self.the_t_var_test.values.ravel(), self.y_pred)

    # get the f1 score
    def get_f1_score(self) -> float:
        return metrics.f1_score(self.the_t_var_test.values.ravel(), self.y_pred)

    # get the precision
    def get_precision(self) -> float:
        return metrics.precision_score(self.the_t_var_test.values.ravel(), self.y_pred)

    # get the recall
    def get_recall(self) -> float:
        return metrics.recall_score(self.the_t_var_test.values.ravel(), self.y_pred)

    # get the roc score
    def get_roc_score(self) -> float:
        return metrics.roc_auc_score(self.the_t_var_test.values.ravel(), self.y_pred)

    # the model best score
    def get_model_best_score(self) -> float:
        return self.grid_search.best_score_

    # get the best params.
    def get_best_params(self) -> str:
        # retrieve the params dict from the
        return convert_dict_to_str(self.grid_search.best_params_)

    # populate assumptions
    def populate_assumptions(self):
        # add keys to assumptions dict
        self.assumptions[MODEL_ACCURACY] = 'get_accuracy'
        self.assumptions[MODEL_AVG_PRECISION] = 'get_avg_precision'
        self.assumptions[MODEL_F1_SCORE] = 'get_f1_score'
        self.assumptions[MODEL_PRECISION] = 'get_precision'
        self.assumptions[MODEL_RECALL] = 'get_recall'
        self.assumptions[MODEL_ROC_SCORE] = 'get_roc_score'
        self.assumptions[MODEL_Y_PRED] = 'get_y_predicted'
        self.assumptions[MODEL_Y_SCORES] = 'get_y_scores'
        self.assumptions[NUMBER_OF_OBS] = 'get_number_of_obs'
        self.assumptions[MODEL_BEST_SCORE] = 'get_model_best_score'
        self.assumptions[MODEL_BEST_PARAMS] = 'get_best_params'

    # get the predicted values for y
    def get_y_predicted(self) -> numpy.ndarray:
        return self.y_pred

    # get the y_scores
    def get_y_scores(self):
        return self.y_scores

    # get the number of observations
    def get_number_of_obs(self) -> int:
        return len(self.the_f_df_test)

    # get a confusion matrix
    def get_confusion_matrix(self) -> ndarray:
        # log that we've been called
        self.logger.debug("A request to generate a confusion matrix has been made.")

        # declare the_result
        the_result = None

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
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_FINAL, the_dataframe=self.cleaned_data)

        # write out CHURN_PREP, the smaller dataset after feature selection and scaling
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_PREP, the_dataframe=self.prepared_data)

        # write out CHURN_X_TRAIN, converting the np.ndarray back to a dataframe.
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_X_TRAIN, the_dataframe=self.the_f_df_train)

        # write out CHURN_X_TRAIN, converting the np.ndarray back to a dataframe.
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_X_TEST, the_dataframe=self.the_f_df_test)

        # write out CHURN_Y_TRAIN  the_t_var_train
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_Y_TRAIN, the_dataframe=self.the_t_var_train)

        # write out CHURN_Y_TEST  the_t_var_test
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_Y_TEST, the_dataframe=self.the_t_var_test)
