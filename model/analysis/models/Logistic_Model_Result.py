import logging
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import D_212_CHURN, CHURN_FINAL, CHURN_PREP
from model.constants.ReportConstants import MODEL_ACCURACY, MODEL_PRECISION, MODEL_RECALL, \
    MODEL_F1_SCORE, PSEUDO_R_SQUARED_HEADER, AIC_SCORE, BIC_SCORE, LOG_LIKELIHOOD, NUMBER_OF_OBS, \
    DEGREES_OF_FREEDOM_MODEL, DEGREES_OF_FREEDOM_RESID, MODEL_CONSTANT
from util.CSV_loader import CSV_Loader


# Linear Model tools
class Logistic_Model_Result(ModelResultBase):

    # init method
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
        elif "the_df" not in argument_dict:
            raise AttributeError("the_df is missing.")

        # call super
        super().__init__(the_target_variable=argument_dict["the_target_variable"],
                         the_variables_list=argument_dict["the_variables_list"],
                         the_encoded_df=argument_dict["the_df"])

        # run validation of the_regression_wrapper.
        if not isinstance(argument_dict["the_model"], BinaryResultsWrapper):
            raise SyntaxError("the_model is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # define and capture internal variables
        self.model = argument_dict["the_model"]

        # define additional variables
        self.accuracy_score = 0.0
        self.precision_score = 0.0
        self.recall_score = 0.0
        self.f1_score = 0.0

    # getter method for model property
    def get_model(self) -> BinaryResultsWrapper:
        return self.model

    # get the pseudo r_squared for model
    def get_pseudo_r_squared(self) -> float:
        return round(self.model.prsquared, 5)

    # get a confusion matrix
    def get_confusion_matrix(self, the_encoded_df: DataFrame) -> ndarray:
        # run validations
        if not isinstance(the_encoded_df, DataFrame):
            raise SyntaxError("the_encoded_df is None or incorrect type.")

        # log that we've been called
        self.logger.debug("A request to generate a confusion matrix has been made.")

        # declare the_result
        the_result = None

        # generate a prediction
        the_prediction = self.get_model().predict()

        # convert back to booleans
        predictions_boolean = [False if x < 0.5 else True for x in the_prediction]

        # create confusion matrix
        c_matrix = metrics.confusion_matrix(the_encoded_df['Churn'], predictions_boolean)

        # create a Series of actual and predicted
        y_actual = pd.Series(the_encoded_df['Churn'], name='Actual')
        y_predicted = pd.Series(predictions_boolean, name='Predicted')

        # store the accuracy_score
        self.accuracy_score = metrics.accuracy_score(y_actual, y_predicted)

        # store the precision_score
        self.precision_score = metrics.precision_score(y_actual, y_predicted)

        # store the recall_score
        self.recall_score = metrics.recall_score(y_actual, y_predicted)

        # store the f1 score
        self.f1_score = metrics.f1_score(y_actual, y_predicted)

        # return
        return c_matrix

    # get accuracy_score
    def get_accuracy_score(self) -> float:
        return self.accuracy_score

    # get precision_score
    def get_precision_score(self):
        return self.precision_score

    # get recall_score
    def get_recall_score(self):
        return self.recall_score

    # get f1_score
    def get_f1_score(self):
        return self.f1_score

    # get the log likelihood
    def get_log_likelihood(self):
        return self.model.llf

    # populate assumptions
    def populate_assumptions(self):
        # add keys to assumptions dict
        self.assumptions[PSEUDO_R_SQUARED_HEADER] = 'get_pseudo_r_squared'
        self.assumptions[MODEL_ACCURACY] = 'get_accuracy_score'
        self.assumptions[MODEL_PRECISION] = 'get_precision_score'
        self.assumptions[MODEL_RECALL] = 'get_recall_score'
        self.assumptions[MODEL_F1_SCORE] = 'get_f1_score'
        self.assumptions[AIC_SCORE] = 'get_aic_for_model'
        self.assumptions[BIC_SCORE] = 'get_bic_for_model'
        self.assumptions[LOG_LIKELIHOOD] = 'get_log_likelihood'
        self.assumptions[NUMBER_OF_OBS] = 'get_number_of_obs'
        self.assumptions[DEGREES_OF_FREEDOM_MODEL] = 'get_degrees_of_freedom_for_model'
        self.assumptions[DEGREES_OF_FREEDOM_RESID] = 'get_degrees_of_freedom_for_residuals'
        self.assumptions[MODEL_CONSTANT] = 'get_constant'

    # persist CSV files for this model.
    def generate_model_csv_files(self, csv_loader: CSV_Loader):
        # log what we are doing
        self.logger.debug("exporting CSV files for KNN model.")

        # validate arguments
        if not isinstance(csv_loader, CSV_Loader):
            raise SyntaxError("csv_loader is None or incorrect type.")

        # write out CHURN_FINAL -> CHURN_CSV_FILE_LOCATION
        csv_loader.generate_output_file(data_set=D_212_CHURN, option=CHURN_FINAL, the_dataframe=self.the_df)
        csv_loader.generate_output_file(data_set=D_212_CHURN, option=CHURN_PREP, the_dataframe=self.the_df)
