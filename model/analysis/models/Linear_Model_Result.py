import logging

from statsmodels.compat import lzip
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import D_209_CHURN, CHURN_FINAL, CHURN_PREP
from model.constants.ModelConstants import LM_P_VALUE,  LM_LAGRANGE_MULTIPLIER_STATISTIC, LM_F_VALUE, LM_F_P_VALUE, \
     LM_JARQUE_BERA_STATISTIC, LM_JARQUE_BERA_PROB, LM_JB_SKEW, LM_JS_KURTOSIS
from model.constants.ReportConstants import R_SQUARED_HEADER, ADJ_R_SQUARED_HEADER, DURBAN_WATSON_STATISTIC, \
    RESIDUAL_STD_ERROR, BREUSCH_PAGAN_P_VALUE, JARQUE_BERA_STATISTIC, F_STATISTIC_HEADER, P_VALUE_F_STATISTIC_HEADER, \
    NUMBER_OF_OBS, DEGREES_OF_FREEDOM_MODEL, DEGREES_OF_FREEDOM_RESID, MODEL_CONSTANT, AIC_SCORE, BIC_SCORE
from util.CSV_loader import CSV_Loader


# Linear Model tools
class Linear_Model_Result(ModelResultBase):

    # init method
    def __init__(self, the_regression_wrapper, the_target_variable, the_variables_list, the_df):
        super().__init__(the_target_variable, the_variables_list, the_df)

        # run validation of the_regression_wrapper.  Swap this out with RegressionResults
        if not isinstance(the_regression_wrapper, RegressionResultsWrapper):
            raise SyntaxError("the_regression_wrapper is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # define and capture internal variables
        self.model = the_regression_wrapper

    # getter method for model property
    def get_model(self) -> RegressionResultsWrapper:
        return self.model

    # get the adjusted r_squared for model
    def get_adjusted_r_squared(self) -> float:
        return round(self.model.rsquared_adj, 5)

    # get the r_squared for model
    def get_r_squared(self) -> float:
        return round(self.model.rsquared, 5)

    # perform durbin watson on residuals
    def get_durbin_watson_statistic_for_residuals(self) -> float:
        # log what we are doing
        self.logger.debug("A request to get Durbin Watson statistic for residuals has been made.")

        # return the statistic
        return durbin_watson(self.get_model().resid).round(10)

    # get the residual standard error
    def get_residual_standard_error(self, round_value=10) -> float:
        # run validation of arguments
        if not isinstance(round_value, int):
            raise SyntaxError("round_value is None or incorrect type.")
        elif round_value < 1:
            raise ValueError("round_value cannot be below one.")

        # log what we are doing
        self.logger.debug("A request to get the residual standard error has been made.")

        # return the rse
        return self.model.resid.std(ddof=self.the_df[self.get_the_variables_list()].shape[1]).round(round_value)

    # get the Breusch Pagan statistic
    def get_breusch_pagan_statistic(self, round_value=10) -> dict:
        # run validation of arguments
        if not isinstance(round_value, int):
            raise SyntaxError("round_value is None or incorrect type.")
        elif round_value < 1:
            raise ValueError("round_value cannot be below one.")

        # log what we are doing
        self.logger.debug("A request to get the Breusch Pagan statistic has been made.")

        # variable declaration
        the_result = {LM_LAGRANGE_MULTIPLIER_STATISTIC: 0, LM_P_VALUE: 0, LM_F_VALUE: 0, LM_F_P_VALUE: 0}

        # define the variable names
        names = [LM_LAGRANGE_MULTIPLIER_STATISTIC, LM_P_VALUE, LM_F_VALUE, LM_F_P_VALUE]

        # get the test result
        test_result = het_breuschpagan(self.model.resid, self.model.model.exog)

        # create a list of the results
        test_result_list = lzip(names, test_result)

        # create the final_result dict
        the_result[LM_LAGRANGE_MULTIPLIER_STATISTIC] = test_result_list[0][1].round(round_value)
        the_result[LM_P_VALUE] = test_result_list[1][1].round(round_value)
        the_result[LM_F_VALUE] = test_result_list[2][1].round(round_value)
        the_result[LM_F_P_VALUE] = test_result_list[3][1].round(round_value)

        return the_result

    # get the Jarque-Bera statistic
    def get_jarque_bera_statistic(self, round_value=10) -> dict:
        # run validation of arguments
        if not isinstance(round_value, int):
            raise SyntaxError("round_value is None or incorrect type.")
        elif round_value < 1:
            raise ValueError("round_value cannot be below one.")

        # log what we are doing
        self.logger.debug("A request to get the Jarque Bera statistic has been made.")

        # variable declaration
        the_result = {LM_JARQUE_BERA_STATISTIC: 0, LM_JARQUE_BERA_PROB: 0, LM_JB_SKEW: 0, LM_JS_KURTOSIS: 0}

        # get the test statistic
        jbs_statistic = jarque_bera(self.model.resid)

        # create the final_result dict
        the_result[LM_JARQUE_BERA_STATISTIC] = jbs_statistic[0].round(round_value)
        the_result[LM_JARQUE_BERA_PROB] = jbs_statistic[1].round(round_value)
        the_result[LM_JB_SKEW] = jbs_statistic[2].round(round_value)
        the_result[LM_JS_KURTOSIS] = jbs_statistic[3].round(round_value)

        # return
        return the_result

    # get the f-value
    def get_f_value(self) -> float:
        return self.model.fvalue

    # get the p-value of the f-statistic
    def get_f_p_value(self) -> float:
        return self.model.f_pvalue

    # populate assumptions
    def populate_assumptions(self):
        # add keys to assumptions dict
        self.assumptions[R_SQUARED_HEADER] = 'get_r_squared'
        self.assumptions[ADJ_R_SQUARED_HEADER] = 'get_adjusted_r_squared'
        self.assumptions[DURBAN_WATSON_STATISTIC] = 'get_durbin_watson_statistic_for_residuals'
        self.assumptions[RESIDUAL_STD_ERROR] = 'get_residual_standard_error'
        self.assumptions[BREUSCH_PAGAN_P_VALUE] = 'get_breusch_pagan_statistic'
        self.assumptions[JARQUE_BERA_STATISTIC] = 'get_jarque_bera_statistic'
        self.assumptions[F_STATISTIC_HEADER] = 'get_f_value'
        self.assumptions[P_VALUE_F_STATISTIC_HEADER] = 'get_f_p_value'
        self.assumptions[NUMBER_OF_OBS] = 'get_number_of_obs'
        self.assumptions[DEGREES_OF_FREEDOM_MODEL] = 'get_degrees_of_freedom_for_model'
        self.assumptions[DEGREES_OF_FREEDOM_RESID] = 'get_degrees_of_freedom_for_residuals'
        self.assumptions[MODEL_CONSTANT] = 'get_constant'
        self.assumptions[AIC_SCORE] = 'get_aic_for_model'
        self.assumptions[BIC_SCORE] = 'get_bic_for_model'

        # persist CSV files for this model.

    def generate_model_csv_files(self, csv_loader: CSV_Loader):
        # log what we are doing
        self.logger.debug("exporting CSV files for KNN model.")

        # validate arguments
        if not isinstance(csv_loader, CSV_Loader):
            raise SyntaxError("csv_loader is None or incorrect type.")

        # there are file CSV files that this model will generate
        # CHURN_CSV_FILE_LOCATION -> "resources/Output/churn_cleaned.csv"
        # CHURN_PREP_CSV_FILE_LOCATION -> "resources/Output/churn_prepared.csv"

        # write out CHURN_FINAL -> CHURN_CSV_FILE_LOCATION
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_FINAL, the_dataframe=self.the_df)
        csv_loader.generate_output_file(data_set=D_209_CHURN, option=CHURN_PREP, the_dataframe=self.the_df)
