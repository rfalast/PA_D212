import logging

from pathlib import Path
from os.path import exists
from sklearn.metrics import classification_report
from model.BaseModel import BaseModel
from model.ReportGenerator import ReportGenerator
from model.analysis.models.KNN_Model import KNN_Model
from model.analysis.models.Linear_Model import Linear_Model
from model.analysis.models.Logistic_Model import Logistic_Model
from model.analysis.StatisticsGenerator import StatisticsGenerator
from model.analysis.models.Random_Forest_Model import Random_Forest_Model
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, ANALYZE_DATASET_INITIAL, D_212_CHURN, D_212_MEDICAL, \
    LOG_FILE_LOCATION, ANALYSIS_TYPE, CHURN_CSV_FILE_LOCATION, MEDICAL_CSV_FILE_LOCATION, \
    CHURN_PREP_CSV_FILE_LOCATION, MT_OPTIONS, MT_LINEAR_REGRESSION, MT_LOGISTIC_REGRESSION, MT_KNN_CLASSIFICATION, \
    MT_RF_REGRESSION
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.PCA_Analysis import PCA_Analysis
from model.analysis.PlotGenerator import PlotGenerator
from model.constants.ModelConstants import LM_FINAL_MODEL, LM_INITIAL_MODEL
from util.CSV_loader import CSV_Loader
from util.FileUtils import FileUtils


class Project_Assessment(BaseModel):

    # init method
    def __init__(self, base_directory=None, report_loc_override=None, ):
        # initialize super class.
        super().__init__()

        # perform validation of base_directory argument
        if not self.is_valid(base_directory, str):
            raise SyntaxError("The base_directory argument was None or invalid type.")
        elif not exists(base_directory):
            raise NotADirectoryError("base_directory does not exist.")

        # perform validation of report_loc_override argument
        if report_loc_override is not None:
            # make sure the argument is a string
            if not self.is_valid(report_loc_override, str):
                raise SyntaxError("The report_loc_override argument is invalid type.")
            # make sure the argument exists
            elif not exists(report_loc_override):
                raise NotADirectoryError("report_loc_override does not exist.")

        # initialize logging
        self.__initialize_logging__(base_directory)
        self.logger = logging.getLogger(__name__)

        # log a message
        self.logger.info("A request to initialize the PA has been made.")

        # define internal variables
        self.csv_l = None  # csv_loader
        self.df = None  # the dataframe that has been adjusted
        self.original_df = None  # the original dataframe
        self.cleaned_df = None  # the cleaned_df
        self.analyzer = None  # dataset analyzer
        self.p_gen = None  # plot generator
        self.s_gen = None  # statistics generator
        self.pca_analysis = None  # pca analysis
        self.dataset_key = None  # dataset name
        self.base_directory = base_directory
        self.report_loc_override = report_loc_override
        self.dataset_keys = {D_212_CHURN: self.base_directory + "resources/Input/churn_raw_data.csv",
                             D_212_MEDICAL: self.base_directory + "resources/Input/medical_raw_data.csv"}
        self.r_gen = ReportGenerator(report_loc_override)
        self.CSV_FILE_LOCATION = None
        self.CSV_PREP_FILE_LOCATION = None

    # setup logging
    def __initialize_logging__(self, base_directory):
        # define full log path
        log_file_path = Path(base_directory + LOG_FILE_LOCATION)

        # log to console.
        print(f"log file available at [{log_file_path}]\n")

        # setup the logger
        logging.basicConfig(
            filename=Path(log_file_path),
            encoding="utf-8",
            filemode='w+',
            level=logging.INFO,
            format="%(levelname)s %(asctime)s %(module)s.%(funcName)s(): %(message)s",
            datefmt="%m-%d-%Y %I:%M:%S %p"
        )

        # log that the logger is setup.  At this point, you have to reference the generic logging.debug
        # as the internal self.logger has not been setup.
        logging.debug(f"logs will be available at [{log_file_path}]")

    # write to stdout that we cannot log.
    @staticmethod
    def __log_error_message__():
        print("********************************************")
        print("log file location message is bad. No logging will occur.")
        print("********************************************")

    # load a specific data set
    def load_dataset(self, dataset_name_key):
        # log that we've been invoked
        self.logger.debug("A call to load_dataset() has been made.")

        # variable declaration
        self.csv_l = CSV_Loader(base_directory=self.base_directory)

        # validate incoming dataset_name_key
        if not self.is_valid(dataset_name_key, str):
            raise TypeError("The dataset_name_key as None or the incorrect type.")
        # validate the key is a known key
        elif dataset_name_key not in self.dataset_keys:
            raise SyntaxError("An unknown dataset_name_key was used.")
        else:
            # log that we're good.
            self.logger.info(f"A dataset key of [{dataset_name_key}] was received.")

            # set the internal dataset_key
            self.dataset_key = dataset_name_key

            # load dataframe
            self.df = self.csv_l.get_data_frame_from_csv(self.dataset_keys[dataset_name_key])
            self.original_df = self.csv_l.get_data_frame_from_csv(self.dataset_keys[dataset_name_key])

            # set the CSV_FILE_LOCATION based on the dataset_name_key variable
            if dataset_name_key == D_212_CHURN:
                self.CSV_FILE_LOCATION = CHURN_CSV_FILE_LOCATION
                self.CSV_PREP_FILE_LOCATION = CHURN_PREP_CSV_FILE_LOCATION
            elif dataset_name_key == D_212_MEDICAL:
                self.CSV_FILE_LOCATION = MEDICAL_CSV_FILE_LOCATION
            else:
                raise RuntimeError("unknown dataset_name_key, unable to set CSV_FILE_LOCATION variable.")

            # log that we've completed loading
            self.logger.info("dataset is now loaded.")

    # perform analysis on dataset
    def analyze_dataset(self, analysis_type):
        # validate argument
        if not self.is_valid(analysis_type, str):
            raise SyntaxError("The analysis_type was None or invalid type.")
        elif analysis_type not in ANALYSIS_TYPE:
            raise SyntaxError(f"The analysis_type [{analysis_type}] is unknown.")

        # validate that the internal df is populated
        if self.df is None:
            raise Exception("Dataframe was not populated.")

        # log that we've been called
        self.logger.info("A request to analyze the dataset has been made.")

        # set up the DatasetAnalyzer
        self.analyzer = DatasetAnalyzer(self.df)

        # add the original dataframe back.  I'm doing this because I don't fully understand
        # what operations cause a new DataFrame to be copied from another.  Thus, I'm
        # explicitly loading a parallel instance and assigning it with this call.
        self.analyzer.add_original_df(self.original_df)

        # determine which analysis type to run
        if analysis_type == ANALYZE_DATASET_FULL:
            # log that we're doing the initial run
            self.logger.debug("performing complete setup.")

            # run full analysis
            self.analyzer.run_complete_setup()
        # initial run
        elif analysis_type == ANALYZE_DATASET_INITIAL:
            # log that we're doing the initial run
            self.logger.debug("performing the initial run.")

            # run only initial pass
            self.analyzer.refresh_model()

            # extract boolean data.
            self.analyzer.extract_boolean()
        # no idea how we get here
        else:
            # raise an error
            raise RuntimeError("Unknown state.")

        # log that we are complete
        self.logger.info("Analysis complete.")

    # perform pca on dataset
    def perform_pca_on_dataset(self, column_dict, column_drop_list=None):
        # perform validation
        if not self.is_valid(column_dict, dict):
            raise TypeError("column_dict is None or wrong type.")
        elif len(column_dict) == 0:
            raise ValueError("column_dict is empty.")

        # validate optional argument of columns to drop prior to normalization
        if column_drop_list is not None:
            # validate that we have a list
            if not self.is_valid(column_drop_list, list):
                raise TypeError("column_drop_list is incorrect type.")

        # this is internal validation that the column_dict ONLY contains columns present
        # in the DataFrame stored at analyzer.the_df.
        # loop over all the columns and make sure they are present in the dataframe
        for key in column_dict:
            # log the columns
            self.logger.debug(f"current column is [{key}][{column_dict[key]}]")

            # we need to check that the column is present in the DataFrame
            if column_dict[key] not in self.analyzer.the_df:
                raise ValueError(f"column [{column_dict[key]}] not present on dataframe.")

        # log that we're running PCA on dataset
        self.logger.debug("performing PCA on dataset.")

        # at this point, I have list of columns in column_dict that have been verified to be present
        # in the internal DataFrame reference.  I need to first tell the DataAnalyzer to normalize
        # all columns in the dataset.

        # normalize the dataset
        self.analyzer.normalize_dataset(column_drop_list=column_drop_list)

        # create PCA Analysis object
        self.pca_analysis = PCA_Analysis(self.analyzer, column_dict)

        # perform analysis
        self.pca_analysis.perform_analysis()

        # generate the charts
        self.p_gen.generate_scree_plots(self.pca_analysis)

        # export the entire dataset and it's normalization
        self.r_gen.export_dataframes_to_excel(self.analyzer, self.pca_analysis)

        # log that we are finished with PCA.
        self.logger.debug("PCA on dataset is complete.")

    # generate the initial Excel based reports
    def generate_initial_report(self):
        # log that we've been called
        self.logger.debug("A request to generate the FINAL output report has been made.")

        # first validate that the report generator isn't None
        if self.r_gen is None:
            # log that we attempted to recreate it
            self.logger.info("The internal reference to the ReportGenerator was None, it was recreated.")

            # create the ReportGenerator
            self.r_gen = ReportGenerator()

        # create a PlotGenerator
        self.p_gen = PlotGenerator(self.analyzer, self.r_gen.report_path)

        # tell it to generate all the plots required
        self.p_gen.generate_all_dataset_plots(statistics_generator=None,
                                              the_model_type=MT_LINEAR_REGRESSION,
                                              the_version=ANALYZE_DATASET_INITIAL)

        # tell it to generate the initial report --> ANALYSIS_TYPE[1]
        self.r_gen.generate_excel_report(self.analyzer,
                                         the_plot_generator=self.p_gen,
                                         stat_generator=None,
                                         the_type=ANALYZE_DATASET_INITIAL,
                                         the_model_type=MT_LINEAR_REGRESSION)

    # generate the FINAL Excel based reports
    def generate_output_report(self, the_model_type=None):
        # log that we've been called
        self.logger.debug(f"Generating the FINAL output report with the_model_type[{the_model_type}].")

        # variable declaration
        model_option = None
        the_model_result = None

        # Generalized note to self.  I need to re-work this use case because it has become a jumbled mess.

        # first validate that the report generator isn't None
        if self.r_gen is None:
            # log that we attempted to recreate it
            self.logger.info("The internal reference to the ReportGenerator was None, it was recreated.")

            # create the ReportGenerator
            self.r_gen = ReportGenerator(self.report_loc_override)

        # create a PlotGenerator
        self.p_gen = PlotGenerator(self.analyzer, self.r_gen.report_path)

        # tell it to generate all the plots required
        self.p_gen.generate_all_dataset_plots(statistics_generator=self.s_gen,
                                              the_model_type=the_model_type,
                                              the_version=ANALYZE_DATASET_FULL)

        # generate heatmap for correlations
        self.p_gen.generate_correlation_heatmap(self.s_gen.get_list_of_correlations())

        # generate heatmap for confusion matrix for MT_LOGISTIC_REGRESSION
        if the_model_type == MT_LOGISTIC_REGRESSION:
            # get the final logistic model
            the_model = self.analyzer.linear_model_storage[LM_FINAL_MODEL]

            # get a model result
            the_model_result = the_model.get_the_result()

            # get the confusion matrix
            confusion_matrix = the_model_result.get_confusion_matrix(the_encoded_df=the_model.encoded_df)

            # generate the confusion matrix heatmap. Set count to 2.  On the file system, the first
            # heatmap file will be the correlation heatmap, and the second will be the confusion matrix
            self.p_gen.generate_confusion_matrix_heatmap(confusion_matrix=confusion_matrix, the_count=2)

            # set the model option to LM_FINAL_MODEL
            model_option = LM_FINAL_MODEL

        # generate heatmap for confusion matrix for MT_KNN_CLASSIFICATION
        elif the_model_type == MT_KNN_CLASSIFICATION:
            # get the model result
            the_model_result = self.analyzer.linear_model_storage[LM_INITIAL_MODEL].get_the_result()

            # generate the confusion matrix heatmap. Set count to 2.  On the file system, the first
            # heatmap file will be the correlation heatmap, and the second will be the confusion matrix
            self.p_gen.generate_confusion_matrix_heatmap(confusion_matrix=the_model_result.get_confusion_matrix(),
                                                         the_count=2)

            # console output required for PA.
            print(classification_report(the_model_result.the_t_var_test, the_model_result.y_pred))

            # generate the AUC/ROC plot
            self.p_gen.generate_auc_roc_plot_knn(the_model=the_model_result.get_model(),
                                                 x_test=the_model_result.the_f_df_test,
                                                 y_test=the_model_result.the_t_var_test,
                                                 the_count=1)

            # set the model option to LM_INITIAL_MODEL
            model_option = LM_INITIAL_MODEL
        # MT_RF_REGRESSION
        elif the_model_type == MT_RF_REGRESSION:
            # get the model result
            the_model_result = self.analyzer.linear_model_storage[LM_INITIAL_MODEL].get_the_result()

            # set the model option to LM_INITIAL_MODEL
            model_option = LM_INITIAL_MODEL
        else:
            # get the model result
            the_model_result = self.analyzer.linear_model_storage[LM_FINAL_MODEL].get_the_result()

        # generate the path to the location where clean data CSV is located.
        the_path = Path(self.base_directory + self.CSV_FILE_LOCATION)

        # log the path where we're writing out clean file.
        self.logger.debug(f"FINAL CSV file(s) being written to [{the_path}]")

        # generate the output CSV files
        the_model_result.generate_model_csv_files(csv_loader=self.csv_l)

        # tell it to generate the final Excel report --> ANALYZE_DATASET_FULL
        self.r_gen.generate_excel_report(self.analyzer,
                                         the_plot_generator=self.p_gen,
                                         stat_generator=self.s_gen,
                                         the_model_type=the_model_type,
                                         the_type=ANALYZE_DATASET_FULL)

    # update the base directory
    def update_base_directory(self, the_path):
        # run validation
        if not self.is_valid(the_path, str):
            raise SyntaxError("The path was None or incorrect type.")
        elif not exists(the_path):
            raise NotADirectoryError(f"the path [{the_path}] does not exist.")

        # variable declaration
        file_util = FileUtils()

        # check if the directory seperator is present at the last character
        if the_path[-1] != file_util.get_directory_seperator():
            the_path = the_path + file_util.get_directory_seperator()

        # capture the base directory
        self.base_directory = the_path

        # log the base directory
        self.logger.info(f"The base directory has been set to [{self.base_directory}].")

    # update column names
    def update_column_names(self, variable_dict):
        if not self.is_valid(variable_dict, dict):
            raise SyntaxError("The variable_dict was None or incorrect type.")

        # this is internal validation that the variable_dict ONLY contains columns present
        # in the DataFrame stored at analyzer.the_df.
        # loop over all the columns and make sure they are present in the dataframe
        for key in list(variable_dict.keys()):
            # log the column we are changing the name of
            self.logger.debug(f"current column is [{key}][{variable_dict[key]}]")

            # we need to check that the column is present in the DataFrame
            if key not in self.analyzer.the_df.columns:
                raise ValueError(f"column [{key}] not present on dataframe.")
            else:
                # log that it was present
                self.logger.debug(f"key[{key}] was found on the DataFrame.")

                # we need to change the name of the column on the dataframe.
                self.df = self.df.rename(mapper=variable_dict, axis='columns')

    # drop column from dataset
    def drop_column_from_dataset(self, columns_to_drop_list):
        if not self.is_valid(columns_to_drop_list, list):
            raise SyntaxError("The columns_to_drop_list was None or incorrect type.")

        # this is internal validation that the columns_to_drop_list ONLY contains columns present
        # in the DataFrame stored at pa.df
        # loop over all the columns and make sure they are present in the dataframe
        for the_column in columns_to_drop_list:
            # log the column we are changing the name of
            self.logger.debug(f"attempting to drop column [{the_column}].")

            # we need to check that the column is present in the DataFrame
            if the_column not in list(self.df.columns):
                raise ValueError(f"column [{the_column}] not present on dataframe.")
            else:
                # log that it was present
                self.logger.debug(f"the_column[{the_column}] was found on the DataFrame.")

                # drop the columns
                self.df = self.df.drop(the_column, axis=1)

    # calculate internal statistics
    def calculate_internal_statistics(self, the_level=0.5):
        # run validations
        if not isinstance(the_level, float):
            raise SyntaxError("the_level argument is None or incorrect type.")
        elif the_level < 0 or the_level > 1:
            raise ValueError("the_level is not in [0,1].")

        # log that we are calculating a plethora of internal statistics
        self.logger.debug(f"Calculating internal statistics.")

        # instantiate a Statistics Generator
        self.s_gen = StatisticsGenerator(self.analyzer)

        # calculate correlations
        self.s_gen.find_correlations(the_level=the_level)

        # check if the categorical variables are independent
        self.s_gen.find_chi_squared_results()

        # calculate the distributions
        self.s_gen.fit_theoretical_dist_to_all_columns()

    # normalize the entire dataset
    def normalize_dataset(self):
        # normalize the dataset
        self.analyzer.normalize_dataset()

    # build a linear model for current dataset
    # NOTE to myself, I need to change the method interface to use a dictionary since the requirements
    # are so broad for across different test types.
    def build_model(self, the_target_column, model_type, max_p_value=1.0, the_max_vif=10.0, suppress_console=False):
        if not isinstance(the_target_column, str):
            raise ValueError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.analyzer.the_df.columns:
            raise SyntaxError("the_target_column argument is not in dataframe.")
        elif model_type not in MT_OPTIONS:
            raise ValueError("model_type is None or incorrect value.")
        # validate that the p-value is a float < 1.0 and > 0
        elif not isinstance(max_p_value, float):
            raise SyntaxError("max_p_value was None or incorrect type.")
        elif max_p_value < 0 or max_p_value > 1.0:
            raise ValueError("max_p_value must be in range (0,1).")
        elif not isinstance(the_max_vif, float):
            raise SyntaxError("the_max_vif was None or incorrect type.")
        elif the_max_vif < 1.0:
            raise ValueError("the_max_vif must be > 1.0")

        # log that we've been called
        self.logger.debug(f"a request to build a [{model_type}] model for the dataset has been made.")

        # variable declaration
        initial_model = None
        final_model = None

        # determine if type is MT_LINEAR_REGRESSION
        if model_type == MT_LINEAR_REGRESSION:
            # initialize the initial_model
            initial_model = Linear_Model(self.analyzer)

            # initialize the final_model
            final_model = Linear_Model(self.analyzer)
        elif model_type == MT_LOGISTIC_REGRESSION:
            # initialize the initial_model
            initial_model = Logistic_Model(self.analyzer)

            # initialize the final_model
            final_model = Logistic_Model(self.analyzer)
        elif model_type == MT_KNN_CLASSIFICATION:
            # initialize the initial_model
            initial_model = KNN_Model(self.analyzer)

            # set the final_model to None
            final_model = None
        elif model_type == MT_RF_REGRESSION:
            # initialize the initial_model
            initial_model = Random_Forest_Model(self.analyzer)

            # set the final_model to None
            final_model = None

        # build the initial model.  This will be the model that we compare our final model too.
        initial_model.fit_a_model(the_target_column=the_target_column,
                                  current_features=initial_model.get_encoded_variables(
                                      the_target_column=the_target_column),
                                  model_type=model_type)

        # check if we dump results to console if not KNN
        if suppress_console or model_type in [MT_KNN_CLASSIFICATION, MT_RF_REGRESSION]:
            # log the that we're not logging to console
            self.logger.debug("suppressing console output.")
        else:
            # log to the console.
            initial_model.log_model_summary_to_console(the_type=LM_INITIAL_MODEL)

        # store the initial model from earlier so we can compare to a reduced model
        self.analyzer.add_model(the_type=LM_INITIAL_MODEL, the_model=initial_model)

        # build the final model if not KNN or Random Forest
        if model_type not in [MT_KNN_CLASSIFICATION, MT_RF_REGRESSION]:
            # invoke solve for target
            final_model.solve_for_target(the_target_column=the_target_column,
                                         model_type=model_type,
                                         max_p_value=max_p_value,
                                         the_max_vif=the_max_vif,
                                         suppress_console=suppress_console)

            # dump results to console
            if suppress_console:
                # log the that we're not logging to console
                self.logger.debug("suppressing console output.")
            else:
                final_model.log_model_summary_to_console(the_type=LM_FINAL_MODEL)

            # add the reduced model to storage.
            self.analyzer.add_model(the_type=LM_FINAL_MODEL, the_model=final_model)

        # log that we are finished
        self.logger.debug("model creation complete.")

    # clean up outliers
    def clean_up_outliers(self, model_type: str, max_p_value: float):
        # argument validation
        if not isinstance(model_type, str):
            raise SyntaxError("model_type argument is None or incorrect type.")
        elif model_type not in MT_OPTIONS:
            raise ValueError("model_type value is unknown.")
        elif not isinstance(max_p_value, float):
            raise SyntaxError("max_p_value argument is None or incorrect type.")
        elif max_p_value <= 0 or max_p_value >= 1:
            raise ValueError("max_p_value is not in (0,1).")

        # log what we're doing
        self.logger.debug("A request to clean up outliers has been made.")

        # invoke the method
        self.analyzer.clean_up_outliers(model_type=model_type, max_p_value=max_p_value)

        # reset the df from the analyzer
        self.df = self.analyzer.the_df
