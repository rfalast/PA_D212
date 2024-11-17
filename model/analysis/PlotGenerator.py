import logging
import re
import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab

from copy import copy
from os.path import exists

from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import roc_curve, auc
from statsmodels.regression.linear_model import RegressionResultsWrapper
from model.BaseModel import BaseModel
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Detector import Detector
from model.analysis.PCA_Analysis import PCA_Analysis
from model.analysis.StatisticsGenerator import StatisticsGenerator
from model.constants.BasicConstants import ANALYZE_DATASET_INITIAL, ANALYZE_DATASET_FULL, MT_OPTIONS, \
    MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION, MT_KNN_CLASSIFICATION
from model.constants.DatasetConstants import UNIQUE_COLUMN_RATIOS, BOOL_COLUMN_COUNT_KEY, UNIQUE_COLUMN_VALUES, \
    INT64_COLUMN_KEY, FLOAT64_COLUMN_KEY, BOOL_COLUMN_KEY, OBJECT_COLUMN_KEY, DATETIME_COLUMN_KEY, \
    OBJECT_COLUMN_COUNT_KEY
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_VALID_TYPES
from model.constants.PlotConstants import BASE_OUTPUT_PATH, PLOT_TYPE_HIST, PLOT_TYPE_BOX_PLOT, PLOT_TYPE_BAR_CHART, \
    PLOT_TYPE_SCATTER_CHART, HIST_BASE_TITLE, BOX_PLOT_BASE_TITLE, BAR_CHART_BASE_TITLE, UNDERSCORE, FILE_NAME_CLOSE, \
    COUNT_VARIABLE, PLOT_TYPE_SCREE, HEATMAP_CORR_CHART_BASE_TITLE, PLOT_TYPE_HEATMAP, GENERAL, SCATTER_PLOT_TITLE, \
    AMPERSAND, BIVARIATE_COUNT_PLOT_TITLE, PLOT_TYPE_BIVARIATE_COUNT, PLOT_TYPE_JOINT_PLOT, JOINT_PLOT_TITLE, \
    PLOT_TYPE_Q_Q_PLOT, Q_Q_PLOT_TITLE, STANDARDIZED_RESIDUALS_LABEL, STANDARDIZED_RESIDUAL_PLOT_TITLE, \
    PLOT_TYPE_STD_RESIDUAL, BIVARIATE_BOX_PLOT_TITLE, PLOT_TYPE_BIVARIATE_BOX, Q_Q_RESIDUAL_PLOT_TITLE, \
    PLOT_TYPE_Q_Q_RESIDUAL_PLOT, PLOT_TYPE_LONG_ODDS, LONG_ODDS_PLOT_TITLE, HEATMAP_CM_CHART_BASE_TITLE, \
    PLOT_TYPE_CM_HEATMAP, PLOT_TYPE_ROC_AUC
from model.constants.StatisticsConstants import ALL_CORRELATIONS
from util.CommonUtils import get_tuples_from_list_with_specific_field
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# class for generating
class PlotGenerator(BaseModel):

    # init method
    def __init__(self, data_analyzer, output_path_override=None):
        # call the super
        super().__init__()

        # perform validations of arguments
        if not self.is_valid(data_analyzer, DatasetAnalyzer):
            raise SyntaxError("The data_analyzer argument is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # log that we've been instantiated
        self.logger.debug("An instance of PlotGenerator has been created.")

        # check if the override variable was passed
        if self.is_valid(output_path_override, str):
            # log that we're overriding the path
            self.logger.debug(f"overriding output path to [{output_path_override}]")

            # generate the output path
            self.output_path = output_path_override

            # validate that the overriden output path is valid
            if not exists(output_path_override):
                raise FileNotFoundError(f"override path [{output_path_override}] does not exist.")
        # generate the full path
        else:
            # log that we're not overriding the path
            self.logger.debug("Not overriding output path.")

            # generate the default output path
            self.output_path = BASE_OUTPUT_PATH

        # define internal variables
        self.data_analyzer = data_analyzer
        self.data_type_map = {INT64_COLUMN_KEY: "INT",
                              FLOAT64_COLUMN_KEY: "FLOAT",
                              OBJECT_COLUMN_KEY: "OBJECT",
                              BOOL_COLUMN_KEY: "BOOL",
                              DATETIME_COLUMN_KEY: "DATETIME"}

        # create the list of plot types
        self.plot_type_list = [PLOT_TYPE_HIST, PLOT_TYPE_BOX_PLOT, PLOT_TYPE_BAR_CHART, PLOT_TYPE_SCATTER_CHART,
                               PLOT_TYPE_HEATMAP, PLOT_TYPE_BIVARIATE_COUNT, PLOT_TYPE_JOINT_PLOT, PLOT_TYPE_Q_Q_PLOT,
                               PLOT_TYPE_STD_RESIDUAL, PLOT_TYPE_BIVARIATE_BOX, PLOT_TYPE_Q_Q_RESIDUAL_PLOT,
                               PLOT_TYPE_LONG_ODDS]

        # populate our storage structure.
        # First level key is column name, GENERAL for heatmap, or a tuple of two columns.
        # second level key is plot type
        # value is the path to the actual plot
        self.plot_storage = {}

        # log the output path
        self.logger.info(f"final report output path set to [{self.output_path}]")

    # method to generate a histogram
    def generate_hist(self, data_type, the_column, the_count):
        # perform validations of data_type
        if not self.is_valid(data_type, str):
            raise SyntaxError("data_type argument is None or incorrect type.")
        elif data_type not in self.data_type_map:
            raise SyntaxError("The data_type argument is unknown.")

        # perform validations of the_column
        elif not self.is_valid(the_column, str):
            raise SyntaxError("The column is None or incorrect type.")
        elif the_column not in self.data_analyzer.storage[data_type]:
            raise SyntaxError("The column was not found on list for the data_type.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("The count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("The count argument must be positive.")

        # log what we're going to do
        self.logger.debug(f"Generating histogram for data_type[{data_type}] "
                          f"the_column[{the_column}] the_count[{the_count}]")

        # variable declaration
        df_column = self.data_analyzer.the_df[the_column]
        the_detector = Detector()
        the_title = HIST_BASE_TITLE + the_column  # the plot title.

        # get the plot name to generate
        the_plot_name = self.get_plot_file_name(data_type, PLOT_TYPE_HIST, the_count)

        # clear everything
        plt.clf()

        # we need to check if the type is INT64 and includes nan, if this is found to be true
        # then we will NOT be able to call plt.hist() directly.
        # check if the type is INT64 and includes nan.
        if the_detector.detect_int_with_na(df_column):
            # log that we have an int with nan
            self.logger.debug("generating different plot for INT64 with nan.")

            # drop the NA columns without touching original dataset.
            temp_column = df_column.dropna()

            # generate the plot
            plt.hist(temp_column, bins="auto")
        else:
            # log that we are generating a standard histogram.
            self.logger.debug("generating standard histogram.")

            # generate the plot
            plt.hist(df_column, bins="auto")

        # add the title
        plt.title(the_title)

        # log that we're about to save the file
        self.logger.debug(f"About to save plot to [{the_plot_name}]")

        # save
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if the_column not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[the_column] = {self.plot_type_list[0]: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][HIST]
            self.plot_storage[the_column][self.plot_type_list[0]] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{the_column}][{self.plot_type_list[0]}]")

    # method to generate a box plot
    def generate_boxplot(self, data_type, the_column, the_count):
        # perform validations of data_type
        if not self.is_valid(data_type, str):
            raise SyntaxError("data_type argument is None or incorrect type.")
        elif data_type not in self.data_type_map:
            raise SyntaxError("The data_type argument is unknown.")

        # perform validations of the_column
        elif not self.is_valid(the_column, str):
            raise SyntaxError("The column is None or incorrect type.")
        elif the_column not in self.data_analyzer.storage[data_type]:
            raise SyntaxError(f"The column was not found on list for the data_type [{data_type}].")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("The count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("The count argument must be positive.")

        # log that we've been called
        self.logger.debug(f"Generating box plot for data_type [{data_type}] "
                          f"the_column[{the_column}] the_count[{the_count}].")

        # variable declaration
        # the plot title.
        the_title = BOX_PLOT_BASE_TITLE + the_column

        # get the plot name to generate
        the_plot_name = self.get_plot_file_name(data_type, PLOT_TYPE_BOX_PLOT, the_count)

        # we need to make sure NaN values are not included in the plot
        filtered_data = self.data_analyzer.the_df[the_column][~np.isnan(self.data_analyzer.the_df[the_column])]

        # clear everything
        plt.clf()

        # generate box plot
        plt.boxplot(filtered_data, autorange=True)
        plt.title(the_title)

        # log that we're about to save the file
        self.logger.debug(f"About to save plot to [{the_plot_name}]")

        # save
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if the_column not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[the_column] = {self.plot_type_list[1]: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][HIST]
            self.plot_storage[the_column][self.plot_type_list[1]] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{the_column}][{self.plot_type_list[1]}]")

    # generate bivariate box plot
    def generate_bivariate_boxplot(self, the_tuple, the_count):
        # perform validations of the_tuple
        if not isinstance(the_tuple, tuple):
            raise SyntaxError("the_tuple argument is None or incorrect type.")

        # perform validations of x_series
        elif not self.is_valid(the_tuple[0], str):
            raise SyntaxError("x_series is None or incorrect type.")
        elif the_tuple[0] not in self.data_analyzer.the_df:
            raise ValueError("x_series is not a valid field on dataframe.")

        # perform validations of y_series
        elif not self.is_valid(the_tuple[1], str):
            raise SyntaxError("y_series is None or incorrect type.")
        elif the_tuple[1] not in self.data_analyzer.the_df:
            raise ValueError("y_series is not a valid field on dataframe.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("the_count argument is None or incorrect type.")
        elif the_count < 0:
            raise ValueError("the_count argument must be positive.")

        # variable declaration
        x_series = the_tuple[0]
        y_series = the_tuple[1]
        the_title = BIVARIATE_BOX_PLOT_TITLE + the_tuple[0] + AMPERSAND + the_tuple[1]
        the_plot_name = copy(self.output_path) + PLOT_TYPE_BIVARIATE_BOX + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # log that we're about to create box plot
        self.logger.debug(f"creating bi-variate box plot for {x_series} vs. {y_series}.")

        # clear everything
        plt.clf()

        # generate the plot
        sns.boxplot(data=self.data_analyzer.the_df, x=x_series, y=y_series, color="cyan")
        plt.xlabel(x_series)
        plt.ylabel(y_series)
        plt.title(the_title)

        # save the plot
        plt.savefig(the_plot_name)
        plt.close()

        # log the name of the file created
        self.logger.debug(f"created box plot [{the_plot_name}] for [{x_series}][{y_series}]")

        # add to storage
        if PLOT_TYPE_BIVARIATE_BOX in self.plot_storage:
            # add to storage
            self.plot_storage[PLOT_TYPE_BIVARIATE_BOX][the_tuple] = the_plot_name
        else:
            # add to storage
            self.plot_storage[PLOT_TYPE_BIVARIATE_BOX] = {the_tuple: the_plot_name}

    # method to generate a bar chart
    def generate_bar_chart(self, data_type, the_column, the_count):
        # perform validations of data_type
        if not self.is_valid(data_type, str):
            raise SyntaxError("data_type argument is None or incorrect type.")
        elif data_type not in self.data_type_map:
            raise SyntaxError("The data_type argument is unknown.")

        # perform validations of the_column
        elif not self.is_valid(the_column, str):
            raise SyntaxError("The column is None or incorrect type.")
        elif the_column not in self.data_analyzer.storage[data_type]:
            raise SyntaxError("The column was not found on list for the data_type.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("The count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("The count argument must be positive.")

        # log that we've been called
        self.logger.debug(f"Generating bar chart for data_type [{data_type}] "
                          f"the_column[{the_column}] the_count[{the_count}].")

        # variable declaration
        the_plot_name = None
        temp_data_list = []
        unique_columns_list = None
        ratio_dict = self.data_analyzer.storage[UNIQUE_COLUMN_RATIOS]

        # in order to build a bar chart, we need a list of columns (unique_columns_list) and a dictionary of
        # valid value counts.  The second dict will be used to create a list of counts by value (temp_data_list).

        # check if the variable is BOOLEAN
        if data_type == BOOL_COLUMN_KEY:
            unique_column_count_dict = self.data_analyzer.storage[BOOL_COLUMN_COUNT_KEY][the_column]

            # get the unique column list.  This is the list of valid values.
            unique_columns_list = list(self.data_analyzer.storage[BOOL_COLUMN_COUNT_KEY][the_column].keys())

            # I need to construct a list of the boolean counts
            # loop over all the unique columns
            for column_value in unique_columns_list:
                temp_data_list.append(unique_column_count_dict[column_value])

            # create the bar chart
            self.__create_bar_chart__(unique_columns_list, temp_data_list, data_type, the_column, the_count)

        # we don't generate a box plot for columns with unique values or a LOT of valid values
        # check if the variable is OBJECT
        elif data_type == OBJECT_COLUMN_KEY and ratio_dict[the_column] < .05:
            # only get the dict if we know we don't have too many values.
            unique_column_count_dict = \
                self.data_analyzer.storage[OBJECT_COLUMN_COUNT_KEY][the_column]

            # get the unique column list
            unique_columns_list = self.data_analyzer.storage[UNIQUE_COLUMN_VALUES][the_column]

            # I need to construct a list of the object counts
            # loop over all the unique columns
            for column_value in unique_columns_list:
                temp_data_list.append(unique_column_count_dict[column_value])

            # create the bar chart
            self.__create_bar_chart__(unique_columns_list, temp_data_list, data_type, the_column, the_count)

        # don't know what to do here except lot a message.
        else:
            # log what we didn't create
            self.logger.debug(f"Did NOT create [{the_plot_name}] on "
                              f"plot_storage[{the_column}][{self.plot_type_list[2]}]")

    # shared code to create a bar chart
    def __create_bar_chart__(self, unique_columns_list, temp_data_list, data_type, the_column, the_count):

        # get the plot name to generate
        the_plot_name = self.get_plot_file_name(data_type, PLOT_TYPE_BAR_CHART, the_count)

        # define the plot title.
        the_title = BAR_CHART_BASE_TITLE + the_column

        # clear everything
        plt.clf()

        # generate the plot
        plt.bar(unique_columns_list, temp_data_list)
        plt.title(the_title)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        # log that we're about to save the file
        self.logger.debug(f"About to save plot to [{the_plot_name}]")

        # save the plot
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if the_column not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[the_column] = {self.plot_type_list[2]: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][BOX_PLOT]
            self.plot_storage[the_column][self.plot_type_list[2]] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{the_column}][{self.plot_type_list[0]}]")

    # method to generate a count plot
    def generate_count_plot(self, the_tuple, the_count):
        # perform validations of data_type
        if not isinstance(the_tuple, tuple):
            raise SyntaxError("the_tuple argument is None or incorrect type.")

        # perform validations of x_series
        elif not self.is_valid(the_tuple[0], str):
            raise SyntaxError("x_series is None or incorrect type.")
        elif the_tuple[0] not in self.data_analyzer.the_df:
            raise SyntaxError("x_series is not a valid field on dataframe.")

        # perform validations of y_series
        elif not self.is_valid(the_tuple[1], str):
            raise SyntaxError("y_series is None or incorrect type.")
        elif the_tuple[1] not in self.data_analyzer.the_df:
            raise SyntaxError("y_series is not a valid field on dataframe.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("the_count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("the_count argument must be positive.")

        # variable declaration
        x_series = the_tuple[0]
        y_series = the_tuple[1]
        the_title = BIVARIATE_COUNT_PLOT_TITLE + the_tuple[0] + AMPERSAND + the_tuple[1]
        the_plot_name = copy(self.output_path) + PLOT_TYPE_BIVARIATE_COUNT + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # log that we're about to create count plot
        self.logger.debug(f"creating bi-variate count plot for {x_series} vs. {y_series}.")

        # clear everything
        plt.clf()

        # generate the plot
        sns.countplot(x=x_series, hue=y_series, data=self.data_analyzer.the_df, palette='hls')
        plt.ylabel("Count")
        plt.title(the_title)

        # save the plot
        plt.savefig(the_plot_name)
        plt.close()

        # log the name of the file created
        self.logger.debug(f"created count plot [{the_plot_name}] for [{x_series}][{y_series}]")

        # add to storage
        if PLOT_TYPE_BIVARIATE_COUNT in self.plot_storage:
            # add to storage
            self.plot_storage[PLOT_TYPE_BIVARIATE_COUNT][the_tuple] = the_plot_name
        else:
            # add to storage
            self.plot_storage[PLOT_TYPE_BIVARIATE_COUNT] = {the_tuple: the_plot_name}

    # generate a scatter plot for two series
    def generate_scatter_plot(self, the_tuple, the_count):
        # perform validations of data_type
        if not isinstance(the_tuple, tuple):
            raise SyntaxError("the_tuple argument is None or incorrect type.")

        # perform validations of x_series
        elif not self.is_valid(the_tuple[0], str):
            raise SyntaxError("x_series is None or incorrect type.")
        elif the_tuple[0] not in self.data_analyzer.the_df:
            raise SyntaxError("x_series is not a valid field on dataframe.")

        # perform validations of y_series
        elif not self.is_valid(the_tuple[1], str):
            raise SyntaxError("y_series is None or incorrect type.")
        elif the_tuple[1] not in self.data_analyzer.the_df:
            raise SyntaxError("y_series is not a valid field on dataframe.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("the_count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("the_count argument must be positive.")

        # check if we've created the plot before
        if PLOT_TYPE_SCATTER_CHART in self.plot_storage and the_tuple in self.plot_storage[PLOT_TYPE_SCATTER_CHART]:
            # log that
            self.logger.debug(f"{the_tuple} already present in storage.")
        # create the plot and store it.
        else:
            # variable declaration
            x_series = the_tuple[0]
            y_series = the_tuple[1]
            the_title = SCATTER_PLOT_TITLE + x_series + AMPERSAND + y_series
            the_plot_name = copy(self.output_path) + PLOT_TYPE_SCATTER_CHART + FILE_NAME_CLOSE

            # inject the count into a copy of the internal variable.
            the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

            # log that we're about to create scatter plot
            self.logger.debug(f"creating scatter plot for {the_tuple}.")

            # clear everything
            plt.clf()

            # create scatter plot
            plt.scatter(x=self.data_analyzer.the_df[x_series], y=self.data_analyzer.the_df[y_series])
            plt.title(the_title)
            plt.xlabel(x_series)
            plt.ylabel(y_series)

            # save the plot
            plt.savefig(the_plot_name)
            plt.close()

            # add to storage
            if PLOT_TYPE_SCATTER_CHART in self.plot_storage:
                # add to storage
                self.plot_storage[PLOT_TYPE_SCATTER_CHART][the_tuple] = the_plot_name
            else:
                # add to storage
                self.plot_storage[PLOT_TYPE_SCATTER_CHART] = {the_tuple: the_plot_name}

    # get the correct name for the plot given the data_type, plot_type, and the_count
    def get_plot_file_name(self, data_type, chart_type, the_count):
        # perform validations of arguments
        if not self.is_valid(data_type, str):
            raise SyntaxError("The data_type argument is None or incorrect type.")
        elif data_type not in self.data_type_map:
            raise SyntaxError("The data_type argument is unknown.")
        elif not self.is_valid(chart_type, str):
            raise SyntaxError("The chart_type argument is None or incorrect type.")
        elif chart_type not in self.plot_type_list:
            raise SyntaxError("The chart_type argument is unknown.")
        elif not self.is_valid(the_count, int):
            raise SyntaxError("The count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("The count argument must be positive.")

        # assemble the file name
        the_result = copy(self.output_path) + self.data_type_map[data_type] + UNDERSCORE + chart_type + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_result = re.sub(COUNT_VARIABLE, str(the_count), the_result)

        # log the name of the plot
        self.logger.debug(f"The plot name created is [{the_result}]")

        # return the file name
        return the_result

    # wrapper function of various smaller functions.
    def generate_all_dataset_plots(self, statistics_generator: StatisticsGenerator, the_model_type: str,
                                   the_version=ANALYZE_DATASET_INITIAL):
        # run validation
        if the_version not in [ANALYZE_DATASET_INITIAL, ANALYZE_DATASET_FULL]:
            raise SyntaxError("the_version is None or unknown value.")
        elif the_model_type not in MT_OPTIONS:
            raise SyntaxError("the_model_type is None or unknown value.")
        elif the_version == ANALYZE_DATASET_FULL and not isinstance(statistics_generator, StatisticsGenerator):
            raise SyntaxError("statistics_generator is None or incorrect type.")

        # log that we've been called
        self.logger.debug(f"A request to generate {the_version} plots for a dataset has been made.")

        # for this use case, we're going to need access to a dataset_analyzer
        # and we're going to retrieve the columns for each data type.  We will
        # then loop through a fixed list of each data type and generate the graphs
        # for that data type.  NOTE - I may want to experiment with making the
        # data types to not be fixed.

        # define variables
        the_count = 0  # integer count used to delimit plots for a dataset.
        count_plot_counter = 0  # integer count used for count plots.
        box_plot_counter = 0  # integer count used for box plots.

        # **********************************************************************
        # Start with FLOAT64
        # **********************************************************************

        # retrieve the list of FLOAT64 columns on key FLOAT64_COLUMN_KEY
        the_columns = self.data_analyzer.storage[FLOAT64_COLUMN_KEY]

        # start looping through float columns
        for the_column in the_columns:
            # log the name of the column
            self.logger.debug(f"starting the plot generation for FLOAT column [{the_column}]")

            # generate a histogram
            self.generate_hist(FLOAT64_COLUMN_KEY, the_column, the_count)

            # generate a box plot
            self.generate_boxplot(FLOAT64_COLUMN_KEY, the_column, the_count)

            # generate qq plot
            self.generate_q_q_plot(FLOAT64_COLUMN_KEY, the_column, the_count)

            # increment count
            the_count = the_count + 1

        # **********************************************************************
        # generate plots for INT64
        # **********************************************************************

        # retrieve the list of INT columns on key INT64_COLUMN_KEY
        the_columns = self.data_analyzer.storage[INT64_COLUMN_KEY]

        # reset the count
        the_count = 0

        # start looping through int columns
        for the_column in the_columns:
            # log the name of the column
            self.logger.debug(f"starting the plot generation for INT64 column [{the_column}]")

            # generate the histogram
            self.generate_hist(INT64_COLUMN_KEY, the_column, the_count)

            # generate a box plot
            self.generate_boxplot(INT64_COLUMN_KEY, the_column, the_count)

            # generate qq plot
            self.generate_q_q_plot(INT64_COLUMN_KEY, the_column, the_count)

            # increment count
            the_count = the_count + 1

        # **********************************************************************
        # generate plots for OBJECT
        # **********************************************************************

        # retrieve the list of OBJECT columns on key OBJECT_COLUMN_KEY
        the_columns = self.data_analyzer.storage[OBJECT_COLUMN_KEY]

        # get the dictionary of ratio's
        ratio_dict = self.data_analyzer.storage[UNIQUE_COLUMN_RATIOS]

        # reset the count
        the_count = 0

        # start looping through OBJECT columns
        for the_column in the_columns:
            # log the name of the column
            self.logger.debug(f"starting the plot generation for OBJECT column [{the_column}]")

            # we only generate plots if the ratio is small.  We can't handle too many options.
            if ratio_dict[the_column] < .05:
                # generate a bar chart
                self.generate_bar_chart(OBJECT_COLUMN_KEY, the_column, the_count)

                # increment count
                the_count = the_count + 1

                # check if we should generate bi-variate box plots.
                if the_version == ANALYZE_DATASET_FULL:
                    # log the name of the column
                    self.logger.debug(f"starting bi-variate box plot generation for BOOLEAN column [{the_column}]")

                    # create a list of integer and float columns.  At this point, we should only
                    # have a cleaned dataset.
                    the_column_list = self.data_analyzer.storage[INT64_COLUMN_KEY] + \
                                      self.data_analyzer.storage[FLOAT64_COLUMN_KEY]

                    # start to loop over the_column_list
                    for second_column in the_column_list:
                        # create the tuple
                        the_box_plot_tuple = (the_column, second_column)

                        # generate bi-variate boxplot
                        self.generate_bivariate_boxplot(the_tuple=the_box_plot_tuple, the_count=box_plot_counter)

                        # increment count
                        box_plot_counter = box_plot_counter + 1

            # log that we're skipping the box plot because there are too many options to display
            else:
                # log that we're skipping the column.
                self.logger.debug(f"Skipping plot generation for column [{the_column}] "
                                  f"due to ratio [{ratio_dict[the_column]}].")

        # **********************************************************************
        # generate plots for BOOLEAN
        # **********************************************************************

        # retrieve the list of BOOLEAN columns on key BOOL_COLUMN_KEY
        the_columns = self.data_analyzer.storage[BOOL_COLUMN_KEY]

        # reset the count
        the_count = 0

        # start looping through BOOLEAN columns
        for the_column in the_columns:
            # log the name of the column
            self.logger.debug(f"starting the plot generation for BOOLEAN column [{the_column}]")

            # generate a bar chart
            self.generate_bar_chart(BOOL_COLUMN_KEY, the_column, the_count)

            # check if we should generate bi-variate count plots.
            if the_version == ANALYZE_DATASET_FULL:
                # create an exclusion list of FLOAT64_COLUMN_KEY and INT64_COLUMN_KEY
                exclusion_list = self.data_analyzer.storage[FLOAT64_COLUMN_KEY] + \
                                 self.data_analyzer.storage[INT64_COLUMN_KEY]

                # get the list of tuples associated with the the_column
                the_list = \
                    statistics_generator.filter_tuples_by_column(the_storage=statistics_generator.chi_square_results,
                                                                 the_column=the_column,
                                                                 exclusion_list=exclusion_list)

                # loop over the tuples in the_list
                for the_tuple in the_list:

                    # check if the count plot has already been generated
                    if not self.is_plot_for_tuple_already_created(the_tuple, PLOT_TYPE_BIVARIATE_COUNT):
                        # generate the requisite count plots
                        self.generate_count_plot(the_tuple, count_plot_counter)

                        # log what plot we're creating
                        self.logger.info(f"creating bi-variate count plot for [{the_tuple}][{count_plot_counter}]")

                        # increment count for count plots
                        count_plot_counter = count_plot_counter + 1

                # *********************************************
                #           BI-VARIATE BOX PLOTS
                # *********************************************

                # log the name of the column
                self.logger.debug(f"starting bi-variate box plot generation for BOOLEAN column [{the_column}]")

                # create a list of integer and float columns.
                the_column_list = self.data_analyzer.storage[INT64_COLUMN_KEY] + \
                                  self.data_analyzer.storage[FLOAT64_COLUMN_KEY]

                # start to loop over the_column_list
                for second_column in the_column_list:
                    # create the tuple
                    the_box_plot_tuple = (the_column, second_column)

                    # generate bi-variate boxplot
                    self.generate_bivariate_boxplot(the_tuple=the_box_plot_tuple, the_count=box_plot_counter)

                    # increment count
                    box_plot_counter = box_plot_counter + 1

            # increment count for bar chart
            the_count = the_count + 1

        # check if we should build scatter plots.
        if the_version == ANALYZE_DATASET_FULL:
            # reset the count
            the_count = 0

            # loop through scatter plots
            for the_tuple in statistics_generator.get_list_of_correlations():
                # log the tuple that we're working with
                self.logger.debug(f"generating scatter plot for {the_tuple}")

                # generate scatter plot for current tuple
                self.generate_scatter_plot(the_tuple, the_count)

                # increment count
                the_count = the_count + 1

            # reset the count
            the_count = 0

            # generate the bi-variate count plots
            for the_tuple in statistics_generator.get_chi_squared_results():
                # make sure the tuple is only OBJECT to OBJECT
                if self.data_analyzer.validate_field_type(the_tuple[0], OBJECT_COLUMN_KEY) \
                        and self.data_analyzer.validate_field_type(the_tuple[1], OBJECT_COLUMN_KEY):
                    # log the tuple that we're working with
                    self.logger.debug(f"generating count plot for {the_tuple}")

                    # generate scatter plot for current tuple
                    self.generate_count_plot(the_tuple, count_plot_counter)

                    # increment count
                    count_plot_counter = count_plot_counter + 1

            # log that we're starting work for the joint plots
            self.logger.debug("About to generate joint plots.")

            # get the list of all correlations.  By definition, this does not include OBJECT.
            list_of_tuples = statistics_generator.get_list_of_correlations(the_type=ALL_CORRELATIONS)

            # retrieve the list of BOOLEAN columns on key BOOL_COLUMN_KEY
            exclude_list = self.data_analyzer.storage[BOOL_COLUMN_KEY]

            # make a copy of the list_of_tuples
            the_result = list_of_tuples.copy()

            # loop over the exclude list
            for element in exclude_list:
                # invoke the method
                tuples_to_be_removed_list = get_tuples_from_list_with_specific_field(list_of_tuples, element)

                # remove those from the_result
                for next_tuple in tuples_to_be_removed_list:
                    # check if next_tuple is in the_result
                    if next_tuple in the_result:
                        # remove it if it is
                        the_result.remove(next_tuple)

            # reset the count
            the_count = 0

            # loop through joint plots
            for the_tuple in the_result:
                # log the tuple that we're working with
                self.logger.debug(f"generating joint plot for [{the_tuple[0]}][{the_tuple[1]}]")

                # generate scatter plot for current tuple
                self.generate_joint_plot(the_tuple, the_count)

                # increment count
                the_count = the_count + 1

        # **********************************************************************
        # generate plots for MODEL(s)
        # **********************************************************************
        if the_version == ANALYZE_DATASET_FULL:
            # reset the count
            the_count = 0

            # so, we need to generate residual plots for the "initial" model, which
            # will encompass the plots for the "final" model.  Thus, we only need
            # to generate plots for the "initial" model.  Please note that this will create
            # variables in the internal storage that are for encoded fields, and not the original fields.

            # get the initial model
            the_initial_model = self.data_analyzer.linear_model_storage[LM_INITIAL_MODEL]

            # get all the features from the initial model.
            the_features_list = the_initial_model.get_encoded_variables(
                the_target_column=the_initial_model.get_the_result().get_the_target_variable())

            # if we have a linear or logistic model, generate required plots
            if the_model_type == MT_LOGISTIC_REGRESSION or the_model_type == MT_LINEAR_REGRESSION:
                # loop over all the features in the_features_list
                for the_feature in the_features_list:
                    # log the current feature
                    self.logger.debug(f"Creating residual plot for [{the_feature}]")

                    # generate the standardized residual plot
                    self.generate_standardized_residual_plot(the_model=the_initial_model.get_the_result().model,
                                                             the_encoded_df=the_initial_model.encoded_df,
                                                             the_column=the_feature,
                                                             the_count=the_count)

                    # generate the QQ residual plot
                    # the_model_type, the_column, the_model, the_count
                    self.generate_q_q_plot_for_residuals(the_model_type=LM_INITIAL_MODEL,
                                                         the_column=the_feature,
                                                         the_model=the_initial_model.get_the_result().model,
                                                         the_count=the_count)

                    # generate long odds plot
                    self.generate_long_odds_linear_plot(
                        the_target_variable=the_initial_model.get_the_result().get_the_target_variable(),
                        the_independent_variable=the_feature,
                        the_df=the_initial_model.encoded_df,
                        the_count=the_count)

                    # increment the count
                    the_count = the_count + 1
            # generate required plots for KNN model
            elif the_model_type == MT_KNN_CLASSIFICATION:
                the_initial_model.get_the_result().get_model()

                # generate auc/roc plot
                self.generate_auc_roc_plot_knn(the_model=the_initial_model.get_the_result().get_model(),
                                               x_test=the_initial_model.get_the_result().the_f_df_test,
                                               y_test=the_initial_model.get_the_result().the_t_var_test,
                                               the_count=1)

    # generate scree plots
    def generate_scree_plots(self, pca_analysis):
        # run validation
        if not self.is_valid(pca_analysis, PCA_Analysis):
            raise SyntaxError("The PCA_Analysis was None or incorrect type.")

        # variable declaration
        the_original_df = pca_analysis.get_original_df()
        the_norm_df = pca_analysis.get_normalized_df()
        cov_matrix = np.dot(the_norm_df.T, the_norm_df) / the_original_df.shape[0]
        eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for
                       eigenvector in pca_analysis.the_pca.components_]

        # log that we've been called
        self.logger.debug("A request to generate a scree plot has been made.")

        # generate the file name
        the_plot_name = copy(self.output_path) + PLOT_TYPE_SCREE + FILE_NAME_CLOSE
        the_plot_name = re.sub(COUNT_VARIABLE, str(1), the_plot_name)

        # clear everything
        plt.clf()

        # generate the plot of the explained variance by number of components
        plt.plot(pca_analysis.the_pca.explained_variance_ratio_)
        plt.xlabel('number of components')
        plt.ylabel('explained variance')

        # save the plot to the file system.
        plt.savefig(the_plot_name)

        # clear everything
        plt.clf()

        # generate the file name for the second plot
        the_plot_name = copy(self.output_path) + PLOT_TYPE_SCREE + FILE_NAME_CLOSE
        the_plot_name = re.sub(COUNT_VARIABLE, str(2), the_plot_name)

        plt.plot(eigenvalues)
        plt.xlabel('number of components')
        plt.ylabel('eigenvalue')

        # save the plot to the file system.
        plt.savefig(the_plot_name)

        # log what we created
        self.logger.debug(f"created scree plot [{the_plot_name}]")

    # generate heatmap of correlations
    def generate_correlation_heatmap(self, the_list_of_corr):
        # run validation
        if not self.is_valid(the_list_of_corr, list):
            raise SyntaxError("The list of correlations was None or incorrect type.")

        # log that we've been called
        self.logger.debug("A request to generate a heatmap of correlations has been made.")

        # variable declaration
        the_plot_name = None
        the_df = None
        column_dict = {}
        the_title = HEATMAP_CORR_CHART_BASE_TITLE  # the plot title.

        # loop over the list of correlation tuples.  We need a simple list of column names for the next step
        for current_corr in the_list_of_corr:
            # check if the first column is in the column_dict
            if current_corr[0] not in column_dict:
                # add the column as a key and value
                column_dict[current_corr[0]] = current_corr[0]

            # check if the second column is in the column_dict
            if current_corr[1] not in column_dict:
                # add the column as a key and value
                column_dict[current_corr[1]] = current_corr[1]

        # log the list of columns
        self.logger.debug(f"The columns in the heatmap are {list(column_dict.keys())}")

        # grab a subset from the dataset analyzer matching the columns from the_list_of_corr
        the_df = self.data_analyzer.the_df[list(column_dict.keys())]

        # generate the file name
        the_plot_name = copy(self.output_path) + PLOT_TYPE_HEATMAP + FILE_NAME_CLOSE
        the_plot_name = re.sub(COUNT_VARIABLE, str(1), the_plot_name)

        # clear everything
        plt.clf()

        # make sure we've initialized correctly
        if len(the_df.corr()) == 0:
            raise RuntimeError("a call to df.corr() returned zero-sized array.")

        # generate the heatmap
        sns.heatmap(the_df.corr(), annot=True, fmt=".1f", cmap=sns.diverging_palette(10, 400, as_cmap=True))

        # add title
        plt.title(the_title)

        # this command forces the full names to show-up.  If you don't invoke this command, longer variable names
        # will be cut off and the general image will be hard to read.
        plt.tight_layout()

        # save the plot to the file system.
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if GENERAL not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[GENERAL] = {self.plot_type_list[4]: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][PLOT_TYPE_HEATMAP]
            self.plot_storage[GENERAL][self.plot_type_list[4]] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{GENERAL}][{self.plot_type_list[4]}]")

    # generate heatmap of a confusion matrix
    def generate_confusion_matrix_heatmap(self, confusion_matrix: ndarray, the_count: int):
        # run validation
        if not self.is_valid(confusion_matrix, ndarray):
            raise SyntaxError("confusion_matrix was None or incorrect type.")
        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("the_count argument is None or incorrect type.")
        elif the_count < 0:
            raise ValueError("the_count argument must be positive.")

        # log that we've been called
        self.logger.debug("A request to generate heatmap of a confusion matrix has been made.")

        # variable declaration
        the_plot_name = None
        the_df = None
        the_title = HEATMAP_CM_CHART_BASE_TITLE  # the plot title.

        # clear everything
        plt.clf()

        # generate names, counts, and percentages
        group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts = ['{0: 0.0f}'.format(value) for value in confusion_matrix.flatten()]
        group_percentages = ["{0: .2%}".format(value)
                             for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

        # generate labels
        labels = [f'{v1}\n {v2}\n {v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        # generate the file name
        the_plot_name = copy(self.output_path) + PLOT_TYPE_HEATMAP + FILE_NAME_CLOSE
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # clear everything
        plt.clf()

        # generate the heatmap
        sns.heatmap(confusion_matrix, annot=labels, fmt="", cmap="Blues")

        # add title
        plt.title(the_title)

        # this command forces the full names to show-up.  If you don't invoke this command, longer variable names
        # will be cut off and the general image will be hard to read.
        plt.tight_layout()

        # save the plot to the file system.
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if GENERAL not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[GENERAL] = {PLOT_TYPE_CM_HEATMAP: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][HIST]
            self.plot_storage[GENERAL][PLOT_TYPE_CM_HEATMAP] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{GENERAL}][{PLOT_TYPE_CM_HEATMAP}]")

    # generate joint plot
    def generate_joint_plot(self, the_tuple, the_count):
        # perform validations of data_type
        if not isinstance(the_tuple, tuple):
            raise SyntaxError("the_tuple argument is None or incorrect type.")

        # perform validations of x_series
        elif not self.is_valid(the_tuple[0], str):
            raise SyntaxError("x_series is None or incorrect type.")
        elif the_tuple[0] not in self.data_analyzer.the_df:
            raise SyntaxError("x_series is not a valid field on dataframe.")

        # perform validations of y_series
        elif not self.is_valid(the_tuple[1], str):
            raise SyntaxError("y_series is None or incorrect type.")
        elif the_tuple[1] not in self.data_analyzer.the_df:
            raise SyntaxError("y_series is not a valid field on dataframe.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("the_count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("the_count argument must be positive.")

        # variable declaration
        x_series = the_tuple[0]
        y_series = the_tuple[1]
        the_title = JOINT_PLOT_TITLE + the_tuple[0] + AMPERSAND + the_tuple[1]
        the_plot_name = copy(self.output_path) + PLOT_TYPE_JOINT_PLOT + FILE_NAME_CLOSE
        the_df = self.data_analyzer.the_df

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # log that we're about to create scatter plot
        self.logger.debug(f"creating joint plot for {x_series} vs. {y_series}.")

        # clear everything
        plt.clf()

        # generate the plot
        g = sns.jointplot(x=x_series, y=y_series, data=the_df, kind='reg', scatter=False)
        sns.scatterplot(y=y_series, x=x_series, data=the_df, ax=g.ax_joint)
        plt.suptitle(the_title)

        # save the plot
        plt.savefig(the_plot_name)
        plt.close()

        # log the name of the file created
        self.logger.debug(f"created joint plot [{the_plot_name}] for [{x_series}][{y_series}]")

        # add to storage
        if PLOT_TYPE_JOINT_PLOT in self.plot_storage:
            # add to storage
            self.plot_storage[PLOT_TYPE_JOINT_PLOT][the_tuple] = the_plot_name
        else:
            # add to storage
            self.plot_storage[PLOT_TYPE_JOINT_PLOT] = {the_tuple: the_plot_name}

    # generate a q-q plot for a series
    def generate_q_q_plot(self, data_type, the_column, the_count):
        # perform validations of data_type
        if not self.is_valid(data_type, str):
            raise SyntaxError("data_type argument is None or incorrect type.")
        elif data_type not in self.data_type_map:
            raise SyntaxError("The data_type argument is unknown.")

        # perform validations of the_column
        elif not self.is_valid(the_column, str):
            raise SyntaxError("The column is None or incorrect type.")
        elif the_column not in self.data_analyzer.storage[data_type]:
            raise SyntaxError("The column was not found on list for the data_type.")

        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("The count argument is None or incorrect type.")
        elif the_count < 0:
            raise SyntaxError("The count argument must be positive.")

        # log what we're going to do
        self.logger.debug(f"Generating QQ Plot for data_type[{data_type}] "
                          f"the_column[{the_column}] the_count[{the_count}]")

        # variable declaration
        df_column = self.data_analyzer.the_df[the_column]  # data column
        the_title = Q_Q_PLOT_TITLE + the_column  # the plot title
        the_plot_name = self.get_plot_file_name(data_type, PLOT_TYPE_Q_Q_PLOT, the_count)  # plot name

        # clear everything
        plt.clf()

        # generate the plot
        stats.probplot(df_column, dist="norm", plot=pylab)

        # add the title
        plt.title(the_title)

        # log that we're about to save the file
        self.logger.debug(f"About to save plot to [{the_plot_name}]")

        # save
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if the_column not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[the_column] = {self.plot_type_list[7]: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][HIST]
            self.plot_storage[the_column][self.plot_type_list[7]] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{the_column}][{self.plot_type_list[7]}]")

    # generate a q-q plot for residuals of a model feature
    def generate_q_q_plot_for_residuals(self, the_model_type, the_column, the_model, the_count):
        # perform validations of the_column
        if not isinstance(the_model_type, str):
            raise SyntaxError("the_model_type is None or incorrect type.")
        elif the_model_type not in LM_VALID_TYPES:
            raise ValueError("the_model_type is not a valid value.")
        elif not self.is_valid(the_column, str):
            raise SyntaxError("The column is None or incorrect type.")
        elif the_column not in self.data_analyzer.get_model(the_model_type).encoded_df.columns:
            raise ValueError("the_column is not a valid column.")
        elif not isinstance(the_model, RegressionResultsWrapper):
            raise SyntaxError("the_model argument is None or incorrect type.")
        # perform validations of the_count
        elif not self.is_valid(the_count, int):
            raise SyntaxError("The count argument is None or incorrect type.")
        elif the_count < 0:
            raise ValueError("The count argument must be positive.")

        # log what we're going to do
        self.logger.debug(f"Generating QQ Plot of residuals for the_column[{the_column}] the_count[{the_count}]")

        # variable declaration
        df_column = self.data_analyzer.get_model(the_model_type).encoded_df[the_column]
        the_title = Q_Q_RESIDUAL_PLOT_TITLE + the_column  # the plot title
        the_plot_name = copy(self.output_path) + PLOT_TYPE_Q_Q_RESIDUAL_PLOT + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # clear everything
        plt.clf()

        # generate the plot
        stats.probplot(df_column, dist="norm", plot=pylab)

        # add the title
        plt.title(the_title)

        # log that we're about to save the file
        self.logger.debug(f"About to save plot to [{the_plot_name}]")

        # save
        plt.savefig(the_plot_name)
        plt.close()

        # add to storage
        if PLOT_TYPE_Q_Q_RESIDUAL_PLOT in self.plot_storage:
            # add to storage
            self.plot_storage[PLOT_TYPE_Q_Q_RESIDUAL_PLOT][the_column] = the_plot_name
        else:
            # add to storage
            self.plot_storage[PLOT_TYPE_Q_Q_RESIDUAL_PLOT] = {the_column: the_plot_name}

    # determine if internal storage already has a specific plot
    def is_plot_for_tuple_already_created(self, the_tuple, the_type) -> bool:
        # perform validation
        if not isinstance(the_tuple, tuple):
            raise SyntaxError("the_tuple argument is None or incorrect type.")
        elif the_type not in self.plot_type_list:
            raise SyntaxError("the_type argument is None or unknown.")

        # variable declaration
        the_result = False

        # log what we're checking
        self.logger.debug(f"checking if [{the_type}] has been created for {the_tuple}.")

        # if this is the first time we're creating a plot of a specific type, we won't have a key in storage
        # at all. Thus, we need to first check if the_type is present as a key.
        if the_type in self.plot_storage:
            # now we know we've created a plot of the_type before, so let's check if the tuple is present.
            if the_tuple in self.plot_storage[PLOT_TYPE_BIVARIATE_COUNT]:
                the_result = True

                # please note, all other outcomes are False

        # return
        return the_result

    # generate standardized residual plot for linear model
    def generate_standardized_residual_plot(self, the_model, the_encoded_df, the_column, the_count):
        # run validations
        if not isinstance(the_model, RegressionResultsWrapper):
            raise SyntaxError("the_model argument is None or incorrect type.")
        elif not isinstance(the_encoded_df, DataFrame):
            raise SyntaxError("the_encoded_df argument is None or incorrect type.")
        elif not isinstance(the_column, str):
            raise SyntaxError("the_column argument is None or incorrect type.")
        elif the_column not in the_encoded_df.columns.to_list():
            raise ValueError("the_column argument is not present in the_encoded_df.")
        elif not self.is_valid(the_count, int):
            raise SyntaxError("the_count argument is None or incorrect type.")
        elif the_count < 0:
            raise ValueError("the_count argument must be positive.")

        # generate the plot name for MonthlyCharge residuals
        the_plot_name = copy(self.output_path) + PLOT_TYPE_STD_RESIDUAL + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # create instance of influence
        influence = the_model.get_influence()

        # check if we can generate a residual plot for this model
        if hasattr(influence, 'resid_studentized_internal'):
            # get the standard residuals
            standardized_residuals = influence.resid_studentized_internal

            # clear everything
            plt.clf()

            # log that we're about to save the file
            self.logger.debug(f"About to save plot to [{the_plot_name}]")

            # generate the plot
            plt.scatter(the_encoded_df[the_column], standardized_residuals)
            plt.xlabel(the_column)
            plt.ylabel(STANDARDIZED_RESIDUALS_LABEL)
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

            # generate the_title
            the_title = STANDARDIZED_RESIDUAL_PLOT_TITLE + the_column

            # add the title
            plt.title(the_title)

            plt.savefig(the_plot_name)
            plt.close()

            # log the name of the file created
            self.logger.debug(f"created standardized residual plot [{the_plot_name}] for [{the_column}].")

            # check if initialization of the storage on key data_type is required.
            # the 1st level is column_name, the 2nd level is plot type, value is path
            if the_column not in self.plot_storage:
                # create the second level on plot_storage.
                self.plot_storage[the_column] = {PLOT_TYPE_STD_RESIDUAL: the_plot_name}
            # the column is present, create second level.
            else:
                # store the plot under [the_column][PLOT_TYPE_STD_RESIDUAL]
                self.plot_storage[the_column][PLOT_TYPE_STD_RESIDUAL] = the_plot_name

            # log what we created
            self.logger.debug(f"created [{the_plot_name}] on plot_storage[{the_column}][{self.plot_type_list[8]}]")

    # generate long odds linear plot
    def generate_long_odds_linear_plot(self, the_target_variable: str, the_independent_variable: str,
                                       the_df: DataFrame, the_count: int):
        # run validations
        if not isinstance(the_target_variable, str):
            raise SyntaxError("the_target_variable is None or incorrect type.")
        elif not isinstance(the_independent_variable, str):
            raise SyntaxError("the_independent_variable is None or incorrect type.")
        elif not isinstance(the_df, DataFrame):
            raise SyntaxError("the_df is None or incorrect type.")
        elif not isinstance(the_count, int):
            raise SyntaxError("the_count in None or incorrect type.")
        elif the_target_variable not in the_df.columns:
            raise ValueError("the_target_variable is not in the_df.")
        elif the_independent_variable not in the_df.columns:
            raise ValueError("the_independent_variable is not in the_df.")
        elif the_count < 0:
            raise ValueError("The count argument must be positive.")

        # log that we've been called
        self.logger.debug(f"generating long odds plot for target[{the_target_variable}] "
                          f"and [{the_independent_variable}]")

        # variable declaration
        the_title = LONG_ODDS_PLOT_TITLE + the_target_variable + AMPERSAND + the_independent_variable
        the_plot_name = copy(self.output_path) + PLOT_TYPE_LONG_ODDS + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # clear everything
        plt.clf()

        # generate the plot
        the_plot = sns.regplot(x=the_independent_variable, y=the_target_variable, data=the_df, logistic=True)

        # add the title
        the_plot.set_title(the_title)

        # log that we're about to save the file
        self.logger.debug(f"About to save plot to [{the_plot_name}]")

        # save the image.
        the_plot.figure.savefig(the_plot_name)

        # add to storage
        if PLOT_TYPE_LONG_ODDS in self.plot_storage:
            # add to storage
            self.plot_storage[PLOT_TYPE_LONG_ODDS][the_independent_variable] = the_plot_name
        else:
            # add to storage
            self.plot_storage[PLOT_TYPE_LONG_ODDS] = {the_independent_variable: the_plot_name}

    # generate AUC/ROC plot for KNN model
    def generate_auc_roc_plot_knn(self, the_model: KNeighborsClassifier,
                                  x_test: numpy.ndarray, y_test: DataFrame, the_count: int):
        # run validations
        if not isinstance(the_model, KNeighborsClassifier):
            raise SyntaxError("the_model is None or incorrect type.")
        elif not isinstance(x_test, DataFrame):
            raise SyntaxError("x_test is None or incorrect type.")
        elif not isinstance(y_test, DataFrame):
            raise SyntaxError("y_test is None or incorrect type.")
        elif not isinstance(the_count, int):
            raise SyntaxError("the_count is None or incorrect type.")
        elif the_count < 0:
            raise ValueError("The count argument must be positive.")

        # create the plot name
        the_plot_name = copy(self.output_path) + PLOT_TYPE_ROC_AUC + FILE_NAME_CLOSE

        # inject the count into a copy of the internal variable.
        the_plot_name = re.sub(COUNT_VARIABLE, str(the_count), the_plot_name)

        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test

        # clear everything
        plt.clf()

        # run calculations
        y_scores = the_model.predict_proba(x_test)
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)

        # generate plot
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')

        # scale the drawing
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # add
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # create title
        plt.title("AUC & ROC Curve")

        # save
        plt.savefig(the_plot_name)
        plt.close()

        # check if initialization of the storage on key data_type is required.
        # the 1st level is column_name, the 2nd level is plot type, value is path
        if GENERAL not in self.plot_storage:
            # create the second level on plot_storage.
            self.plot_storage[GENERAL] = {PLOT_TYPE_ROC_AUC: the_plot_name}
        # the column is present, create second level.
        else:
            # store the plot under [the_column][PLOT_TYPE_ROC_AUC]
            self.plot_storage[GENERAL][PLOT_TYPE_ROC_AUC] = the_plot_name

        # log what we created
        self.logger.debug(f"created [{the_plot_name}] on plot_storage[{GENERAL}][{PLOT_TYPE_ROC_AUC}]")
