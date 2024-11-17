import logging
import pandas as pd
import warnings

from distfit import distfit
from scipy.stats import chi2_contingency, chi2
from model.BaseModel import BaseModel
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.constants.DatasetConstants import INT64_COLUMN_KEY, OBJECT_COLUMN_KEY, COLUMN_KEY, FLOAT64_COLUMN_KEY
from model.constants.StatisticsConstants import ALL_CORRELATIONS, CORR_EXCEED_LEVEL, DIST_NAME, DIST_PARAMETERS
from util.CommonUtils import are_tuples_the_same, get_tuples_from_list_with_specific_field


# class for generating output file
class StatisticsGenerator(BaseModel):

    # init method
    def __init__(self, the_dsa):
        super().__init__()

        # run validation
        if not isinstance(the_dsa, DatasetAnalyzer):
            raise AttributeError("the_dsa was None or incorrect type.")

        # define logger
        self.logger = logging.getLogger(__name__)

        # capture internal variables
        self.the_dsa = the_dsa
        self.relevant_corr_storage = []         # storage for correlations with absolute value > the_level
        self.all_corr_storage = []              # storage for all correlations, regardless of
        self.contingency_tables = []
        self.chi_square_results = []
        self.corr_level = 0.0
        self.distribution = {}

        # log that init has been called
        self.logger.debug("A request to create a StatisticsGenerator has been made.")

    # find correlations that reach a specific threshold
    def find_correlations(self, the_level=0.5):
        """
        Find correlations between two columns.  If the correlation exceed the_level, then add it to the
        internal storage of things we care about.  The code does a pairwise calculation of all internal
        variables on the central dataframe, and only captures those that reach a specific threshold defined
        by the_level argument.  Afterward the results are stored internally on the self.corr_storage variable.
        Please note that no categorical variables are considered.
        :param the_level: the default level of correlation we consider significant.
        :return: None
        """
        # log what we're attempting to do
        self.logger.debug(f"Attempting to find correlations between all columns. Level is {the_level}.")

        # get the main dataframe
        the_df = self.the_dsa.the_df

        # get the list of all columns
        all_columns = the_df.columns.to_list()

        # capture the correlation level
        self.corr_level = the_level

        # check if the index of the_df is in all_columns
        if the_df.index.name in all_columns:
            # remove the index from all_columns
            all_columns.remove(the_df.index.name)

            # log that we removed the index from all_columns
            self.logger.debug(f"removing [{the_df.index.name}] from list of all columns.")

        # get the list of type OBJECT
        object_list = self.the_dsa.storage[OBJECT_COLUMN_KEY]

        # remove all the columns of type OBJECT from all_columns
        all_columns = [i for i in all_columns if i not in object_list]

        # log the cleaned list of columns
        self.logger.debug(f"the clean list of columns is {all_columns}.")

        # loop through all_columns
        for current_column in all_columns:
            # create a copy of all_columns
            temp_list = all_columns.copy()

            # remove the current_column from temp_list.  We know that correlation will be 1.
            temp_list.remove(current_column)

            # now, loop through the temp_list
            for current_temp_column in temp_list:
                # find the correlation between current_column and current_temp_column
                the_corr = the_df[current_column].corr(the_df[current_temp_column])

                # log the correlation that we found
                self.logger.debug(f"the correlation between {current_column} and {current_temp_column} is {the_corr}.")

                # create a tuple of the columns.  Due to inconsistencies with the corr() function
                # you need to round the digits as the 10th or 11th digit could undulate slightly.
                current_tuple = (current_column, current_temp_column, round(the_corr, 6))

                # check if the correlation is greater than the_level
                if abs(the_corr) > the_level:
                    # add to storage just for abs(correlations) that exceed the_level
                    self.add_to_storage(self.relevant_corr_storage, current_tuple)

                # no matter what, add the correlation to the overall storage
                self.add_to_storage(self.all_corr_storage, current_tuple)

    # get list of correlations
    def get_list_of_correlations(self, the_type=CORR_EXCEED_LEVEL) -> list:
        """
        get the list of correlations found to be relevant.
        :return: list of correlations stored as a tuple
        """
        # variable declaration
        the_result = None

        # log that the method has been called
        self.logger.debug(f"A request to get {the_type} has been made.")

        if the_type == CORR_EXCEED_LEVEL:
            the_result = self.relevant_corr_storage
        elif the_type == ALL_CORRELATIONS:
            the_result = self.all_corr_storage
        # return
        return the_result

    # calculate chi-squared results between variables
    def find_chi_squared_results(self, the_level=0.05):
        # log what we're attempting to do
        self.logger.debug(f"Calculating chi-squared results.")

        # variable declaration
        all_columns = self.the_dsa.storage[COLUMN_KEY]
        the_df = self.the_dsa.the_df
        the_alpha = 1 - the_level
        int_list = self.the_dsa.storage[INT64_COLUMN_KEY]
        float_list = self.the_dsa.storage[FLOAT64_COLUMN_KEY]

        # remove the int64 and float variables from all_columns
        all_columns = [i for i in all_columns if i not in int_list + float_list]

        # log the alpha level
        self.logger.debug(f"the alpha level is {the_alpha}")

        # loop through the categorical_list to find categorical to categorical relationships
        for current_column in all_columns:
            # create a copy of categorical_list
            # temp_list = list(all_columns.keys()).copy()
            temp_list = all_columns.copy()

            # remove the current_column from temp_list.  We don't need to test a column against itself.
            temp_list.remove(current_column)

            # now, loop through the temp_list
            for current_temp_column in temp_list:
                # create a contingency table.  The current_column is the target, the current_temp_column is the
                # predictor
                contingency = pd.crosstab(the_df[current_column], the_df[current_temp_column])

                # run the chi squared test
                c, p_value, dof, expected = chi2_contingency(contingency)

                # calculate the critical statistic
                the_critical = chi2.ppf(the_level, dof)

                # log the final calculation
                self.logger.debug(f"for [{current_column}][{current_temp_column}] "
                                  f"the critical statistic is {the_critical}.")

                # create a tuple of the p value.
                chi_squared_tuple = (current_column, current_temp_column, round(p_value, 6))

                # add to storage
                self.add_to_storage(self.chi_square_results, chi_squared_tuple)

    # get list of chi-squared results
    def get_chi_squared_results(self) -> list:
        """
        get the list of chi-squared results for categorical variables
        :return: list of chi-squared stored as a tuple
        """
        # log that the method has been called
        self.logger.debug("A request to get chi-squared results has been made.")

        # return the list of correlations
        return self.chi_square_results

    # add a tuple to storage if it is unique
    def add_to_storage(self, storage_list, current_tuple):
        """
        Add a tuple to storage if it requires an update.
        :param storage_list: the list that represents the storage
        :param current_tuple: the current tuple to evaluate
        :return: None
        """
        # run validations
        if not isinstance(storage_list, list):
            raise AttributeError("storage_list is None or incorrect type.")
        elif not isinstance(current_tuple, tuple):
            raise AttributeError("current_tuple is None or incorrect type.")

        # check if the storage_list is empty
        if len(storage_list) == 0:
            self.logger.debug(f"Adding first tuple between [{current_tuple[0]}] and [{current_tuple[1]}]")

            # add to storage
            storage_list.append(current_tuple)
        # the storage list is not empty
        else:
            # set the add_to_storage_flag
            add_to_storage_flag = False

            # loop over the storage_list.  Only add the tuple if it doesn't exist on the storage.
            for current_element in storage_list:
                # check if the tuples are the same
                if are_tuples_the_same(current_element, current_tuple):
                    # log that it already exists
                    self.logger.debug(f"Already have tuple between [{current_tuple[0]}] and [{current_tuple[1]}]")

                    # set back to False
                    add_to_storage_flag = False

                    # jump out of current loop, we don't need to go further.
                    break
                else:
                    # log that we are adding the tuple
                    self.logger.debug(f"adding have tuple between [{current_tuple[0]}] and [{current_tuple[1]}]")

                    # set the flag to True
                    add_to_storage_flag = True

            # add to storage
            if add_to_storage_flag:
                # add to storage
                storage_list.append(current_tuple)

                # set the add_to_storage_flag back to False
                add_to_storage_flag = False

    # get list of tuples of only a certain type
    def get_list_of_variable_relationships_of_type(self, the_type) -> list:
        # run validations
        if not isinstance(the_type, str):
            raise AttributeError("the_type is None or invalid type.")
        elif not self.the_dsa.is_data_type_valid(the_type):
            raise AttributeError("the_type is invalid.")

        # variable declarations
        the_result = []

        # loop through all the chi-squared results, return only those of type the_type
        for the_tuple in self.get_chi_squared_results():
            # make sure the tuple is only OBJECT to OBJECT
            if self.the_dsa.validate_field_type(the_tuple[0], the_type) \
                    and self.the_dsa.validate_field_type(the_tuple[1], the_type):

                # add the tuple to the_results
                the_result.append(the_tuple)

        # return
        return the_result

    # filter tuples by column, and exclude certain columns
    def filter_tuples_by_column(self, the_storage, the_column, exclusion_list=None) -> list:
        # run validations
        if the_storage not in [self.all_corr_storage, self.relevant_corr_storage, self.chi_square_results]:
            raise AttributeError("the_storage is None or invalid type.")
        elif the_column not in self.the_dsa.the_df.columns:
            raise AttributeError("the_column is not present on underlying dataframe.")

        # log that we've been called
        self.logger.debug(f"filtering storage by {the_column} with exclusion list [{exclusion_list}]")

        # variable declaration
        the_result = the_storage.copy()

        # filter the list by the column, meaning that the column must appear in one of the two fields in a tuple
        the_result = get_tuples_from_list_with_specific_field(the_result, the_column)

        # check if the exclusion list is not None
        if exclusion_list is not None:
            # loop over the exclude list
            for element in exclusion_list:
                # invoke the method
                tuples_to_be_removed_list = get_tuples_from_list_with_specific_field(the_result, element)

                # remove those from the_result
                for next_tuple in tuples_to_be_removed_list:
                    # check if next_tuple is in the_result
                    if next_tuple in the_result:
                        # remove it if it is
                        the_result.remove(next_tuple)

        # return
        return the_result

    # get the level used to filter correlations on
    def get_correlation_level(self) -> float:
        return self.corr_level

    # fit a theoretical distribution to a specific column
    def fit_theoretical_distribution(self, the_column):
        # run validations
        if not isinstance(the_column, str):
            raise SyntaxError("the_column is None or incorrect type.")
        elif the_column not in self.the_dsa.the_df:
            raise SyntaxError("the_column is not present in the underlying dataframe.")

        # suppress warnings.
        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        warnings.filterwarnings('ignore', 'overflow encountered in divide')
        warnings.filterwarnings('ignore', 'overflow encountered in reduce')
        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
        warnings.filterwarnings('ignore', 'invalid value encountered in subtract')
        warnings.filterwarnings('ignore', 'invalid value encountered in log')

        # log that we've been called
        self.logger.debug(f"attempting to fit [{the_column}] to a theoretical distribution.")

        # get the series for the column, and convert to a numpy array.
        the_series = self.the_dsa.the_df[the_column].to_numpy()

        # get the series name
        the_series_name = self.the_dsa.the_df[the_column].name

        # initialize
        theo_dist = distfit()

        # fit distributions
        theo_dist.fit_transform(the_series, verbose=0)

        # predict distributions
        theo_dist.predict(the_series)

        # store the distribution
        self.distribution[the_series_name] = {DIST_NAME: theo_dist.summary['name'][0],
                                              DIST_PARAMETERS: theo_dist.summary['params'][0]}

    # fit distributions to all numeric columns
    def fit_theoretical_dist_to_all_columns(self):
        # log what we've been asked to do
        self.logger.debug("fitting distribution to all numerical columns.")

        # get the list of INT and FLOAT columns
        the_list = self.the_dsa.storage[INT64_COLUMN_KEY] + self.the_dsa.storage[FLOAT64_COLUMN_KEY]

        # get a reference to the internal dataframe
        the_df = self.the_dsa.the_df

        # loop over the list
        for the_column in the_list:
            # log the current column
            self.logger.debug(f"calculating dist for column[{the_column}]")

            # determine the distribution
            self.fit_theoretical_distribution(the_column=the_column)

