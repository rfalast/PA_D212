import logging
import pandas as pd

from pandas import DataFrame
from sklearn.decomposition import PCA
from model.BaseModel import BaseModel
from model.analysis.DatasetAnalyzer import DatasetAnalyzer


# PCA Analysis tools
class PCA_Analysis(BaseModel):
    # constants
    Z_SCORE = "_z_score"

    # init method
    def __init__(self, dataset_analyzer, column_dict):
        super().__init__()

        # perform validation
        if not self.is_valid(dataset_analyzer, DatasetAnalyzer):
            raise ValueError("the DatasetAnalyzer is None or incorrect type.")
        elif not self.is_valid(column_dict, dict):
            raise ValueError("the column_dict is None or incorrect type.")

        # log that we've been instantiated
        logging.debug("An instance of PCA_Analysis has been created.")

        # define internal variables
        self.dataset_analyzer = dataset_analyzer
        self.column_dict = column_dict
        self.the_pca_df = None
        self.the_pca = None
        self.pca_loadings = None

        # we want to capture a truncated version of both the original dataframe, and the normalized
        # dataframe.  The two truncated dataframes should only contain columns present in the column_dict
        self.original_df = self.__truncate_dataframe__(dataset_analyzer.the_df)
        self.normalized_df = self.__truncate_dataframe__(dataset_analyzer.the_normal_df)

    # get the truncated version of the original dataframe
    def get_original_df(self):
        return self.original_df

    # get the normalized dataframe
    def get_normalized_df(self):
        return self.normalized_df

    # perform analysis
    def perform_analysis(self):
        # log that we were called
        logging.debug("A request to perform PCA Analysis has been made.")

        # variable declaration
        the_columns = None

        # log the shape
        logging.debug(f"shape for PCA is [{self.get_original_df().shape[1]}]")

        # create an instance of a PCA sized to the truncated dataframe
        self.the_pca = PCA(n_components=self.get_original_df().shape[1])

        # call fit on the truncated normalized dataframe
        self.the_pca.fit(self.get_normalized_df())

        # capture the columns label from the column_dict
        the_columns = [*self.column_dict]

        # invoke pca.transform()
        self.the_pca_df = pd.DataFrame(data=self.the_pca.transform(self.get_normalized_df()), columns=the_columns)

        # create the loadings
        self.pca_loadings = pd.DataFrame(data=self.the_pca.components_.T,
                                         columns=the_columns,
                                         index=self.original_df.columns)

        # log that we are complete
        logging.debug("The PCA Analysis is complete.")

    # shrink a dataframe to only a certain set of columns defined in the column_dict internal
    # variable.
    def __truncate_dataframe__(self, the_dataframe):
        # log that we've been called
        logging.debug(f"truncating dataframe on [{self.column_dict}]")

        # define variables
        the_result = DataFrame()

        # loop over all the keys in the column_dict
        for key in self.column_dict:
            # get the column for the key
            the_column = self.column_dict[key]

            # log the current column
            logging.debug(f"current column is [{key}][{the_column}]")

            # first check if the column is in the DataFrame
            if the_column in the_dataframe:
                # log that this occurred
                logging.debug(f"the_column[{the_column}] ")

                # add the column to the final result
                the_result[the_column] = the_dataframe[the_column]
            elif the_column + self.Z_SCORE in the_dataframe:

                # add the column to the final result
                the_result[the_column + self.Z_SCORE] = the_dataframe[the_column + self.Z_SCORE]
            else:
                # this has to be bad, raise error
                raise Exception(f"unable to handle key[{key}] column[{the_column}]")

        # return
        return the_result
