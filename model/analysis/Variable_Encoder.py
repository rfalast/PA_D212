import logging
import pandas as pd

from pandas import DataFrame
from pandas.core.dtypes.common import is_object_dtype
from model.BaseModel import BaseModel


# Linear Model tools
class Variable_Encoder(BaseModel):

    # init method
    def __init__(self, unencoded_df: DataFrame):
        super().__init__()

        # perform validation
        if not self.is_valid(unencoded_df, DataFrame):
            raise ValueError("unencoded_df is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # log that we've been instantiated
        self.logger.debug("An instance of Variable_Encoder has been created.")

        # define internal variables
        self.original_df = unencoded_df
        self.encoded_df = None
        self.storage = {}

    # encode dataframe
    def encode_dataframe(self, drop_first=True):
        # log that we've been called
        self.logger.debug(f"A request to encode variables for dataframe with has been made.")

        # retrieve all list of columns
        list_of_df_columns = self.original_df.columns

        # create final list
        object_column_list = []

        # loop through the list
        for the_column in list_of_df_columns:
            # check if the type is object
            if is_object_dtype(self.original_df[the_column]):
                # add the column to object_column_list
                object_column_list.append(the_column)

                # log that we're adding the column
                self.logger.debug(f"Adding [{the_column}] to {object_column_list}")

        # encode the object variables.  The argument drop_first must be set to true to reduce
        # multi-collinearity risks if the model is susceptible.
        if len(object_column_list) > 0:
            self.encoded_df = pd.get_dummies(self.original_df, columns=object_column_list, drop_first=drop_first)

    # get the encoded dataframe
    def get_encoded_dataframe(self, drop_first=True) -> DataFrame:
        # log that we've been called.
        self.logger.debug(f"getting encoded dataframe with drop_first set to [{drop_first}].")

        # check if encoded_df is forced to None somehow
        if self.encoded_df is None:
            # call encode_dataframe()
            self.encode_dataframe(drop_first=drop_first)

        # return
        return self.encoded_df
