import logging
import pandas as pd

from pathlib import Path
from os.path import exists
from pandas import DataFrame
from model.constants.BasicConstants import OUTPUT_OPTIONS, D_212_CHURN, CHURN_FINAL
from util.BaseUtil import BaseUtil


# A class to read a csv file, and do various things with it.
class CSV_Loader(BaseUtil):

    # init method
    def __init__(self, base_directory=None):
        # instantiate the super class.
        super().__init__()

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # run internal validation
        if base_directory is not None:
            if not exists(base_directory):
                raise NotADirectoryError("base_directory does not exist.")

        # define internal variables
        self.base_directory = base_directory

    # method for instantiating a dataframe from csv
    # the instance returned is not persistent
    def get_data_frame_from_csv(self, csv_path):
        # log that we've been called
        logging.debug("A request to get_data_frame_from_csv() has been made.")

        # variable definition
        the_result = None

        # validate argument is correct type
        if not self.is_valid(csv_path, str):
            raise TypeError("csv_path was None or not a string.")
        # validate that that path is valid
        elif not exists(csv_path):
            print(f"the path is {csv_path}")
            raise FileNotFoundError("The file references from csv_path does not exist.")
        else:
            # log what we're doing
            logging.debug(f"Retrieving file from [{csv_path}]")

            # get a dataframe reference
            the_result = pd.read_csv(Path(csv_path))

            # log what we're returning
            logging.debug("Retrieved df-->\n" + str(the_result))

        # return statement
        return the_result

    # generate output file path
    def generate_output_file_path(self, data_set=None, option=None) -> Path:
        # run validations
        if data_set not in list(OUTPUT_OPTIONS.keys()):
            raise ValueError("data_set is None or incorrect type.")
        elif option not in list(OUTPUT_OPTIONS[data_set].keys()):
            raise ValueError("option is None or incorrect type.")

        # generate the path to the CSV file.
        the_path = Path(self.base_directory + OUTPUT_OPTIONS[data_set][option])

        # log the path
        self.logger.debug(f"Path is {the_path}")

        # return
        return the_path

    # generate output file
    def generate_output_file(self, data_set=None, option=None, the_dataframe=None):
        # run validations
        if data_set not in list(OUTPUT_OPTIONS.keys()):
            raise ValueError("data_set is None or incorrect type.")
        elif option not in list(OUTPUT_OPTIONS[data_set].keys()):
            raise ValueError("option is None or incorrect type.")
        elif not isinstance(the_dataframe, DataFrame):
            raise SyntaxError("the_dataframe is None or incorrect type.")

        # variable declaration
        the_path = None

        # generate log message
        self.logger.debug(f"A request to generate an output file with data_set[{data_set}], option[{option}].")

        # get the path
        the_path = self.generate_output_file_path(data_set=data_set, option=option)

        # spit out the clean CSV file
        the_dataframe.to_csv(the_path, index=False)
