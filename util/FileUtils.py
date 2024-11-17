import logging
import os
import platform
import re
import shutil

from os import makedirs
from os.path import exists
from util.BaseUtil import BaseUtil


# class for holding utilities for dealing with files and directories
class FileUtils(BaseUtil):
    # constants
    OS_OPTIONS = ['macOS', 'Windows', 'Darwin']
    DIRECTORY_SEPERATOR = {"macOS": "/", "Windows": "\\", "Darwin": "/"}

    # init method
    def __init__(self):
        super().__init__()

        # define internal variables
        self.os_type = None

        # log that we've been created
        logging.debug("Creating an instance of FileUtils.")

    # create output directories
    def create_output_directories(self, the_path):
        # run validations
        if not self.is_valid(the_path, str):
            raise SyntaxError("The path argument was None or incorrect type.")

        # log that we've been called
        logging.debug(f"A request has been made to create the output directory [{the_path}].")

        # check if the directory exists
        if exists(the_path):
            # log that the path exists
            logging.debug(f"The path [{the_path}] was found.")
        # directory does not exist
        else:
            # create it.
            makedirs(the_path)

    # remove output directory
    def remove_output_directory(self, the_path):
        # run validations
        if not self.is_valid(the_path, str):
            raise SyntaxError("The path argument was None or incorrect type.")

        # log that we've been called
        logging.debug(f"A request has been made to remove the output directory [{the_path}].")

        # remove the directory and all it's contents
        shutil.rmtree(the_path)

    # find the OS type
    def find_os_type(self) -> str:
        # log that we've been called
        logging.debug("A request to determine the OS has been made.")

        # capture the OS type
        self.os_type = platform.system()

        # log the platform
        logging.debug(f"found to be running on [{self.os_type}]")

        # return the result
        return self.os_type

    # get the absolute base directory, relative to this file
    @staticmethod
    def get_base_directory() -> str:
        # log that we've been called
        logging.debug("A request to find the base directory has been made.")

        # variable declaration
        the_result = os.path.dirname(os.path.abspath(__file__))

        # log what we found
        logging.debug(f"the relative base directory is [{the_result}]")

        # return the result
        return the_result

    # get the directory seperator for this machine
    def get_directory_seperator(self) -> str:
        # log that we've been called
        logging.debug("A request to get directory seperator has been made.")

        # variable declaration
        the_result = None

        # check if the os_type has been set
        if self.os_type is None:
            # retrieve the os type
            self.os_type = self.find_os_type()

        # get the answer
        if self.os_type in self.DIRECTORY_SEPERATOR:
            # get the result
            the_result = self.DIRECTORY_SEPERATOR[self.os_type]

            # log that we found a match
            logging.debug(f"A match was found for [{self.os_type}]-->[{the_result}]")
        else:
            raise KeyError(f"no match found for [{self.os_type}]")

        # return the result
        return the_result

    # correct path for this OS
    def correct_path_for_this_os(self, the_path: str) -> str:
        # run validations
        if not self.is_valid(the_path, str):
            raise SyntaxError("The path argument was None or incorrect type.")

        # get the directory seperator for this OS
        seperator = self.get_directory_seperator()

        # loop over all the options defined in the DIRECTORY_SEPERATOR dict
        for option in self.DIRECTORY_SEPERATOR:
            # get the current_seperator
            current_seperator = self.DIRECTORY_SEPERATOR[option]

            print(f"current_seperator-->{current_seperator}")

            # check if it doesn't match
            if seperator != current_seperator:
                the_path = re.sub(current_seperator, seperator, the_path)

        # return the cleaned path
        return the_path
