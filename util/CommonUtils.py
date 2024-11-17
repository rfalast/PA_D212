import collections
import logging

from pandas import DataFrame, Series

from model.constants.BasicConstants import COLON, COMMA, BLANK_SPACE
from util.BaseUtil import BaseUtil
from collections.abc import Hashable


# class holding common utility functions.
class CommonUtils(BaseUtil):
    # constants

    # init method
    def __init__(self):
        # call super class
        super().__init__()

        # log that we were initialized
        logging.debug("Initialized instance of CommonUtils")

    # check if a string is "nan"
    @staticmethod
    def isnan(the_value):
        try:
            import math
            return math.isnan(float(the_value))
        except:
            return False


# take a string with an integer at the end, and remove the integer
# the return is just the value of the integer.  If you pass only
# the string representation of an integer, you'll get back an
# integer value.
def strip_trailing_digit_from_field(the_field):
    # declare the result
    the_result = None

    # make sure we got an argument
    if the_field is None or not isinstance(the_field, str) or len(the_field) == 0:
        raise SyntaxError("Argument is None or empty.")

    # detect if the argument is exclusively a string of digits only
    elif the_field.isdigit():
        the_result = int(the_field)

    # looks like we have a valid field
    else:
        # declarations
        i = 1

        # reverse the string
        test_string = the_field[::-1]

        # loop over the characters in the test string
        while i <= len(test_string):

            # check if the characters retrieved constitute a number
            if test_string[0:i].strip().isnumeric():

                # increment i
                i = i + 1

            # we found our length, or all the characters are non-numeric
            else:
                # remove one digit
                i = i - 1

                if the_field[len(the_field) - i:].strip().isnumeric():
                    # get the result
                    the_result = int(the_field[len(the_field) - i:].strip())
                else:
                    # log a warning that we can't handle the string
                    logging.info("The incoming argument [%s] has no trailing digits to strip", the_field)

                # break
                break

    # return the result
    return the_result


# Take a string with the integer at the end, and return the integer.
# Ex- If you have a string like AB201 or C5, the returned value would be
# '201' in the first case, and '5' in the second.
def strip_preceding_letter_from_field(the_field: str):
    # declare the result
    the_result = None

    # make sure we got an argument
    if the_field is None or not isinstance(the_field, str) or len(the_field) == 0:
        raise SyntaxError("Argument is None or empty.")

    # detect if the argument is exclusively a string of alphabetic characters only
    elif the_field.isalpha():
        the_result = the_field

    # looks like we have a valid field
    else:
        # declarations
        i = 1

        # loop over the characters in the test string
        for element in range(0, len(the_field)):
            # check if the next character is alphabetic
            if the_field[element].isalpha():
                # increment i
                i = i + 1
            elif the_field[element].isnumeric():
                break

        # take the characters from the left
        if len(the_field) > i and the_field[i] == " ":
            the_result = the_field[:i]
        else:
            the_result = the_field[:i - 1]

    # return the result
    return the_result


# determine if tuples are the same
def are_tuples_the_same(tuple_1: tuple, tuple_2: tuple) -> bool:
    # run validations
    if not isinstance(tuple_1, tuple):
        raise AttributeError("tuple_1 is None or incorrect type.")
    elif not isinstance(tuple_2, tuple):
        raise AttributeError("tuple_2 is None or incorrect type.")

    # variable declaration
    the_result = False
    i = 0

    # check if length is the same
    if len(tuple_1) == len(tuple_2):
        # convert the tuples into lists
        tuple_1_list = list(tuple_1)
        tuple_2_list = list(tuple_2)

        # if an element of the tuple is a contingency table, we need to exclude it from the list
        tuple_1_list = remove_item_from_list_by_type(tuple_1_list, DataFrame)
        tuple_2_list = remove_item_from_list_by_type(tuple_2_list, DataFrame)

        # make sure the lists are hashable
        if are_list_elements_hashable(tuple_1_list) and are_list_elements_hashable(tuple_2_list):

            # check if elements are equal, regardless of order
            if collections.Counter(tuple_1_list) == collections.Counter(tuple_2_list):
                the_result = True

    # return result
    return the_result


# determine if elements in a list are hashable
def are_list_elements_hashable(the_list: list) -> bool:
    # run validations
    if not isinstance(the_list, list):
        raise AttributeError("the_list is None or incorrect type.")

    # variable declaration
    the_result = False

    # make sure we have a list
    if isinstance(the_list, list):
        # check if all the items in the lists are hashable
        for current_item in the_list:
            # set the_result to True
            the_result = True

            # check if it is hashable
            if not isinstance(current_item, Hashable):
                # set to the_result to False
                the_result = False

                # break out of the list
                break
    # return
    return the_result


# remove item from list
def remove_item_from_list_by_type(the_list: list, the_type) -> list:
    # run validations
    if not isinstance(the_list, list):
        raise AttributeError("the_list is None or incorrect type.")
    elif the_type is None:
        raise AttributeError("the_type is None.")

    # variable declaration
    i = 0

    # loop through the list
    for next_element in the_list:
        if isinstance(next_element, the_type):
            # remove the item by index
            the_list.pop(i)

        # increment index
        i = i + 1

    # return the list
    return the_list


# get tuples with specific field
def get_tuples_from_list_with_specific_field(the_list: list, the_field) -> list:
    # run validations
    if not isinstance(the_list, list):
        raise AttributeError("the_list is None or incorrect type.")
    elif the_field is None:
        raise AttributeError("the_type is None.")
    elif not all(isinstance(n, tuple) for n in the_list):
        raise AttributeError("the_list is not all of type tuple.")

    # variable declaration
    the_result = []
    i = 0

    # loop through the_list
    for element in the_list:
        # reset i
        i = 0

        # loop over all the fields in the tuple
        while i < len(element):
            # check if field is in the tuple
            if element[i] == the_field:
                # append to the_result
                the_result.append(element)

                break
            else:
                # increment i
                i = i + 1

    # return
    return the_result


# remove a tuple from a list if all fields in the tuple are on the exclusion list
def remove_tuples_from_list(original_list: list, exclusion_list: list) -> list:
    # run validations
    if not isinstance(original_list, list):
        raise AttributeError("original_list is None or incorrect type.")
    elif not isinstance(exclusion_list, list):
        raise AttributeError("exclusion_list is None or incorrect type.")
    elif not all(isinstance(n, tuple) for n in original_list):
        raise AttributeError("original_list is not all of type tuple.")

    # variable declaration
    the_result = original_list.copy()  # final result
    remove_flag = True  # we start out assuming True, remove this element from the original_list
    i = 0  # counter for number of elements in the tuple
    tuple_index = 0  # index of current tuple on original_lis

    # loop through original_list
    for element in original_list:
        # reset element counter
        i = 0

        # loop over all the fields in the tuple
        while i < len(element):
            # check if field in the tuple element is in the exclusion list
            if element[i] in exclusion_list:
                # do nothing
                pass
            # if the field is not in the exclusion list, we don't exclude this tuple
            else:
                # set the flag to false
                remove_flag = False

            # increment i
            i = i + 1

        # check if we remove the element from the original list
        if remove_flag:
            the_result.remove(element)
        # we don't remove element from the_result
        else:
            # flip it back to true
            remove_flag = True

    # return
    return the_result


# convert a dict to a str.  Only keys of type str will be used.
def convert_dict_to_str(the_dict: dict) -> str:
    # run validations
    if not isinstance(the_dict, dict):
        raise AttributeError("the_dict is None or incorrect type.")

    # variable declaration
    the_result = ''

    # iterate over the keys in the dict
    for the_key in the_dict.keys():
        # check if the key is a str
        if isinstance(the_key, str):
            # now append the value to the_result
            the_result = the_result + the_key + COLON + BLANK_SPACE + str(the_dict[the_key]) + COMMA + BLANK_SPACE

    # validate that we have a non-zero length string result
    if len(the_result) > 0:
        # we need to remove the last two characters, since there will be a superfluous ", ".
        the_result = the_result.rstrip(COMMA + BLANK_SPACE)

    # return
    return the_result


# convert a series to a dataframe.  If the_series is already a DataFrame, it will return the same object.
def convert_series_to_dataframe(the_series: Series) -> DataFrame:
    # run assertions
    if not isinstance(the_series, Series):
        # make sure it's not already a DataFrame
        if isinstance(the_series, DataFrame):
            return the_series
        else:
            raise AttributeError("the_series is None or incorrect type.")

    return the_series.to_frame()
