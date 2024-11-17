import numpy as np
import pandas as pd

# Excel specific constants
NA_VALUE_LIST = ['#N/A', '<NA>', 'NA', np.nan, np.NaN, np.NAN, pd.NA]
EMPTY_STRING = ""
SINGLE_SPACE = " "
ILLEGAL_TAB_FIELDS = ['/', '\\']

LETTER_KEY = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10,
              "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19,
              "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}

