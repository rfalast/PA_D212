# constants
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_FINAL_MODEL

COLUMN_KEY = "COLUMN_KEY"  # a dict of column name and type
INT64_COLUMN_KEY = "INT64_COLUMN_KEY"  # list of INT column names
FLOAT64_COLUMN_KEY = "FLOAT64_COLUMN_KEY"  # list of FLOAT column names
OBJECT_COLUMN_KEY = "OBJECT_COLUMN_KEY"  # list of OBJECT column names
BOOL_COLUMN_KEY = "BOOL_COLUMN_KEY"  # list of BOOLEAN column names
BOOL_VALUE_KEY = "BOOL_VALUE_KEY"  # dict of BOOLEAN. Key is column, value is list of un-cleaned True/False
DATETIME_COLUMN_KEY = "DATETIME_COLUMN_KEY"  # list of DATETIME column names
BOOL_COLUMN_COUNT_KEY = "BOOL_COLUMN_COUNT_KEY"  # a dictionary of column names and a dict of True / False counts
COLUMN_COUNT_KEY = "COLUMN_COUNT_KEY"  # dict of column name, count. Where count is presentation layer orig col #.
COLUMN_TOTAL_COUNT_KEY = "COLUMN_TOTAL_COUNT"  # dict of column name, total count in column.
UNIQUE_COLUMN_VALUES = "UNIQUE_COLUMN_VALUES"  # dict of column name, list of unique values for column name
UNIQUE_COLUMN_RATIOS = "UNIQUE_COLUMN_RATIOS"  # dict of column name, and float value of column ratio calculation
UNIQUE_COLUMN_FLAG = "UNIQUE_COLUMN_FLAG"  # dict of column name, Y/N string if the column is unique
UNIQUE_COLUMN_LIST_KEY = "UNIQUE_COLUMN_LIST_KEY"  # list of unique columns
COLUMN_NA_COUNT = "COLUMN_NA_COUNT"  # dict of column name, int representing number of NA rows present
BOOL_CLEANED_FLAG = "BOOL_CLEANED_FLAG"  # state based flag to designate the booleans have been cleaned.
INT64_NAN_FLAG = "INT64_NAN_FLAG"  # state based flag to designate there are INT64 with Nan
OBJECT_COLUMN_COUNT_KEY = "OBJECT_COLUMN_COUNT_KEY"  # a dict of column names, and a dict of answer then count
FLOAT64_BIN_SIZES = "FLOAT64_BIN_SIZES"  # dict of column name, bin sizes as int.
DATASET_TYPES = "DATASET_TYPES"  # list of strings representing each of the dataset keys that are present.
OBJECT_NAN_REPL_KEY = "No Response"

# LIST CONSTANTS
YES_NO_LIST = ["Yes", "No"]
DATA_TYPE_KEYS = [INT64_COLUMN_KEY, BOOL_COLUMN_KEY, FLOAT64_COLUMN_KEY, OBJECT_COLUMN_KEY]
LM_MODEL_TYPES = [LM_INITIAL_MODEL, LM_FINAL_MODEL]