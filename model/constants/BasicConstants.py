# project wide constants
import math

TIMEZONE_DICTIONARY = {'America/Chicago': -6, 'America/Los_Angeles': -8, 'America/Denver': -7,
                       'America/Detroit': -6, 'America/Indiana/Indianapolis': -5, 'America/Phoenix': -7,
                       'America/Boise': -7, 'America/Anchorage': -9, 'America/Puerto_Rico': -4,
                       'Pacific/Honolulu': -10, 'America/Menominee': -6, 'America/Nome': -9,
                       'America/Kentucky/Louisville': -5, 'America/Sitka': -9,
                       'America/Indiana/Vincennes': -5, 'America/Indiana/Tell_City': -6,
                       'America/Toronto': -5, 'America/Indiana/Petersburg': -5, 'America/Juneau': -9,
                       'America/North_Dakota/New_Salem': -6, 'America/Indiana/Knox': -6,
                       'America/Indiana/Winamac': -5, 'America/Indiana/Marengo': -5, 'America/Ojinaga': -6,
                       'America/New_York': -5}

EDUCATION_DICTIONARY = {'Regular High School Diploma': 12,
                        'Bachelor\'s Degree': 16,
                        'Some College, 1 or More Years, No Degree': 13,
                        '9th Grade to 12th Grade, No Diploma': 10,
                        'Master\'s Degree': 18,
                        'Associate\'s Degree': 14,
                        'Some College, Less than 1 Year': 13,
                        'Nursery School to 8th Grade': 8,
                        'GED or Alternative Credential': 12,
                        'Professional School Degree': 17,
                        'No Schooling Completed': 0,
                        'Doctorate Degree': 20}

NAN_SWAP_DICT = {math.nan: 'No response'}

TIMEZONE_COLUMN = "TimeZone"

Z_SCORE = "_z_score"

BOOL_SWAP_DICTIONARY = {'Y': True, 'YES': True, 'N': False, 'NO': False, 'True': True, 'False': False,
                        'Yes': True, 'No': False, '1': True, '0': False, 'y': True, 'n': False}

LOG_FILE_LOCATION = "resources/Output/Output.log"
CHURN_CSV_FILE_LOCATION = "resources/Output/churn_cleaned.csv"
CHURN_PREP_CSV_FILE_LOCATION = "resources/Output/churn_prepared.csv"
CHURN_X_TRAIN_CSV_FILE_LOCATION = "resources/Output/churn_X_train.csv"
CHURN_X_TEST_CSV_FILE_LOCATION = "resources/Output/churn_X_test.csv"
CHURN_Y_TRAIN_CSV_FILE_LOCATION = "resources/Output/churn_Y_train.csv"
CHURN_Y_TEST_CSV_FILE_LOCATION = "resources/Output/churn_Y_test.csv"
MEDICAL_CSV_FILE_LOCATION = "resources/Output/medical_cleaned.csv"
MEDICAL_PREP_FILE_LOCATION = "resources/Output/medical_prepared.csv"

UNNAMED_COLUMN = "Unnamed: 0"
DEFAULT_INDEX_NAME = "INDEX_COL"
ANALYZE_DATASET_INITIAL = "INITIAL"
ANALYZE_DATASET_FULL = "FULL"

ANALYSIS_TYPE = [ANALYZE_DATASET_FULL, ANALYZE_DATASET_INITIAL]

D_209_CHURN = "D_209_CHURN"
CHURN_PREP = "CHURN_PREP"
CHURN_FINAL = "CHURN_FINAL"
CHURN_X_TRAIN = "CHURN_X_TRAIN"
CHURN_X_TEST = "CHURN_X_TEST"
CHURN_Y_TRAIN = "CHURN_Y_TRAIN"
CHURN_Y_TEST = "CHURN_Y_TEST"
D_209_MEDICAL = "D_209_MEDICAL"
MEDICAL_PREP = "MEDICAL_PREP"
MEDICAL_FINAL = "MEDICAL_FINAL"

OUTPUT_OPTIONS = {D_209_CHURN: {CHURN_FINAL: CHURN_CSV_FILE_LOCATION, CHURN_PREP: CHURN_PREP_CSV_FILE_LOCATION,
                                CHURN_X_TRAIN: CHURN_X_TRAIN_CSV_FILE_LOCATION,
                                CHURN_X_TEST: CHURN_X_TEST_CSV_FILE_LOCATION,
                                CHURN_Y_TRAIN: CHURN_Y_TRAIN_CSV_FILE_LOCATION,
                                CHURN_Y_TEST: CHURN_Y_TEST_CSV_FILE_LOCATION},
                  D_209_MEDICAL: {MEDICAL_FINAL: MEDICAL_CSV_FILE_LOCATION, MEDICAL_PREP: MEDICAL_PREP_FILE_LOCATION}}

MT_LOGISTIC_REGRESSION = "MT_LOGISTIC_REGRESSION"
MT_LINEAR_REGRESSION = "MT_LINEAR_REGRESSION"
MT_KNN_CLASSIFICATION = "MT_KNN_CLASSIFICATION"
MT_DT_CLASSIFICATION = "MT_DT_CLASSIFICATION"
MT_RF_CLASSIFICATION = "MT_RF_CLASSIFICATION"
MT_RF_REGRESSION = "MT_RF_REGRESSION"
MT_OPTIONS = [MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION, MT_KNN_CLASSIFICATION, MT_DT_CLASSIFICATION,
              MT_RF_CLASSIFICATION, MT_RF_REGRESSION]

COLON = ":"
BLANK_SPACE = " "
COMMA = ","


