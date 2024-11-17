import os

from model.constants.BasicConstants import ANALYZE_DATASET_INITIAL, ANALYZE_DATASET_FULL, D_209_CHURN, D_209_MEDICAL, \
    MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION, MT_KNN_CLASSIFICATION, MT_RF_REGRESSION
from model.Project_Assessment import Project_Assessment

LOG_FILE_LOCATION = "Output.log"

# main method
if __name__ == '__main__':
    # get current path
    curr_path = os.path.dirname(os.path.abspath(__file__)) + "/"

    # create Project_Assessment
    pa = Project_Assessment(base_directory=curr_path)

    # write the base directory to the console.
    print(f"base directory is set to [{pa.base_directory}].")

    # a dictionary of variables on the dataframe that we will rename.
    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    # columns that we will drop because they either have too much dimensionality or are irrelevant
    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    # the level that we consider relevant for the correlations testing
    the_level = 0.5
    the_target_variable = 'Bandwidth_GB_Year'
    the_p_value = 0.001
    the_max_p_value = 0.01
    the_max_vif = 7.0
    the_model_type = MT_RF_REGRESSION

    # invoke steps of use case
    pa.load_dataset(dataset_name_key=D_209_CHURN)                   # can be used interchangeably with either data set
    pa.analyze_dataset(ANALYZE_DATASET_INITIAL)                     # initial dataset analysis
    pa.generate_initial_report()                                    # output of dirty data
    pa.analyze_dataset(ANALYZE_DATASET_FULL)                        # full dataset analysis
    pa.update_column_names(field_rename_dict)                       # rename fields
    pa.drop_column_from_dataset(column_drop_list)                   # drop spurious columns from dataset
    pa.analyze_dataset(ANALYZE_DATASET_FULL)                        # perform full dataset analysis
    pa.calculate_internal_statistics(the_level=the_level)           # calculate internal statistics
    pa.clean_up_outliers(model_type=the_model_type,
                         max_p_value=the_max_p_value)               # remove outliers
    pa.analyze_dataset(ANALYZE_DATASET_FULL)                        # full dataset analysis
    pa.calculate_internal_statistics(the_level=the_level)           # calculate internal statistics

    pa.build_model(the_target_variable,
                   model_type=the_model_type,
                   max_p_value=the_p_value,
                   the_max_vif=the_max_vif,
                   suppress_console=False)                          # build a model again, without outliers
    pa.generate_output_report(the_model_type=the_model_type)        # output of clean data
