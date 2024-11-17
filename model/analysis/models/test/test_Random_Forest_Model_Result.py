import os
import unittest
from os.path import exists

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV

from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.models.ModelResultBase import ModelResultBase
from model.analysis.models.Random_Forest_Model import Random_Forest_Model
from model.analysis.models.Random_Forest_Model_Result import Random_Forest_Model_Result
from model.constants.BasicConstants import D_209_CHURN, MT_RF_REGRESSION, ANALYZE_DATASET_FULL
from model.constants.ModelConstants import X_TRAIN, Y_TRAIN, Y_TEST, X_ORIGINAL, X_TEST, LM_FEATURE_NUM, LM_P_VALUE, \
    LM_PREDICTOR
from model.constants.ReportConstants import MODEL_MEAN_ABSOLUTE_ERROR, MODEL_ROOT_MEAN_SQUARED_ERROR, MODEL_R_SQUARED, \
    MODEL_Y_PRED, MODEL_MEAN_SQUARED_ERROR, NUMBER_OF_OBS, MODEL_BEST_SCORE, MODEL_BEST_PARAMS, R_SQUARED_HEADER
from util.CSV_loader import CSV_Loader


class test_Random_Forest_Model_Result(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../../../resources/Output/"
    VALID_CSV_PATH = "../../resources/Input/churn_raw_data.csv"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    params_dict_str = 'bootstrap: True, max_depth: None, max_features: sqrt, min_samples_leaf: 1, ' \
                      'min_samples_split: 2, n_estimators: 300'

    CHURN_KEY = D_209_CHURN

    # negative test method for __init__
    def test_init_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # run assertions
        self.assertIsNotNone(split_results)
        self.assertIsInstance(split_results, dict)
        self.assertEqual(len(split_results), 5)
        self.assertTrue(X_TRAIN in split_results)
        self.assertTrue(X_TEST in split_results)
        self.assertTrue(Y_TRAIN in split_results)
        self.assertTrue(Y_TEST in split_results)
        self.assertTrue(X_ORIGINAL in split_results)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # verify we handle all None(s)
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=None,
                                       the_target_variable=None,
                                       the_variables_list=None,
                                       the_f_df_train=None,
                                       the_f_df_test=None,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=None,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_target_variable is None or incorrect type.", context.exception.msg)

        # verify we handle when the_model=rfr
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=None,
                                       the_target_variable=the_target_column,
                                       the_variables_list=None,
                                       the_f_df_train=None,
                                       the_f_df_test=None,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=None,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_variables_list is None or incorrect type.", context.exception.msg)

        # verify we handle when the_variables_list=current_features_list
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=None,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=None,
                                       the_f_df_test=None,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=None,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_encoded_df is None or incorrect type.", context.exception.msg)

        # verify we handle when the_encoded_df=encoded_df
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=None,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=None,
                                       the_f_df_test=None,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=encoded_df,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_model is None or incorrect type.", context.exception.msg)

        # verify we handle when the_model=rfr
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=None,
                                       the_f_df_test=None,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=encoded_df,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_f_df_train is None or incorrect type.", context.exception.msg)

        # verify we handle when the_f_df_train=the_f_df_train
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=None,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=encoded_df,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_f_df_test is None or incorrect type.", context.exception.msg)

        # verify we handle when the_f_df_test=the_f_df_test
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=the_f_df_test,
                                       the_t_var_train=None,
                                       the_t_var_test=None,
                                       the_encoded_df=encoded_df,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_t_var_train is None or incorrect type.", context.exception.msg)

        # verify we handle when the_t_var_train=the_t_var_train
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=the_f_df_test,
                                       the_t_var_train=the_t_var_train,
                                       the_t_var_test=None,
                                       the_encoded_df=encoded_df,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_t_var_test is None or incorrect type.", context.exception.msg)

        # verify we handle when he_t_var_test=the_t_var_test
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=the_f_df_test,
                                       the_t_var_train=the_t_var_train,
                                       the_t_var_test=the_t_var_test,
                                       the_encoded_df=encoded_df,
                                       the_p_values=None,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("the_p_values is None or incorrect type.", context.exception.msg)

        # verify we handle when the_p_values=p_values
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=the_f_df_test,
                                       the_t_var_train=the_t_var_train,
                                       the_t_var_test=the_t_var_test,
                                       the_encoded_df=encoded_df,
                                       the_p_values=p_values,
                                       gridsearch=None,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("gridsearch is None or incorrect type.", context.exception.msg)

        # verify we handle when gridsearch=grid_search
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=the_f_df_test,
                                       the_t_var_train=the_t_var_train,
                                       the_t_var_test=the_t_var_test,
                                       the_encoded_df=encoded_df,
                                       the_p_values=p_values,
                                       gridsearch=grid_search,
                                       prepared_data=None,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("prepared_data is None or incorrect type.", context.exception.msg)

        # verify we handle when gridsearch=grid_search
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            Random_Forest_Model_Result(the_model=rfr,
                                       the_target_variable=the_target_column,
                                       the_variables_list=current_features_list,
                                       the_f_df_train=the_f_df_train,
                                       the_f_df_test=the_f_df_test,
                                       the_t_var_train=the_t_var_train,
                                       the_t_var_test=the_t_var_test,
                                       the_encoded_df=encoded_df,
                                       the_p_values=p_values,
                                       gridsearch=grid_search,
                                       prepared_data=the_current_features_df,
                                       cleaned_data=None)

        # validate the error message.
        self.assertEqual("cleaned_data is None or incorrect type.", context.exception.msg)

    # test method for __init__
    def test_init(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # run assertions on parameters of KNN_Model_Result
        self.assertIsNotNone(the_result.model)
        self.assertIsNotNone(the_result.y_pred)
        self.assertIsNotNone(the_result.the_f_df_train)
        self.assertIsNotNone(the_result.the_f_df_test)
        self.assertIsNotNone(the_result.the_t_var_train)
        self.assertIsNotNone(the_result.the_t_var_test)
        self.assertIsNotNone(the_result.the_df)
        self.assertIsNotNone(the_result.assumptions)
        self.assertIsNotNone(the_result.the_p_values)
        self.assertIsNotNone(the_result.grid_search)
        self.assertIsNotNone(the_result.prepared_data)
        self.assertIsNotNone(the_result.cleaned_data)

        # validate the data type
        self.assertIsInstance(the_result.the_f_df_train, DataFrame)
        self.assertIsInstance(the_result.the_f_df_test, DataFrame)
        self.assertIsInstance(the_result.the_t_var_train, Series)
        self.assertIsInstance(the_result.the_t_var_test, Series)
        self.assertIsInstance(the_result.the_p_values, DataFrame)
        self.assertIsInstance(the_result.grid_search, GridSearchCV)
        self.assertIsInstance(the_result.prepared_data, DataFrame)
        self.assertIsInstance(the_result.cleaned_data, DataFrame)

    # test get_the_p_values()
    def test_get_the_p_values(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # invoke the method
        p_values = the_result.get_the_p_values()

        # run assertions
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, dict)
        self.assertEqual(len(p_values), 9)

        # validate the actual p_value for each feature.
        self.assertEqual(p_values["Tenure"], 0.0)
        self.assertEqual(p_values["Churn"], 0.0)
        self.assertEqual(p_values["InternetService_DSL"], 0.0)
        self.assertEqual(p_values["MonthlyCharge"], 0.0)
        self.assertEqual(p_values["InternetService_Fiber Optic"], 0.0)
        self.assertEqual(p_values["StreamingTV"], 0.0)
        self.assertEqual(p_values["StreamingMovies"], 0.000006)
        self.assertEqual(p_values["OnlineBackup"], 0.000015)
        self.assertEqual(p_values["InternetService_No response"], 0.000034)

    # test method for get_model()
    def test_get_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_result.get_model())
        self.assertIsInstance(the_result.get_model(), RandomForestRegressor)

    # test method for get_results_dataframe()
    def test_get_results_dataframe(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # get the results_df
        results_df = the_result.get_results_dataframe()

        # run assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertEqual(len(results_df), 9)

        # validate the columns in the results_df
        self.assertEqual(list(results_df.columns), [LM_FEATURE_NUM, LM_P_VALUE])
        self.assertEqual(results_df.index.name, LM_PREDICTOR)

        # validate that status of get_feature_columns()
        self.assertIsNotNone(the_result.get_feature_columns())
        self.assertIsInstance(the_result.get_feature_columns(), list)

        # capture the feature columns list
        feature_column_list = the_result.get_feature_columns()

        # run assertions on feature_column_list
        self.assertEqual(len(feature_column_list), 2)
        self.assertTrue(LM_FEATURE_NUM in feature_column_list)
        self.assertTrue(LM_P_VALUE in feature_column_list)

    # test method for get_model_best_score()
    def test_get_model_best_score(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        print(f"the_result.get_model_best_score()-->{the_result.get_model_best_score()}")

    # test method for get_best_params()
    def test_get_best_params(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # get the params
        the_params = the_result.get_best_params()

        # run assertions on params
        self.assertIsInstance(the_params, str)
        self.assertEqual(the_params, self.params_dict_str)

    # test method for get_mean_absolute_error()
    def test_get_mean_absolute_error(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # run assertions on params
        self.assertEqual(the_result.get_mean_absolute_error(), 111.0500439299238)

    def test_get_mean_squared_error(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # run final assertion on method
        self.assertEqual(the_result.get_mean_squared_error(), 20099.984180287032)

    # test method for get_root_mean_squared_error()
    def test_get_root_mean_squared_error(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # run final assertions
        self.assertEqual(the_result.get_root_mean_squared_error(), 141.77441299574136)

    # test method for get_r_squared()
    def test_get_r_squared(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # run final assertions
        self.assertEqual(the_result.get_r_squared(), 0.9958042668098587)

    # test method for populate_assumptions()
    def test_populate_assumptions(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # invoke the method.  populate_assumptions() is called in ModelResultBase.init()
        the_result.populate_assumptions()

        # run assertions on assumptions.
        self.assertIsNotNone(the_result.get_assumptions())
        self.assertIsNotNone(the_result.get_assumptions()[MODEL_MEAN_ABSOLUTE_ERROR])
        self.assertIsNotNone(the_result.get_assumptions()[MODEL_MEAN_SQUARED_ERROR])
        self.assertIsNotNone(the_result.get_assumptions()[MODEL_ROOT_MEAN_SQUARED_ERROR])
        self.assertIsNotNone(the_result.get_assumptions()[R_SQUARED_HEADER])
        self.assertIsNotNone(the_result.get_assumptions()[MODEL_Y_PRED])
        self.assertIsNotNone(the_result.get_assumptions()[NUMBER_OF_OBS])
        self.assertIsNotNone(the_result.get_assumptions()[MODEL_BEST_SCORE])
        self.assertIsNotNone(the_result.get_assumptions()[MODEL_BEST_PARAMS])

        # test actual values of assumptions
        the_assumptions = the_result.get_assumptions()

        self.assertEqual(the_assumptions[MODEL_MEAN_ABSOLUTE_ERROR], 'get_mean_absolute_error')
        self.assertEqual(the_assumptions[MODEL_MEAN_SQUARED_ERROR], 'get_mean_squared_error')
        self.assertEqual(the_assumptions[MODEL_ROOT_MEAN_SQUARED_ERROR], 'get_root_mean_squared_error')
        self.assertEqual(the_assumptions[R_SQUARED_HEADER], 'get_r_squared')
        self.assertEqual(the_assumptions[MODEL_Y_PRED], 'get_y_predicted')
        self.assertEqual(the_assumptions[NUMBER_OF_OBS], 'get_number_of_obs')
        self.assertEqual(the_assumptions[MODEL_BEST_SCORE], 'get_model_best_score')
        self.assertEqual(the_assumptions[MODEL_BEST_PARAMS], 'get_best_params')

    # negative test method for generate_model_csv_files()
    def test_generate_model_csv_files_negative(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # create CSV_Loader
        csv_loader = CSV_Loader()

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            the_result.generate_model_csv_files(csv_loader=None)

        # validate the error message.
        self.assertEqual("csv_loader is None or incorrect type.", context.exception.msg)

    # test method for generate_model_csv_files()
    def test_generate_model_csv_files(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_REGRESSION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = Random_Forest_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, Random_Forest_Model)
        self.assertEqual(the_model.model_type, MT_RF_REGRESSION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)

        # get a list of the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features_list = the_model.get_encoded_variables(the_target_column='Bandwidth_GB_Year')

        # get the actual encoded_df
        encoded_df = the_model.encoded_df[the_encoded_features_list]

        # run assertions
        self.assertIsInstance(encoded_df, DataFrame)

        # define the target_column
        the_target_column = 'Bandwidth_GB_Year'

        # run assertion to verify that the target variable is not present
        self.assertFalse(the_target_column in the_encoded_features_list)
        self.assertFalse(the_target_column in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df[the_target_column]

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, the_target_column)

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order in the encoded_df that is expected
        self.assertEqual(encoded_df.iloc[:, 0].name, 'Population')
        self.assertEqual(encoded_df.iloc[:, 1].name, 'TimeZone')
        self.assertEqual(encoded_df.iloc[:, 2].name, 'Children')
        self.assertEqual(encoded_df.iloc[:, 3].name, 'Age')
        self.assertEqual(encoded_df.iloc[:, 4].name, 'Income')
        self.assertEqual(encoded_df.iloc[:, 5].name, 'Churn')
        self.assertEqual(encoded_df.iloc[:, 6].name, 'Outage_sec_perweek')
        self.assertEqual(encoded_df.iloc[:, 7].name, 'Email')
        self.assertEqual(encoded_df.iloc[:, 8].name, 'Contacts')
        self.assertEqual(encoded_df.iloc[:, 9].name, 'Yearly_equip_failure')
        self.assertEqual(encoded_df.iloc[:, 10].name, 'Techie')
        self.assertEqual(encoded_df.iloc[:, 11].name, 'Port_modem')
        self.assertEqual(encoded_df.iloc[:, 12].name, 'Tablet')
        self.assertEqual(encoded_df.iloc[:, 13].name, 'Phone')
        self.assertEqual(encoded_df.iloc[:, 14].name, 'Multiple')
        self.assertEqual(encoded_df.iloc[:, 15].name, 'OnlineSecurity')
        self.assertEqual(encoded_df.iloc[:, 16].name, 'OnlineBackup')
        self.assertEqual(encoded_df.iloc[:, 17].name, 'DeviceProtection')
        self.assertEqual(encoded_df.iloc[:, 18].name, 'TechSupport')
        self.assertEqual(encoded_df.iloc[:, 19].name, 'StreamingTV')
        self.assertEqual(encoded_df.iloc[:, 20].name, 'StreamingMovies')
        self.assertEqual(encoded_df.iloc[:, 21].name, 'PaperlessBilling')
        self.assertEqual(encoded_df.iloc[:, 22].name, 'Tenure')
        self.assertEqual(encoded_df.iloc[:, 23].name, 'MonthlyCharge')
        self.assertEqual(encoded_df.iloc[:, 24].name, 'Timely_Response')
        self.assertEqual(encoded_df.iloc[:, 25].name, 'Timely_Fixes')
        self.assertEqual(encoded_df.iloc[:, 26].name, 'Timely_Replacements')
        self.assertEqual(encoded_df.iloc[:, 27].name, 'Reliability')
        self.assertEqual(encoded_df.iloc[:, 28].name, 'Options')
        self.assertEqual(encoded_df.iloc[:, 29].name, 'Respectful_Response')
        self.assertEqual(encoded_df.iloc[:, 30].name, 'Courteous_Exchange')
        self.assertEqual(encoded_df.iloc[:, 31].name, 'Active_Listening')
        self.assertEqual(encoded_df.iloc[:, 32].name, 'Area_Rural')
        self.assertEqual(encoded_df.iloc[:, 33].name, 'Area_Suburban')
        self.assertEqual(encoded_df.iloc[:, 34].name, 'Area_Urban')
        self.assertEqual(encoded_df.iloc[:, 35].name, 'Marital_Divorced')
        self.assertEqual(encoded_df.iloc[:, 36].name, 'Marital_Married')
        self.assertEqual(encoded_df.iloc[:, 37].name, 'Marital_Never Married')
        self.assertEqual(encoded_df.iloc[:, 38].name, 'Marital_Separated')
        self.assertEqual(encoded_df.iloc[:, 39].name, 'Marital_Widowed')
        self.assertEqual(encoded_df.iloc[:, 40].name, 'Gender_Female')
        self.assertEqual(encoded_df.iloc[:, 41].name, 'Gender_Male')
        self.assertEqual(encoded_df.iloc[:, 42].name, 'Gender_Nonbinary')
        self.assertEqual(encoded_df.iloc[:, 43].name, 'Contract_Month-to-month')
        self.assertEqual(encoded_df.iloc[:, 44].name, 'Contract_One year')
        self.assertEqual(encoded_df.iloc[:, 45].name, 'Contract_Two Year')
        self.assertEqual(encoded_df.iloc[:, 46].name, 'InternetService_DSL')
        self.assertEqual(encoded_df.iloc[:, 47].name, 'InternetService_Fiber Optic')
        self.assertEqual(encoded_df.iloc[:, 48].name, 'InternetService_No response')
        self.assertEqual(encoded_df.iloc[:, 49].name, 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(encoded_df.iloc[:, 50].name, 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(encoded_df.iloc[:, 51].name, 'PaymentMethod_Electronic Check')
        self.assertEqual(encoded_df.iloc[:, 52].name, 'PaymentMethod_Mailed Check')

        # eliminate all features with a p_value > 0.001
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # define the list of current_features
        current_features_list = the_current_features_df.columns.to_list()

        # split and scale
        split_results = the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                                           the_target=the_target_df,
                                                           test_size=0.2)

        # define variables
        the_f_df_train = split_results[X_TRAIN]
        the_f_df_test = split_results[X_TEST]
        the_t_var_train = split_results[Y_TRAIN]
        the_t_var_test = split_results[Y_TEST]
        the_f_df_test_orig = split_results[X_ORIGINAL]

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 7665)
        self.assertEqual(len(the_f_df_train.columns), 9)
        self.assertEqual(the_f_df_train.shape, (7665, 9))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 1917)
        self.assertEqual(len(the_f_df_test.columns), 9)
        self.assertEqual(the_f_df_test.shape, (1917, 9))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 7665)
        self.assertEqual(the_t_var_train.shape, (7665,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 1917)
        self.assertEqual(the_t_var_test.shape, (1917,))

        # run assertions on the_f_df_test_orig
        self.assertIsNotNone(the_f_df_test_orig)
        self.assertIsInstance(the_f_df_test_orig, DataFrame)
        self.assertEqual(len(the_f_df_test_orig), 1917)
        self.assertEqual(the_f_df_test_orig.shape, (1917, 9))

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit.  You have to call this first.
        rfr.fit(the_f_df_train, the_t_var_train)

        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [200, 300, 1000], 'max_features': ['sqrt'],
                      'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 5], 'bootstrap': [True]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

        # invoke the method
        the_result = Random_Forest_Model_Result(the_model=rfr,
                                                the_target_variable=the_target_column,
                                                the_variables_list=current_features_list,
                                                the_f_df_train=the_f_df_train,
                                                the_f_df_test=the_f_df_test,
                                                the_t_var_train=the_t_var_train,
                                                the_t_var_test=the_t_var_test,
                                                the_encoded_df=the_f_df_test_orig,
                                                the_p_values=p_values,
                                                gridsearch=grid_search,
                                                prepared_data=the_current_features_df,
                                                cleaned_data=the_model.encoded_df)

        # run assertions on the result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsInstance(the_result, ModelResultBase)

        # create CSV_Loader
        csv_loader = CSV_Loader()

        # check if the previous file is there, and if so, delete it.
        if exists("../../../../resources/Output/churn_cleaned.csv"):
            os.remove("../../../../resources/Output/churn_cleaned.csv")

        if exists("../../../../resources/Output/churn_X_train.csv"):
            os.remove("../../../../resources/Output/churn_X_train.csv")

        if exists("../../../../resources/Output/churn_X_test.csv"):
            os.remove("../../../../resources/Output/churn_X_test.csv")

        if exists("../../../../resources/Output/churn_Y_train.csv"):
            os.remove("../../../../resources/Output/churn_Y_train.csv")

        if exists("../../../../resources/Output/churn_Y_test.csv"):
            os.remove("../../../../resources/Output/churn_Y_test.csv")

        # invoke the method
        the_result.generate_model_csv_files(csv_loader=pa.csv_l)

        # run assertions
        self.assertTrue(exists("../../../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_X_train.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_X_test.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_Y_train.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_Y_test.csv"))
