import unittest

import numpy
import numpy as np
import pandas as pd
from numpy import float64
from pandas import DataFrame, Series
from scipy.stats import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models.Random_Forest_Model import Random_Forest_Model
from model.analysis.models.Random_Forest_Model_Result import Random_Forest_Model_Result
from model.constants.BasicConstants import D_209_CHURN, ANALYZE_DATASET_FULL, MT_RF_CLASSIFICATION, MT_RF_REGRESSION
from model.constants.DatasetConstants import INT64_COLUMN_KEY, FLOAT64_COLUMN_KEY
from model.constants.ModelConstants import X_TRAIN, Y_TRAIN, Y_TEST, X_TEST, X_ORIGINAL


class test_Random_Forest_Model(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_209/"
    OVERRIDE_PATH = "../../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_209_CHURN

    # proof of concept for a Random Forest
    def test_random_forest_proof_of_concept(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dataset_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(dataset_analyzer)
        self.assertIsInstance(dataset_analyzer, DatasetAnalyzer)

        # initialize class
        the_class = Random_Forest_Model(dataset_analyzer=dataset_analyzer)

        # get a reference to the encoded_df
        encoded_df = the_class.encoded_df

        # run validations on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(the_class.encoded_df, DataFrame)
        self.assertEqual(len(the_class.encoded_df), 9582)
        self.assertEqual(len(the_class.encoded_df.columns), 54)
        self.assertEqual(encoded_df.shape, (9582, 54))

        # get the variable columns
        the_variable_columns = encoded_df.columns.to_list()

        # run assertions on the_variable_columns
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # validate what is in the_variable_columns
        self.assertTrue('Population' in the_variable_columns)
        self.assertTrue('TimeZone' in the_variable_columns)
        self.assertTrue('Children' in the_variable_columns)
        self.assertTrue('Age' in the_variable_columns)
        self.assertTrue('Income' in the_variable_columns)
        self.assertTrue('Churn' in the_variable_columns)
        self.assertTrue('Outage_sec_perweek' in the_variable_columns)
        self.assertTrue('Email' in the_variable_columns)
        self.assertTrue('Contacts' in the_variable_columns)
        self.assertTrue('Yearly_equip_failure' in the_variable_columns)
        self.assertTrue('Techie' in the_variable_columns)
        self.assertTrue('Port_modem' in the_variable_columns)
        self.assertTrue('Tablet' in the_variable_columns)
        self.assertTrue('Phone' in the_variable_columns)
        self.assertTrue('Multiple' in the_variable_columns)
        self.assertTrue('OnlineSecurity' in the_variable_columns)
        self.assertTrue('OnlineBackup' in the_variable_columns)
        self.assertTrue('DeviceProtection' in the_variable_columns)
        self.assertTrue('TechSupport' in the_variable_columns)
        self.assertTrue('StreamingTV' in the_variable_columns)
        self.assertTrue('StreamingMovies' in the_variable_columns)
        self.assertTrue('PaperlessBilling' in the_variable_columns)
        self.assertTrue('Tenure' in the_variable_columns)
        self.assertTrue('MonthlyCharge' in the_variable_columns)
        self.assertTrue('Bandwidth_GB_Year' in the_variable_columns)
        self.assertTrue('Timely_Response' in the_variable_columns)
        self.assertTrue('Timely_Fixes' in the_variable_columns)
        self.assertTrue('Timely_Replacements' in the_variable_columns)
        self.assertTrue('Reliability' in the_variable_columns)
        self.assertTrue('Options' in the_variable_columns)
        self.assertTrue('Respectful_Response' in the_variable_columns)
        self.assertTrue('Courteous_Exchange' in the_variable_columns)
        self.assertTrue('Active_Listening' in the_variable_columns)
        self.assertTrue('Area_Rural' in the_variable_columns)
        self.assertTrue('Area_Suburban' in the_variable_columns)
        self.assertTrue('Area_Urban' in the_variable_columns)
        self.assertTrue('Marital_Divorced' in the_variable_columns)
        self.assertTrue('Marital_Married' in the_variable_columns)
        self.assertTrue('Marital_Never Married' in the_variable_columns)
        self.assertTrue('Marital_Separated' in the_variable_columns)
        self.assertTrue('Marital_Widowed' in the_variable_columns)
        self.assertTrue('Gender_Female' in the_variable_columns)
        self.assertTrue('Gender_Male' in the_variable_columns)
        self.assertTrue('Gender_Nonbinary' in the_variable_columns)
        self.assertTrue('Contract_Month-to-month' in the_variable_columns)
        self.assertTrue('Contract_One year' in the_variable_columns)
        self.assertTrue('Contract_Two Year' in the_variable_columns)
        self.assertTrue('InternetService_DSL' in the_variable_columns)
        self.assertTrue('InternetService_Fiber Optic' in the_variable_columns)
        self.assertTrue('InternetService_No response' in the_variable_columns)
        self.assertTrue('PaymentMethod_Bank Transfer(automatic)' in the_variable_columns)
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in the_variable_columns)
        self.assertTrue('PaymentMethod_Electronic Check' in the_variable_columns)
        self.assertTrue('PaymentMethod_Mailed Check' in the_variable_columns)

        # validate the order is expected
        self.assertEqual(the_variable_columns[0], 'Population')
        self.assertEqual(the_variable_columns[1], 'TimeZone')
        self.assertEqual(the_variable_columns[2], 'Children')
        self.assertEqual(the_variable_columns[3], 'Age')
        self.assertEqual(the_variable_columns[4], 'Income')
        self.assertEqual(the_variable_columns[5], 'Churn')
        self.assertEqual(the_variable_columns[6], 'Outage_sec_perweek')
        self.assertEqual(the_variable_columns[7], 'Email')
        self.assertEqual(the_variable_columns[8], 'Contacts')
        self.assertEqual(the_variable_columns[9], 'Yearly_equip_failure')
        self.assertEqual(the_variable_columns[10], 'Techie')
        self.assertEqual(the_variable_columns[11], 'Port_modem')
        self.assertEqual(the_variable_columns[12], 'Tablet')
        self.assertEqual(the_variable_columns[13], 'Phone')
        self.assertEqual(the_variable_columns[14], 'Multiple')
        self.assertEqual(the_variable_columns[15], 'OnlineSecurity')
        self.assertEqual(the_variable_columns[16], 'OnlineBackup')
        self.assertEqual(the_variable_columns[17], 'DeviceProtection')
        self.assertEqual(the_variable_columns[18], 'TechSupport')
        self.assertEqual(the_variable_columns[19], 'StreamingTV')
        self.assertEqual(the_variable_columns[20], 'StreamingMovies')
        self.assertEqual(the_variable_columns[21], 'PaperlessBilling')
        self.assertEqual(the_variable_columns[22], 'Tenure')
        self.assertEqual(the_variable_columns[23], 'MonthlyCharge')
        self.assertEqual(the_variable_columns[24], 'Bandwidth_GB_Year')
        self.assertEqual(the_variable_columns[25], 'Timely_Response')
        self.assertEqual(the_variable_columns[26], 'Timely_Fixes')
        self.assertEqual(the_variable_columns[27], 'Timely_Replacements')
        self.assertEqual(the_variable_columns[28], 'Reliability')
        self.assertEqual(the_variable_columns[29], 'Options')
        self.assertEqual(the_variable_columns[30], 'Respectful_Response')
        self.assertEqual(the_variable_columns[31], 'Courteous_Exchange')
        self.assertEqual(the_variable_columns[32], 'Active_Listening')
        self.assertEqual(the_variable_columns[33], 'Area_Rural')
        self.assertEqual(the_variable_columns[34], 'Area_Suburban')
        self.assertEqual(the_variable_columns[35], 'Area_Urban')
        self.assertEqual(the_variable_columns[36], 'Marital_Divorced')
        self.assertEqual(the_variable_columns[37], 'Marital_Married')
        self.assertEqual(the_variable_columns[38], 'Marital_Never Married')
        self.assertEqual(the_variable_columns[39], 'Marital_Separated')
        self.assertEqual(the_variable_columns[40], 'Marital_Widowed')
        self.assertEqual(the_variable_columns[41], 'Gender_Female')
        self.assertEqual(the_variable_columns[42], 'Gender_Male')
        self.assertEqual(the_variable_columns[43], 'Gender_Nonbinary')
        self.assertEqual(the_variable_columns[44], 'Contract_Month-to-month')
        self.assertEqual(the_variable_columns[45], 'Contract_One year')
        self.assertEqual(the_variable_columns[46], 'Contract_Two Year')
        self.assertEqual(the_variable_columns[47], 'InternetService_DSL')
        self.assertEqual(the_variable_columns[48], 'InternetService_Fiber Optic')
        self.assertEqual(the_variable_columns[49], 'InternetService_No response')
        self.assertEqual(the_variable_columns[50], 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(the_variable_columns[51], 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(the_variable_columns[52], 'PaymentMethod_Electronic Check')
        self.assertEqual(the_variable_columns[53], 'PaymentMethod_Mailed Check')

        # validate the datatype in the encoded_df
        self.assertEqual(str(encoded_df['Population'].dtype), 'int64')
        self.assertEqual(str(encoded_df['TimeZone'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Children'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Age'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Income'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Churn'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Outage_sec_perweek'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Email'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Contacts'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Yearly_equip_failure'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Techie'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Port_modem'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Tablet'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Phone'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Multiple'].dtype), 'bool')
        self.assertEqual(str(encoded_df['OnlineSecurity'].dtype), 'bool')
        self.assertEqual(str(encoded_df['OnlineBackup'].dtype), 'bool')
        self.assertEqual(str(encoded_df['DeviceProtection'].dtype), 'bool')
        self.assertEqual(str(encoded_df['TechSupport'].dtype), 'bool')
        self.assertEqual(str(encoded_df['StreamingTV'].dtype), 'bool')
        self.assertEqual(str(encoded_df['StreamingMovies'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaperlessBilling'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Tenure'].dtype), 'float64')
        self.assertEqual(str(encoded_df['MonthlyCharge'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Bandwidth_GB_Year'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Timely_Response'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Timely_Fixes'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Timely_Replacements'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Reliability'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Options'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Respectful_Response'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Courteous_Exchange'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Active_Listening'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Area_Rural'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Area_Suburban'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Area_Urban'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Divorced'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Married'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Never Married'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Separated'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Widowed'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Gender_Female'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Gender_Male'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Gender_Nonbinary'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Contract_Month-to-month'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Contract_One year'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Contract_Two Year'].dtype), 'bool')
        self.assertEqual(str(encoded_df['InternetService_DSL'].dtype), 'bool')
        self.assertEqual(str(encoded_df['InternetService_Fiber Optic'].dtype), 'bool')
        self.assertEqual(str(encoded_df['InternetService_No response'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Bank Transfer(automatic)'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Credit Card (automatic)'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Electronic Check'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Mailed Check'].dtype), 'bool')

        # At this point we have 54 encoded variables, and we know the exact type.

        # define the target variable
        the_target_column = 'Bandwidth_GB_Year'

        # set the target data frame (y)
        the_target_series = encoded_df[the_target_column]

        # run assertions on the_target_df
        self.assertIsInstance(the_target_series, Series)
        self.assertEqual(the_target_series.shape, (9582,))

        # remove the target variable from the_variable_columns
        the_variable_columns.remove(the_target_column)

        # make sure the_target_column is not in current_features
        self.assertIsInstance(the_variable_columns, list)
        self.assertFalse(the_target_column in the_variable_columns)
        self.assertEqual(len(the_variable_columns), 53)

        # get the feature dataframe (X)
        the_current_features_df = encoded_df[the_variable_columns]

        # run assertions on the_current_features_df
        self.assertIsInstance(the_current_features_df, DataFrame)
        self.assertEqual(the_current_features_df.shape, (9582, 53))

        # one last assertion
        self.assertFalse(the_target_column in the_current_features_df.columns.to_list())

        # Initialize the SelectKBest class to identify relevance of all features
        sk_best = SelectKBest(score_func=f_regression, k='all')

        # call fit() on our instance of SelectKBest
        sk_best.fit_transform(the_current_features_df, the_target_series)

        # run assertions
        self.assertIsNotNone(sk_best.pvalues_)
        self.assertIsInstance(sk_best.pvalues_, numpy.ndarray)
        self.assertEqual(len(sk_best.pvalues_), 53)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_current_features_df.columns,
                                 'p_value': sk_best.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on the selected features
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Churn")
        self.assertEqual(p_values['Feature'].iloc[2], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[3], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[4], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[7], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[8], "InternetService_No response")

        self.assertEqual(round(p_values['p_value'].iloc[0], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[1], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[2], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[3], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[4], 9), 0.000000002)
        self.assertEqual(round(p_values['p_value'].iloc[5], 9), 0.00000001)
        self.assertEqual(round(p_values['p_value'].iloc[6], 9), 0.000005879)
        self.assertEqual(round(p_values['p_value'].iloc[7], 9), 0.000015262)
        self.assertEqual(round(p_values['p_value'].iloc[8], 9), 0.000033956)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = the_current_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the shape of the_current_features_df
        self.assertEqual(the_current_features_df.shape, (9582, 9))

        # now we need to split our data.  The the_current_features_df should only contain our pruned list of
        # features.  Providing a variable name mapping
        # x_train--> the_f_df_train = the training features (not including the target)
        # x_test--> the_f_df_test = the test features (not including the target)
        # y_train--> the_t_var_train = the training target variable
        # y_test-->the_t_var_test = the test target variable
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(
            the_current_features_df,
            the_target_series,
            test_size=0.2,
            random_state=12345)

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

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = dataset_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
                                                                                         FLOAT64_COLUMN_KEY])

        # now we need to get the features we can scale.
        the_f_df_train_to_scale = [i for i in the_f_df_train.columns.to_list() if i in list_of_features_to_scale]

        # now we need to scale the_f_df_train, the_f_df_test only using the columns present in the list
        # the_f_df_train_to_scale
        # Note: The datatype changes from a DataFrame to a numpy.array after the call to fit_tranform().
        the_scaled_f_train_df = the_scalar.fit_transform(the_f_df_train[the_f_df_train_to_scale])
        the_scaled_f_test_df = the_scalar.fit_transform(the_f_df_test[the_f_df_train_to_scale])

        # now we need to cast the_scaled_f_train_df and the_scaled_f_test_df back to DataFrames
        the_scaled_f_train_df = pd.DataFrame(data=the_scaled_f_train_df, columns=the_f_df_train_to_scale)
        the_scaled_f_test_df = pd.DataFrame(data=the_scaled_f_test_df, columns=the_f_df_train_to_scale)

        # re-inject the scaled feature Series back into the_f_df_train and the_f_df_test.
        for the_field in the_f_df_train_to_scale:
            # update the original dataframe
            the_f_df_train[the_field] = the_scaled_f_train_df[the_field].values
            the_f_df_test[the_field] = the_scaled_f_test_df[the_field].values

        # run assertions on the type now
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertIsInstance(the_f_df_test, DataFrame)

        # create a base RandomForestRegressor to compare with.
        rfr = RandomForestRegressor(random_state=12345, oob_score=True)

        # call fit
        rfr.fit(the_f_df_train, the_t_var_train)

        # verify the out of bag score for the baseline model
        self.assertEqual(rfr.oob_score_, 0.9967121230861942)

        # get the predictions
        predictions = rfr.predict(the_f_df_test)

        # # Calculate the absolute errors
        errors = abs(predictions - the_t_var_test)

        # run assertions on the size of errors against the size of the_t_var_test
        self.assertEqual(errors.shape, (1917,))
        self.assertEqual(the_t_var_test.shape, errors.shape)

        # run assertions on mean absolute error
        self.assertEqual(round(np.mean(errors), 2), 111.05)

        # calculate the mse
        mse = mean_squared_error(the_t_var_test, predictions)

        # run assertions on the mse
        self.assertEqual(round(mse, 1), 20100)

        # get the r2
        r2 = r2_score(the_t_var_test, predictions)

        # run assertions on mean absolute error
        self.assertEqual(r2, 0.9958042668098587)

        # Now, let's test using RandomizedSearchCV

        # referenced Hyperparameter Tuning the Random Forest in Python
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

        # # Create a parameter grid for RandomizedSearchCV
        # param_grid = {'bootstrap': [True, False], 'max_depth': [None, 10, 20, 30, 50, 100],
        #               'max_features': ['auto', 'sqrt'], 'min_samples_leaf': [1, 2, 5, 10],
        #               'min_samples_split': [1, 2, 5, 10],
        #               'n_estimators': [20, 50, 100, 200, 300, 500, 1000]}
        #
        # # define the folds.  Set n_splits to 5 in final model.
        # kf = KFold(n_splits=5, shuffle=True, random_state=12345)
        #
        # rf_random = RandomizedSearchCV(estimator=rfr, param_distributions=param_grid,
        #                                n_iter=100, cv=kf, verbose=0, random_state=12345, n_jobs=-1)
        #
        # # Fit grid search
        # rf_random.fit(the_f_df_train, the_t_var_train)
        #
        # # get the predictions
        # predictions = rf_random.predict(the_f_df_test)
        #
        # # validate the best parameters and scores
        # self.assertEqual(rf_random.best_params_, {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1,
        #                                           'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': True})
        #
        # self.assertEqual(rf_random.best_score_, 0.9961921071918635)
        #
        # # calculate the mse & R^2
        # mse = mean_squared_error(the_t_var_test, predictions)
        # r2 = r2_score(the_t_var_test, predictions)
        #
        # # run assertions on the mse and R^2
        # self.assertEqual(round(mse, 1), 22411.3)
        # self.assertEqual(r2, 0.9953218010253184)
        #
        # Create a parameter grid for GridSearchCV
        param_grid = {'n_estimators': [20, 50, 100, 200, 300, 500, 1000], 'max_features': ['auto', 'sqrt'],
                      'max_depth': [None, 10, 20, 30, 50], 'min_samples_split': [2, 5, 10, 20],
                      'min_samples_leaf': [1, 2, 5, 10], 'bootstrap': [True, False]}

        # define the folds.  Set n_splits to 5 in final model.
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)

        # # create a grid search
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(the_f_df_train, the_t_var_train)

        # get the predictions
        predictions = grid_search.predict(the_f_df_test)

        # validate the best parameters and scores
        self.assertEqual(grid_search.best_params_, {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1,
                                                    'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True})

        self.assertEqual(grid_search.best_score_, -18094.330446819684)

        # calculate the mse & R^2
        mse = mean_squared_error(the_t_var_test, predictions)
        r2 = r2_score(the_t_var_test, predictions)

        # run assertions on the mse and R^2
        self.assertEqual(round(mse, 1), 22313.6)
        self.assertEqual(r2, 0.9953421876242985)

    # negative test method for get_selected_features()
    def test_get_selected_features_negative(self):
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

        # run assertion to verify that the target variable is not present
        self.assertFalse('Bandwidth_GB_Year' in the_encoded_features_list)
        self.assertFalse('Bandwidth_GB_Year' in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df['Bandwidth_GB_Year']

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)

        # verify we handle None, None, None, None
        with self.assertRaises(AttributeError) as context:
            the_model.get_selected_features(current_features_df=None,
                                            the_target=None,
                                            p_val_sig=None,
                                            model_type=None)

        # validate the error message.
        self.assertEqual("current_features_df is None or incorrect type.", context.exception.args[0])

        # verify we handle encoded_df, None, None
        with self.assertRaises(AttributeError) as context:
            the_model.get_selected_features(current_features_df=encoded_df,
                                            the_target=None,
                                            p_val_sig=None,
                                            model_type=None)

        # validate the error message.
        self.assertEqual("the_target is None or incorrect type.", context.exception.args[0])

        # verify we handle encoded_df, the_target_df, None
        with self.assertRaises(AttributeError) as context:
            the_model.get_selected_features(current_features_df=encoded_df,
                                            the_target=the_target_df,
                                            p_val_sig=None,
                                            model_type=None)

        # validate the error message.
        self.assertEqual("p_val_sig is None or incorrect type.", context.exception.args[0])

        # verify we handle encoded_df, the_target_df, 2.0
        with self.assertRaises(AttributeError) as context:
            the_model.get_selected_features(current_features_df=encoded_df,
                                            the_target=the_target_df,
                                            p_val_sig=2.0,
                                            model_type=None)

        # validate the error message.
        self.assertEqual("p_val_sig must be in [0, 1].", context.exception.args[0])

        # verify we handle encoded_df, the_target_df, -2.0
        with self.assertRaises(AttributeError) as context:
            the_model.get_selected_features(current_features_df=encoded_df,
                                            the_target=the_target_df,
                                            p_val_sig=-2.0,
                                            model_type=None)

        # validate the error message.
        self.assertEqual("p_val_sig must be in [0, 1].", context.exception.args[0])

        # verify we handle encoded_df, the_target_df, 0.001
        with self.assertRaises(SyntaxError) as context:
            the_model.get_selected_features(current_features_df=encoded_df,
                                            the_target=the_target_df,
                                            p_val_sig=0.001,
                                            model_type=None)

        # validate the error message.
        self.assertEqual("model_type must be in [MT_KNN_CLASSIFICATION, MT_RF_REGRESSION].", context.exception.msg)

    # test method for get_selected_features
    def test_get_selected_features(self):
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

        # run assertion to verify that the target variable is not present
        self.assertFalse('Bandwidth_GB_Year' in the_encoded_features_list)
        self.assertFalse('Bandwidth_GB_Year' in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df['Bandwidth_GB_Year']

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, 'Bandwidth_GB_Year')

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order that is expected
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

        # invoke the method
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # run assertions
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)

        # run assertions on the selected features
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Churn")
        self.assertEqual(p_values['Feature'].iloc[2], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[3], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[4], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[7], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[8], "InternetService_No response")

        self.assertEqual(round(p_values['p_value'].iloc[0], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[1], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[2], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[3], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[4], 9), 0.000000002)
        self.assertEqual(round(p_values['p_value'].iloc[5], 9), 0.00000001)
        self.assertEqual(round(p_values['p_value'].iloc[6], 9), 0.000005879)
        self.assertEqual(round(p_values['p_value'].iloc[7], 9), 0.000015262)
        self.assertEqual(round(p_values['p_value'].iloc[8], 9), 0.000033956)

    # negative tests for split_and_scale_features()
    def test_split_and_scale_features_negative(self):
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

        # run assertion to verify that the target variable is not present
        self.assertFalse('Bandwidth_GB_Year' in the_encoded_features_list)
        self.assertFalse('Bandwidth_GB_Year' in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df['Bandwidth_GB_Year']

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, 'Bandwidth_GB_Year')

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order that is expected
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

        # invoke the method
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # run assertions
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)

        # run assertions on the selected features
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Churn")
        self.assertEqual(p_values['Feature'].iloc[2], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[3], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[4], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[7], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[8], "InternetService_No response")

        self.assertEqual(round(p_values['p_value'].iloc[0], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[1], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[2], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[3], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[4], 9), 0.000000002)
        self.assertEqual(round(p_values['p_value'].iloc[5], 9), 0.00000001)
        self.assertEqual(round(p_values['p_value'].iloc[6], 9), 0.000005879)
        self.assertEqual(round(p_values['p_value'].iloc[7], 9), 0.000015262)
        self.assertEqual(round(p_values['p_value'].iloc[8], 9), 0.000033956)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # run assertions on the shape of the_current_features_df
        self.assertEqual(the_current_features_df.shape, (9582, 9))

        # verify we handle None, None, None
        with self.assertRaises(AttributeError) as context:
            the_model.split_and_scale_features(current_features_df=None,
                                               the_target=None,
                                               test_size=None)

        # validate the error message.
        self.assertEqual("current_features_df is None or incorrect type.", context.exception.args[0])

        # verify we handle the_current_features_df, None, None
        with self.assertRaises(AttributeError) as context:
            the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                               the_target=None,
                                               test_size=None)

        # validate the error message.
        self.assertEqual("the_target is None or incorrect type.", context.exception.args[0])

        # verify we handle the_current_features_df, the_target_df, None
        with self.assertRaises(AttributeError) as context:
            the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                               the_target=the_target_df,
                                               test_size=None)

        # validate the error message.
        self.assertEqual("test_size is None or incorrect type.", context.exception.args[0])

        # verify we handle the_current_features_df, the_target_df, 1
        with self.assertRaises(AttributeError) as context:
            the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                               the_target=the_target_df,
                                               test_size=1)

        # validate the error message.
        self.assertEqual("test_size is None or incorrect type.", context.exception.args[0])

        # verify we handle the_current_features_df, the_target_df, 2.0
        with self.assertRaises(AttributeError) as context:
            the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                               the_target=the_target_df,
                                               test_size=2.0)

        # validate the error message.
        self.assertEqual("test_size must be in [0, 1].", context.exception.args[0])

        # verify we handle the_current_features_df, the_target_df, -2.0
        with self.assertRaises(AttributeError) as context:
            the_model.split_and_scale_features(current_features_df=the_current_features_df,
                                               the_target=the_target_df,
                                               test_size=-2.0)

        # validate the error message.
        self.assertEqual("test_size must be in [0, 1].", context.exception.args[0])

    # test method for split_and_scale_features()
    def test_split_and_scale_features(self):
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

        # run assertion to verify that the target variable is not present
        self.assertFalse('Bandwidth_GB_Year' in the_encoded_features_list)
        self.assertFalse('Bandwidth_GB_Year' in encoded_df.columns.to_list())

        # set the target series
        the_target_df = the_model.encoded_df['Bandwidth_GB_Year']

        # run assertions on the the_target_df
        self.assertIsInstance(the_target_df, Series)
        self.assertEqual(the_target_df.shape, (9582,))
        self.assertEqual(the_target_df.name, 'Bandwidth_GB_Year')

        # run assertions on encoded_df
        self.assertEqual(encoded_df.shape, (9582, 53))

        # validate the order that is expected
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

        # invoke the method
        p_values = the_model.get_selected_features(current_features_df=encoded_df,
                                                   the_target=the_target_df,
                                                   p_val_sig=0.001,
                                                   model_type=MT_RF_REGRESSION)

        # run assertions
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)

        # run assertions on the selected features
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Churn")
        self.assertEqual(p_values['Feature'].iloc[2], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[3], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[4], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[7], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[8], "InternetService_No response")

        self.assertEqual(round(p_values['p_value'].iloc[0], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[1], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[2], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[3], 9), 0.0)
        self.assertEqual(round(p_values['p_value'].iloc[4], 9), 0.000000002)
        self.assertEqual(round(p_values['p_value'].iloc[5], 9), 0.00000001)
        self.assertEqual(round(p_values['p_value'].iloc[6], 9), 0.000005879)
        self.assertEqual(round(p_values['p_value'].iloc[7], 9), 0.000015262)
        self.assertEqual(round(p_values['p_value'].iloc[8], 9), 0.000033956)

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = encoded_df[p_values['Feature'].tolist()].copy()

        # invoke the method
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

    # test method for fit_a_model()
    def test_fit_a_model(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_RF_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        dataset_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(dataset_analyzer)
        self.assertIsInstance(dataset_analyzer, DatasetAnalyzer)

        # initialize class
        the_class = Random_Forest_Model(dataset_analyzer=dataset_analyzer)

        # get a reference to the encoded_df
        encoded_df = the_class.encoded_df

        # run validations on encoded_df
        self.assertIsNotNone(encoded_df)
        self.assertIsInstance(the_class.encoded_df, DataFrame)
        self.assertEqual(len(the_class.encoded_df), 9582)
        self.assertEqual(len(the_class.encoded_df.columns), 54)
        self.assertEqual(encoded_df.shape, (9582, 54))

        # get the variable columns
        the_variable_columns = encoded_df.columns.to_list()

        # run assertions on the_variable_columns
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # validate the order that is expected for each feature in the_variable_columns
        self.assertEqual(the_variable_columns[0], 'Population')
        self.assertEqual(the_variable_columns[1], 'TimeZone')
        self.assertEqual(the_variable_columns[2], 'Children')
        self.assertEqual(the_variable_columns[3], 'Age')
        self.assertEqual(the_variable_columns[4], 'Income')
        self.assertEqual(the_variable_columns[5], 'Churn')
        self.assertEqual(the_variable_columns[6], 'Outage_sec_perweek')
        self.assertEqual(the_variable_columns[7], 'Email')
        self.assertEqual(the_variable_columns[8], 'Contacts')
        self.assertEqual(the_variable_columns[9], 'Yearly_equip_failure')
        self.assertEqual(the_variable_columns[10], 'Techie')
        self.assertEqual(the_variable_columns[11], 'Port_modem')
        self.assertEqual(the_variable_columns[12], 'Tablet')
        self.assertEqual(the_variable_columns[13], 'Phone')
        self.assertEqual(the_variable_columns[14], 'Multiple')
        self.assertEqual(the_variable_columns[15], 'OnlineSecurity')
        self.assertEqual(the_variable_columns[16], 'OnlineBackup')
        self.assertEqual(the_variable_columns[17], 'DeviceProtection')
        self.assertEqual(the_variable_columns[18], 'TechSupport')
        self.assertEqual(the_variable_columns[19], 'StreamingTV')
        self.assertEqual(the_variable_columns[20], 'StreamingMovies')
        self.assertEqual(the_variable_columns[21], 'PaperlessBilling')
        self.assertEqual(the_variable_columns[22], 'Tenure')
        self.assertEqual(the_variable_columns[23], 'MonthlyCharge')
        self.assertEqual(the_variable_columns[24], 'Bandwidth_GB_Year')
        self.assertEqual(the_variable_columns[25], 'Timely_Response')
        self.assertEqual(the_variable_columns[26], 'Timely_Fixes')
        self.assertEqual(the_variable_columns[27], 'Timely_Replacements')
        self.assertEqual(the_variable_columns[28], 'Reliability')
        self.assertEqual(the_variable_columns[29], 'Options')
        self.assertEqual(the_variable_columns[30], 'Respectful_Response')
        self.assertEqual(the_variable_columns[31], 'Courteous_Exchange')
        self.assertEqual(the_variable_columns[32], 'Active_Listening')
        self.assertEqual(the_variable_columns[33], 'Area_Rural')
        self.assertEqual(the_variable_columns[34], 'Area_Suburban')
        self.assertEqual(the_variable_columns[35], 'Area_Urban')
        self.assertEqual(the_variable_columns[36], 'Marital_Divorced')
        self.assertEqual(the_variable_columns[37], 'Marital_Married')
        self.assertEqual(the_variable_columns[38], 'Marital_Never Married')
        self.assertEqual(the_variable_columns[39], 'Marital_Separated')
        self.assertEqual(the_variable_columns[40], 'Marital_Widowed')
        self.assertEqual(the_variable_columns[41], 'Gender_Female')
        self.assertEqual(the_variable_columns[42], 'Gender_Male')
        self.assertEqual(the_variable_columns[43], 'Gender_Nonbinary')
        self.assertEqual(the_variable_columns[44], 'Contract_Month-to-month')
        self.assertEqual(the_variable_columns[45], 'Contract_One year')
        self.assertEqual(the_variable_columns[46], 'Contract_Two Year')
        self.assertEqual(the_variable_columns[47], 'InternetService_DSL')
        self.assertEqual(the_variable_columns[48], 'InternetService_Fiber Optic')
        self.assertEqual(the_variable_columns[49], 'InternetService_No response')
        self.assertEqual(the_variable_columns[50], 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(the_variable_columns[51], 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(the_variable_columns[52], 'PaymentMethod_Electronic Check')
        self.assertEqual(the_variable_columns[53], 'PaymentMethod_Mailed Check')

        # validate the datatype in the encoded_df
        self.assertEqual(str(encoded_df['Population'].dtype), 'int64')
        self.assertEqual(str(encoded_df['TimeZone'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Children'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Age'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Income'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Churn'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Outage_sec_perweek'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Email'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Contacts'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Yearly_equip_failure'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Techie'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Port_modem'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Tablet'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Phone'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Multiple'].dtype), 'bool')
        self.assertEqual(str(encoded_df['OnlineSecurity'].dtype), 'bool')
        self.assertEqual(str(encoded_df['OnlineBackup'].dtype), 'bool')
        self.assertEqual(str(encoded_df['DeviceProtection'].dtype), 'bool')
        self.assertEqual(str(encoded_df['TechSupport'].dtype), 'bool')
        self.assertEqual(str(encoded_df['StreamingTV'].dtype), 'bool')
        self.assertEqual(str(encoded_df['StreamingMovies'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaperlessBilling'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Tenure'].dtype), 'float64')
        self.assertEqual(str(encoded_df['MonthlyCharge'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Bandwidth_GB_Year'].dtype), 'float64')
        self.assertEqual(str(encoded_df['Timely_Response'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Timely_Fixes'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Timely_Replacements'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Reliability'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Options'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Respectful_Response'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Courteous_Exchange'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Active_Listening'].dtype), 'int64')
        self.assertEqual(str(encoded_df['Area_Rural'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Area_Suburban'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Area_Urban'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Divorced'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Married'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Never Married'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Separated'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Marital_Widowed'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Gender_Female'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Gender_Male'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Gender_Nonbinary'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Contract_Month-to-month'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Contract_One year'].dtype), 'bool')
        self.assertEqual(str(encoded_df['Contract_Two Year'].dtype), 'bool')
        self.assertEqual(str(encoded_df['InternetService_DSL'].dtype), 'bool')
        self.assertEqual(str(encoded_df['InternetService_Fiber Optic'].dtype), 'bool')
        self.assertEqual(str(encoded_df['InternetService_No response'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Bank Transfer(automatic)'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Credit Card (automatic)'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Electronic Check'].dtype), 'bool')
        self.assertEqual(str(encoded_df['PaymentMethod_Mailed Check'].dtype), 'bool')

        # At this point we have 54 encoded variables, and we know the exact type.

        # define the target variable
        the_target_column = 'Bandwidth_GB_Year'

        # set the target data frame (y)
        the_target_series = encoded_df[the_target_column]

        # run assertions on the_target_df
        self.assertIsInstance(the_target_series, Series)
        self.assertEqual(the_target_series.shape, (9582,))

        # remove the target variable from the_variable_columns
        the_variable_columns.remove(the_target_column)

        # make sure the_target_column is not in current_features
        self.assertIsInstance(the_variable_columns, list)
        self.assertFalse(the_target_column in the_variable_columns)
        self.assertEqual(len(the_variable_columns), 53)

        # get the feature dataframe (X)
        the_current_features_df = encoded_df[the_variable_columns]

        # run assertions on the_current_features_df
        self.assertIsInstance(the_current_features_df, DataFrame)
        self.assertEqual(the_current_features_df.shape, (9582, 53))

        # invoke the method
        the_result = the_class.fit_a_model(the_target_column=the_target_column,
                                           current_features=encoded_df.columns.to_list(),
                                           model_type=MT_RF_REGRESSION)

        # run assertions on the_result
        self.assertIsNotNone(the_result)
        self.assertIsInstance(the_result, Random_Forest_Model_Result)
        self.assertIsNotNone(the_result.get_model())
        self.assertIsInstance(the_result.get_model(), RandomForestRegressor)

        self.assertEqual(the_result.get_model_best_score(), 18094.330446819684)
        self.assertEqual(the_result.get_best_params(), "bootstrap: True, max_depth: None, max_features: sqrt, "
                                                       "min_samples_leaf: 1, min_samples_split: 2, n_estimators: 1000")
        self.assertEqual(the_result.get_mean_absolute_error(), 111.0500439299238)
        self.assertEqual(the_result.get_mean_squared_error(), 20099.984180287032)
        self.assertEqual(the_result.get_root_mean_squared_error(), 141.77441299574136)
        self.assertEqual(the_result.get_r_squared(), 0.9958042668098587)
        self.assertEqual(the_result.get_number_of_obs(), 1917)

