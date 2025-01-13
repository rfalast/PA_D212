import os
import unittest
import numpy
import numpy as np
import pandas as pd

from os.path import exists
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models.KNN_Model import KNN_Model
from model.analysis.models.KNN_Model_Result import KNN_Model_Result
from model.analysis.models.ModelResultBase import ModelResultBase
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, MT_KNN_CLASSIFICATION, D_212_CHURN
from model.constants.DatasetConstants import INT64_COLUMN_KEY, FLOAT64_COLUMN_KEY
from model.constants.ModelConstants import LM_FEATURE_NUM, LM_P_VALUE, LM_PREDICTOR
from model.constants.ReportConstants import MODEL_ACCURACY, MODEL_AVG_PRECISION, MODEL_F1_SCORE, MODEL_ROC_SCORE, \
    MODEL_PRECISION, MODEL_RECALL, MODEL_Y_PRED, MODEL_Y_SCORES, NUMBER_OF_OBS, MODEL_BEST_SCORE, MODEL_BEST_PARAMS
from util.CSV_loader import CSV_Loader


class test_KNN_Model_Result(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"
    OVERRIDE_PATH = "../../../../resources/Output/"
    VALID_CSV_PATH = "../../resources/Input/churn_raw_data.csv"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    params_dict_str = 'leaf_size: 2, n_neighbors: 4, p: 1'

    CHURN_KEY = D_212_CHURN

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # at this point, we're ready to run negative tests of our class.

        # Note: due to the call to the superclass, the variables validated are in this order
        # the_target_variable, the_variables_list, the_encoded_df

        # create the argument_dict
        argument_dict = dict()

        # verify we handle None, None, None, None, None, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=None)

        # validate the error message.
        self.assertTrue("argument_dict is None or incorrect type.", context.exception.args)

        # verify we handle None, None, None, None, None, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_target_variable is None or incorrect type.", context.exception.args)

        # add the_target_variable
        argument_dict['the_target_variable'] = 'Churn'

        # add the_target_variable, the_variables_list should fail.
        # verify we handle None, 'Churn', None, None, None, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_variables_list is None or incorrect type.", context.exception.args)

        # add the_target_variable
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()

        # add the_variables_list, the_encoded_df should fail
        # verify we handle None, 'Churn', the_features_df.columns.to_list(), None, None, None
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_encoded_df is None or incorrect type.", context.exception.args)

        # add the_encoded_df
        argument_dict['the_encoded_df'] = the_features_df

        # add the_encoded_df, the_model should fail
        # verify we handle None, 'Churn', the_features_df.columns.to_list(), None, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_model is None or incorrect type.", context.exception.args)

        # add the_model
        argument_dict['the_model'] = knn

        # add the_model, the_f_df_train should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), None, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_f_df_train is None or incorrect type.", context.exception.args)

        # add the_f_df_train
        argument_dict['the_f_df_train'] = the_f_df_train

        # add the_model, the_f_df_test should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), None, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_f_df_test is None or incorrect type.", context.exception.args)

        # add the_f_df_test
        argument_dict['the_f_df_test'] = the_f_df_train

        # add the_f_df_test, the_t_var_test should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_t_var_train is None or incorrect type.", context.exception.args)

        # add the_t_var_train
        argument_dict['the_t_var_train'] = the_t_var_train

        # add the_f_df_test, the_t_var_test should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_t_var_test is None or incorrect type.", context.exception.args)

        # add the_t_var_test
        argument_dict['the_t_var_test'] = the_t_var_test

        # add the_f_df_test, the_p_values should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_p_values is None or incorrect type.", context.exception.args)

        # add the_p_values
        argument_dict['the_p_values'] = p_values

        # add the_p_values, gridsearch should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("gridsearch is None or incorrect type.", context.exception.args)

        # add gridsearch
        argument_dict['gridsearch'] = knn_cv

        # add the_p_values, gridsearch should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("prepared_data is None or incorrect type.", context.exception.args)

        # add prepared_data
        argument_dict['prepared_data'] = the_features_df

        # add the_p_values, gridsearch should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("cleaned_data is None or incorrect type.", context.exception.args)

        # add cleaned_data
        argument_dict['cleaned_data'] = the_df

        # add cleaned_data, the_df should fail
        # verify we handle knn, 'Churn', the_features_df.columns.to_list(), the_f_df_test, None, the_features_df
        with self.assertRaises(AttributeError) as context:
            # invoke the method
            KNN_Model_Result(argument_dict=argument_dict)

        # validate the error message.
        self.assertTrue("the_df is missing.", context.exception.args)

        # add the_df
        argument_dict['the_df'] = the_df

        # make sure there are no additional checks we skipped for the arguments
        KNN_Model_Result(argument_dict=argument_dict)

    # test method for init()
    def test_init(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # run assertions on parameters of KNN_Model_Result
        self.assertIsNotNone(the_knn_result.model)
        self.assertIsNotNone(the_knn_result.y_pred)
        self.assertIsNotNone(the_knn_result.the_f_df_train)
        self.assertIsNotNone(the_knn_result.the_f_df_test)
        self.assertIsNotNone(the_knn_result.the_t_var_train)
        self.assertIsNotNone(the_knn_result.the_t_var_test)
        self.assertIsNotNone(the_knn_result.the_df)
        self.assertIsNotNone(the_knn_result.assumptions)
        self.assertIsNotNone(the_knn_result.the_p_values)
        self.assertIsNotNone(the_knn_result.grid_search)
        self.assertIsNotNone(the_knn_result.prepared_data)
        self.assertIsNotNone(the_knn_result.cleaned_data)

        # validate the data type
        self.assertIsInstance(the_knn_result.the_f_df_train, DataFrame)
        self.assertIsInstance(the_knn_result.the_f_df_test, DataFrame)
        self.assertIsInstance(the_knn_result.the_t_var_train, DataFrame)
        self.assertIsInstance(the_knn_result.the_t_var_test, DataFrame)
        self.assertIsInstance(the_knn_result.the_p_values, DataFrame)
        self.assertIsInstance(the_knn_result.grid_search, GridSearchCV)
        self.assertIsInstance(the_knn_result.prepared_data, DataFrame)
        self.assertIsInstance(the_knn_result.cleaned_data, DataFrame)

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_model())
        self.assertIsInstance(the_knn_result.get_model(), KNeighborsClassifier)

    # test method for get_accuracy()
    def test_get_accuracy(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_accuracy())
        self.assertIsInstance(the_knn_result.get_accuracy(), float)
        self.assertEqual(the_knn_result.get_accuracy(), 0.8855652173913043)

    # test method for get_avg_precision()
    def test_get_avg_precision(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())


        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_avg_precision())
        self.assertIsInstance(the_knn_result.get_avg_precision(), float)
        self.assertEqual(the_knn_result.get_avg_precision(), 0.6578707642960415)

    # test method for get_f1_score()
    def test_get_f1_score(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_f1_score())
        self.assertIsInstance(the_knn_result.get_f1_score(), float)
        self.assertEqual(the_knn_result.get_f1_score(), 0.7589743589743589)

    # test method for get_precision()
    def test_get_precision(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_precision())
        self.assertIsInstance(the_knn_result.get_precision(), float)
        self.assertEqual(the_knn_result.get_precision(), 0.8301282051282052)

    # test method for get_recall()
    def test_get_recall(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_recall())
        self.assertIsInstance(the_knn_result.get_recall(), float)
        self.assertEqual(the_knn_result.get_recall(), 0.699055330634278)

    # test method for get_roc_score()
    def test_get_roc_score(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_roc_score())
        self.assertIsInstance(the_knn_result.get_roc_score(), float)
        self.assertEqual(the_knn_result.get_roc_score(), 0.8246916765636246)

    # test method for get_y_predicted()
    def test_get_y_predicted(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_y_predicted())
        self.assertIsInstance(the_knn_result.get_y_predicted(), numpy.ndarray)
        self.assertEqual(len(the_knn_result.get_y_predicted()), 2875)

    # test method for get_y_scores()
    def test_get_y_scores(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())


        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_y_scores())
        self.assertIsInstance(the_knn_result.get_y_scores(), numpy.ndarray)
        self.assertEqual(len(the_knn_result.get_y_scores()), 2875)

    # test method for get_number_of_obs()
    def test_get_number_of_obs(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_number_of_obs())
        self.assertIsInstance(the_knn_result.get_number_of_obs(), int)
        self.assertEqual(the_knn_result.get_number_of_obs(), 2875)

    # test method for get_params()
    def test_get_params(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        self.assertIsNotNone(the_knn_result.get_best_params())
        self.assertIsInstance(the_knn_result.get_best_params(), str)

        the_params = the_knn_result.get_best_params()

        self.assertEqual(the_params, self.params_dict_str)

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method.  populate_assumptions() is called in ModelResultBase.init()
        the_knn_result.populate_assumptions()

        # run assertions on assumptions.
        self.assertIsNotNone(the_knn_result.get_assumptions())
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_ACCURACY])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_AVG_PRECISION])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_F1_SCORE])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_PRECISION])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_RECALL])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_ROC_SCORE])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_Y_PRED])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_Y_SCORES])
        self.assertIsNotNone(the_knn_result.get_assumptions()[NUMBER_OF_OBS])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_BEST_SCORE])
        self.assertIsNotNone(the_knn_result.get_assumptions()[MODEL_BEST_PARAMS])

        # test actual values of assumptions
        the_assumptions = the_knn_result.get_assumptions()

        self.assertEqual(the_assumptions[MODEL_ACCURACY], 'get_accuracy')
        self.assertEqual(the_assumptions[MODEL_AVG_PRECISION], 'get_avg_precision')
        self.assertEqual(the_assumptions[MODEL_F1_SCORE], 'get_f1_score')
        self.assertEqual(the_assumptions[MODEL_PRECISION], 'get_precision')
        self.assertEqual(the_assumptions[MODEL_RECALL], 'get_recall')
        self.assertEqual(the_assumptions[MODEL_ROC_SCORE], 'get_roc_score')
        self.assertEqual(the_assumptions[MODEL_Y_PRED], 'get_y_predicted')
        self.assertEqual(the_assumptions[MODEL_Y_SCORES], 'get_y_scores')
        self.assertEqual(the_assumptions[NUMBER_OF_OBS], 'get_number_of_obs')
        self.assertEqual(the_assumptions[MODEL_BEST_SCORE], 'get_model_best_score')
        self.assertEqual(the_assumptions[MODEL_BEST_PARAMS], 'get_best_params')

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # validate the default status of get_feature_columns()
        self.assertIsNone(the_knn_result.get_feature_columns())

        # get the results_df
        results_df = the_knn_result.get_results_dataframe()

        # run assertions
        self.assertIsNotNone(results_df)
        self.assertIsInstance(results_df, DataFrame)
        self.assertEqual(len(results_df), 15)

        # validate the columns in the results_df
        self.assertEqual(list(results_df.columns), [LM_FEATURE_NUM, LM_P_VALUE])
        self.assertEqual(results_df.index.name, LM_PREDICTOR)

        # validate that status of get_feature_columns()
        self.assertIsNotNone(the_knn_result.get_feature_columns())
        self.assertIsInstance(the_knn_result.get_feature_columns(), list)

        # capture the feature columns list
        feature_column_list = the_knn_result.get_feature_columns()

        # run assertions on feature_column_list
        self.assertEqual(len(feature_column_list), 2)
        self.assertTrue(LM_FEATURE_NUM in feature_column_list)
        self.assertTrue(LM_P_VALUE in feature_column_list)

    # test method for get_the_p_values()
    def test_get_the_p_values(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        p_values = the_knn_result.get_the_p_values()

        # run assertions
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, dict)
        self.assertEqual(len(p_values), 15)

        # validate the actual p_value for each feature.
        self.assertEqual(p_values["Tenure"], 0.0)
        self.assertEqual(p_values["Bandwidth_GB_Year"], 0.0)
        self.assertEqual(p_values["MonthlyCharge"], 0.0)
        self.assertEqual(p_values["StreamingMovies"], 0.0)
        self.assertEqual(p_values["Contract_Month-to-month"], 0.0)
        self.assertEqual(p_values["StreamingTV"], 0.0)
        self.assertEqual(p_values["Contract_Two Year"], 0.0)
        self.assertEqual(p_values["Contract_One year"], 0.0)
        self.assertEqual(p_values["Multiple"], 0.0)
        self.assertEqual(p_values["InternetService_DSL"], 0.0)
        self.assertEqual(p_values["Techie"], 0.0)
        self.assertEqual(p_values["InternetService_Fiber Optic"], 0.0)
        self.assertEqual(p_values["DeviceProtection"], 0.0)
        self.assertEqual(p_values["OnlineBackup"], 0.000002)
        self.assertEqual(p_values["InternetService_No response"], 0.000064)

    # test method for get_feature_columns()
    def test_get_feature_columns(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # validate the default status of get_feature_columns()
        self.assertIsNone(the_knn_result.get_feature_columns())

        # call get_results_dataframe()
        the_knn_result.get_results_dataframe()

        # capture the feature columns list
        feature_column_list = the_knn_result.get_feature_columns()

        # run assertions on feature_column_list
        self.assertEqual(len(feature_column_list), 2)
        self.assertTrue(LM_FEATURE_NUM in feature_column_list)
        self.assertTrue(LM_P_VALUE in feature_column_list)

    # test method for get_confusion_matrix()
    def test_get_confusion_matrix(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # invoke the method
        c_matrix = the_knn_result.get_confusion_matrix()

        # run assertions
        self.assertIsNotNone(c_matrix)
        self.assertIsInstance(c_matrix, numpy.ndarray)

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

        # create CSV_Loader
        csv_loader = CSV_Loader()

        # verify we handle None
        with self.assertRaises(SyntaxError) as context:
            the_knn_result.generate_model_csv_files(csv_loader=None)

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize the initial_model
        the_model = KNN_Model(the_analyzer)

        # run assertions on the_model
        self.assertIsNotNone(the_model)
        self.assertIsInstance(the_model, KNN_Model)
        self.assertEqual(the_model.model_type, MT_KNN_CLASSIFICATION)

        # get the base dataframe
        the_df = the_analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe, set drop_first to False
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable column names only
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)

        # make sure that 'Churn' is in the list the_variable_columns
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the features dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions on the_target_variable
        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # run assertions on the_feature_variables
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 53)

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 15)
        self.assertEqual(len(p_values.columns), 2)

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[1], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[8], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 15)

        # now we need to split our data
        # X_train -> the_f_df_train
        # X_test -> the_f_df_test
        # y_train -> the_t_var_train
        # y_test -> the_t_var_test
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 15)
        self.assertEqual(the_f_df_train.shape, (6707, 15))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 15)
        self.assertEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6707)
        self.assertEqual(the_t_var_train.shape, (6707,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2875)
        self.assertEqual(the_t_var_test.shape, (2875,))

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
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

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # define the folds, use a smaller value
        kf = KFold(n_splits=2, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 5, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2)}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        ########
        # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
        # an array, or else you'll get a warning that a column-vector y was passed.
        knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble model.
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8717757093612801)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # create the argument_dict
        argument_dict = dict()

        # populate argument_dict
        argument_dict['the_model'] = knn
        argument_dict['the_target_variable'] = 'Churn'
        argument_dict['the_variables_list'] = the_features_df.columns.to_list()
        argument_dict['the_f_df_train'] = the_f_df_train
        argument_dict['the_f_df_test'] = the_f_df_test
        argument_dict['the_t_var_train'] = the_t_var_train
        argument_dict['the_t_var_test'] = the_t_var_test
        argument_dict['the_encoded_df'] = the_features_df
        argument_dict['the_p_values'] = p_values
        argument_dict['gridsearch'] = knn_cv
        argument_dict['prepared_data'] = the_features_df
        argument_dict['cleaned_data'] = the_df
        argument_dict['the_df'] = the_df

        # create the result
        the_knn_result = KNN_Model_Result(argument_dict=argument_dict)

        # run assertions on the result
        self.assertIsNotNone(the_knn_result)
        self.assertIsInstance(the_knn_result, KNN_Model_Result)
        self.assertIsInstance(the_knn_result, ModelResultBase)

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
        the_knn_result.generate_model_csv_files(csv_loader=pa.csv_l)

        # run assertions
        self.assertTrue(exists("../../../../resources/Output/churn_cleaned.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_X_train.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_X_test.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_Y_train.csv"))
        self.assertTrue(exists("../../../../resources/Output/churn_Y_test.csv"))
