import os
import unittest
import numpy
import numpy as np
import pandas as pd

from os.path import exists
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from model.Project_Assessment import Project_Assessment
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models.KNN_Model import KNN_Model
from model.analysis.models.KNN_Model_Result import KNN_Model_Result
from model.constants.BasicConstants import ANALYZE_DATASET_FULL, D_212_CHURN, MT_KNN_CLASSIFICATION
from model.constants.DatasetConstants import INT64_COLUMN_KEY, FLOAT64_COLUMN_KEY, BOOL_COLUMN_KEY, OBJECT_COLUMN_KEY


class test_KNN_Model(unittest.TestCase):
    # test constants
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"
    OVERRIDE_PATH = "../../../../resources/Output/"

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_212_CHURN

    # test method for __init__()
    def test_init(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        # pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the dataset analyzer
        the_analyzer = pa.analyzer

        # run assertions
        self.assertIsNotNone(the_analyzer)
        self.assertIsInstance(the_analyzer, DatasetAnalyzer)

        # initialize class
        the_class = KNN_Model(dataset_analyzer=the_analyzer)

        # this test needs to make sure that we don't have any encoded variables excluded, which
        # you don't want to do for a KNN model.  Collinearity is not an issue.

        # run validations on encoded_df
        self.assertIsNotNone(the_class.encoded_df)
        self.assertIsInstance(the_class.encoded_df, DataFrame)
        self.assertEqual(len(the_class.encoded_df), 10000)
        self.assertEqual(len(the_class.encoded_df.columns), 54)

        # define endcoded_df
        encoded_df = the_class.encoded_df

        # run assertions that the list of object variables are missing
        self.assertFalse('Area' in encoded_df)
        self.assertFalse('Marital' in encoded_df)
        self.assertFalse('Gender' in encoded_df)
        self.assertFalse('Contract' in encoded_df)
        self.assertFalse('InternetService' in encoded_df)
        self.assertFalse('PaymentMethod' in encoded_df)

        # run assertions that the new encoded columns are present
        self.assertTrue('Area_Rural' in encoded_df)  # due to drop_first=True
        self.assertTrue('Area_Suburban' in encoded_df)
        self.assertTrue('Area_Urban' in encoded_df)

        self.assertTrue('Marital_Divorced' in encoded_df)  # due to drop_first=True
        self.assertTrue('Marital_Married' in encoded_df)
        self.assertTrue('Marital_Never Married' in encoded_df)
        self.assertTrue('Marital_Separated' in encoded_df)
        self.assertTrue('Marital_Widowed' in encoded_df)

        self.assertTrue('Gender_Female' in encoded_df)  # due to drop_first=True
        self.assertTrue('Gender_Male' in encoded_df)
        self.assertTrue('Gender_Nonbinary' in encoded_df)

        self.assertTrue('Contract_Month-to-month' in encoded_df)  # due to drop_first=True
        self.assertTrue('Contract_One year' in encoded_df)
        self.assertTrue('Contract_Two Year' in encoded_df)

        self.assertTrue('InternetService_DSL' in encoded_df)  # due to drop_first=True
        self.assertTrue('InternetService_Fiber Optic' in encoded_df)
        self.assertTrue('InternetService_No response' in encoded_df)

        self.assertTrue('PaymentMethod_Bank Transfer(automatic)' in encoded_df)  # due to drop_first=True
        self.assertTrue('PaymentMethod_Credit Card (automatic)' in encoded_df)
        self.assertTrue('PaymentMethod_Electronic Check' in encoded_df)
        self.assertTrue('PaymentMethod_Mailed Check' in encoded_df)

    # proof of concept how to use KNN
    def test_proof_of_concept_on_KNN(self):
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

        # get the base dataframe
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable columns
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the dataframe itself from the variable_encoder
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

        # this is the order of operations that we are going to use for KNN
        # 1) Feature select
        # 2) Split
        # 3) scale - this has to fall after split or you risk data leakage.
        # 4) determine optimum K value
        # 5) fit KNN model

        # ************************************ SELECT ************************************
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

        # print(f"p_values\n{p_values}")

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

        # ************************************ SPLIT ************************************
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

        # ************************************ SCALE ************************************
        # now that we've selected our features and split our data, we need to scale the data

        # create a StandardScaler
        the_scalar = StandardScaler()

        # scale the_f_df_train, the_f_df_test
        the_f_df_train = the_scalar.fit_transform(the_f_df_train)
        the_f_df_test = the_scalar.fit_transform(the_f_df_test)

        # reshape and recast
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # Now we have selected our features, split the dataset, scale the dataset,
        # so we're ready to find our optimum K.  I'm first going to attempt to use

        train_score = {}
        test_score = {}
        n_neighbors = np.arange(2, 30, 1)

        # taken from https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7

        for neighbor in n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto')

            # if you don't call the_t_var_train.values.ravel(), you'll get an ugly warning.
            knn.fit(the_f_df_train, the_t_var_train.values.ravel())
            train_score[neighbor] = knn.score(the_f_df_train, the_t_var_train)
            test_score[neighbor] = knn.score(the_f_df_test, the_t_var_test)

        if exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"):
            os.remove(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")

        # generate a plot of results
        plt.clf()
        plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
        plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
        plt.xlabel("Number Of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN: Varying number of Neighbors")
        plt.legend()
        plt.xlim(0, 33)
        plt.ylim(0.60, 1.00)
        plt.grid()

        plt.savefig(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")
        plt.close()

        # run assertions on the graph
        self.assertTrue(exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"))

        # run assertions on test_score
        self.assertIsNotNone(test_score)
        self.assertIsInstance(test_score, dict)
        self.assertEqual(max(test_score.values()), 0.896695652173913)

        # get the best k
        best_k = list({i for i in test_score if test_score[i] == max(test_score.values())})

        # run assertions on best k
        self.assertIsNotNone(best_k)
        self.assertIsInstance(best_k, list)
        self.assertEqual(len(best_k), 1)
        self.assertEqual(best_k[0], 15)

        # define the folds
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 30, 1), 'leaf_size': np.arange(2, 10, 1), 'p': (1, 2),
                      'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # print out the best params
        # print(knn_cv.best_params_)

        # print out the best score
        # print('Best Score - KNN:', knn_cv.best_score_)

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8892200695479385)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # predict based on our test data
        y_pred = knn.predict(the_f_df_test)

        print('Accuracy Score - KNN:', metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred))
        print('Average Precision - KNN:', metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred))
        print('F1 Score - KNN:', metrics.f1_score(the_t_var_test.values.ravel(), y_pred))
        print('Precision - KNN:', metrics.precision_score(the_t_var_test.values.ravel(), y_pred))
        print('Recall - KNN:', metrics.recall_score(the_t_var_test.values.ravel(), y_pred))
        print('ROC Score - KNN:', metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred))

        # run assertions
        self.assertEqual(metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred), 0.8900869565217391)
        self.assertEqual(metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred), 0.6712833357210726)
        self.assertEqual(metrics.f1_score(the_t_var_test.values.ravel(), y_pred), 0.7771509167842031)
        self.assertEqual(metrics.precision_score(the_t_var_test.values.ravel(), y_pred), 0.8138847858197932)
        self.assertEqual(metrics.recall_score(the_t_var_test.values.ravel(), y_pred), 0.7435897435897436)
        self.assertEqual(metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred), 0.8422728474274866)

        # https://www.datasklr.com/select-classification-methods/k-nearest-neighbors

    # proof of concept using KNN for outlier detection
    def test_proof_of_concept_using_knn_for_outlier_detection(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)

        # get the base dataframe
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 10000)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable columns
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # run assertions to make sure nothing went goofy
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 54)
        self.assertEqual(len(the_df), 10000)

        # https://towardsdatascience.com/k-nearest-neighbors-knn-for-anomaly-detection-fdf8ee160d13

        # instantiate model
        nbrs = NearestNeighbors(n_neighbors=3)

        # fit model
        nbrs.fit(the_df)

        # distances and indexes of k-neighbors from model outputs
        distances, indexes = nbrs.kneighbors(the_df)

        if exists(self.OVERRIDE_PATH + "KNN_outlier.png"):
            os.remove(self.OVERRIDE_PATH + "KNN_outlier.png")

        # clear the plot
        plt.clf()

        # plot mean of k-distances of each observation
        plt.plot(distances.mean(axis=1))

        plt.savefig(self.OVERRIDE_PATH + "KNN_outlier.png")
        plt.close()

        # run assertions on the graph
        self.assertTrue(exists(self.OVERRIDE_PATH + "KNN_outlier.png"))

        # visually determine cutoff values > 1000
        outlier_index = np.where(distances.mean(axis=1) > 1200)

        # print(f"outlier_index-->{outlier_index[0].tolist()}")
        # print(f"size is {len(outlier_index[0].tolist())}")

        the_df.drop(outlier_index[0], axis=0, inplace=True)

        # run assertions on the remaining result
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 54)
        self.assertEqual(len(the_df), 9199)

        # ************************************ SELECT ************************************

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')  # k= features

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9199)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.05
        p_values = p_values[p_values['p_value'] < .02]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 18)
        self.assertEqual(len(p_values.columns), 2)

        # print(f"p_values\n{p_values}")

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[1], "Tenure")
        self.assertEqual(p_values['Feature'].iloc[2], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[3], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[4], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[5], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[7], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[8], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[9], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[10], "Techie")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[13], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[14], "InternetService_No response")
        self.assertEqual(p_values['Feature'].iloc[15], "PaymentMethod_Electronic Check")
        self.assertEqual(p_values['Feature'].iloc[16], "Phone")
        self.assertEqual(p_values['Feature'].iloc[17], "Gender_Male")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9199)
        self.assertEqual(len(the_features_df.columns), 18)

        # ************************************ SPLIT ************************************
        # now we need to split our data
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6439)
        self.assertEqual(len(the_f_df_train.columns), 18)
        self.assertEqual(the_f_df_train.shape, (6439, 18))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2760)
        self.assertEqual(len(the_f_df_test.columns), 18)
        self.assertEqual(the_f_df_test.shape, (2760, 18))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6439)
        self.assertEqual(the_t_var_train.shape, (6439,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2760)
        self.assertEqual(the_t_var_test.shape, (2760,))

        # ************************************ SCALE ************************************
        # now that we've selected our features and split our data, we need to scale the data

        # create a StandardScaler
        the_scalar = StandardScaler()

        # scale the_f_df_train, the_f_df_test
        the_f_df_train = the_scalar.fit_transform(the_f_df_train)
        the_f_df_test = the_scalar.fit_transform(the_f_df_test)

        # scale the_t_var_train, the_t_var_test.
        # before we scaled the target data, we need to reshape it.
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # Now we have selected our features, split the dataset, scaled the dataset,
        # so we're ready to find our optimum K.  I'm first going to attempt to use

        train_score = {}
        test_score = {}
        n_neighbors = np.arange(2, 30, 1)

        # taken from https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7

        for neighbor in n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto')

            # if you don't call the_t_var_train.values.ravel(), you'll get an ugly warning.
            knn.fit(the_f_df_train, the_t_var_train.values.ravel())
            train_score[neighbor] = knn.score(the_f_df_train, the_t_var_train)
            test_score[neighbor] = knn.score(the_f_df_test, the_t_var_test)

        if exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"):
            os.remove(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")

        # generate a plot of results
        plt.clf()
        plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
        plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
        plt.xlabel("Number Of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN: Varying number of Neighbors")
        plt.legend()
        plt.xlim(0, 33)
        plt.ylim(0.60, 1.00)
        plt.grid()

        plt.savefig(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")
        plt.close()

        # run assertions on the graph
        self.assertTrue(exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"))

        # run assertions on test_score
        self.assertIsNotNone(test_score)
        self.assertIsInstance(test_score, dict)
        self.assertEqual(max(test_score.values()), 0.8818840579710145)

        # get the best k
        best_k = list({i for i in test_score if test_score[i] == max(test_score.values())})

        # run assertions on best k
        self.assertIsNotNone(best_k)
        self.assertIsInstance(best_k, list)
        self.assertEqual(len(best_k), 1)
        self.assertEqual(best_k[0], 23)

        # define the folds
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 30, 1), 'leaf_size': np.arange(2, 10, 1), 'p': (1, 2),
                      'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # print out the best params
        # print(knn_cv.best_params_)

        # print out the best score
        # print('Best Score - KNN:', knn_cv.best_score_)

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8858555695512218)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # predict based on our test data
        y_pred = knn.predict(the_f_df_test)

        # print('Accuracy Score - KNN:', metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred))
        # print('Average Precision - KNN:', metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred))
        # print('F1 Score - KNN:', metrics.f1_score(the_t_var_test.values.ravel(), y_pred))
        # print('Precision - KNN:', metrics.precision_score(the_t_var_test.values.ravel(), y_pred))
        # print('Recall - KNN:', metrics.recall_score(the_t_var_test.values.ravel(), y_pred))
        # print('ROC Score - KNN:', metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred))

        # run assertions
        self.assertEqual(metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred), 0.8869565217391304)
        self.assertEqual(metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred), 0.6762394735405299)
        self.assertEqual(metrics.f1_score(the_t_var_test.values.ravel(), y_pred), 0.7735849056603774)
        self.assertEqual(metrics.precision_score(the_t_var_test.values.ravel(), y_pred), 0.8341158059467919)
        self.assertEqual(metrics.recall_score(the_t_var_test.values.ravel(), y_pred), 0.7212449255751014)
        self.assertEqual(metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred), 0.8343978215208511)

    # proof of concept using MCD for outlier detection
    def test_proof_of_concept_using_mcd_for_outlier_detection(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # get the base dataframe
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable columns
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions to make sure nothing went goofy
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df.columns), 53)
        self.assertEqual(len(the_features_df), 9582)

        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # ************************************ SELECT ************************************
        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')  # k= features

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.05
        p_values = p_values[p_values['p_value'] < .05]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 20)
        self.assertEqual(len(p_values.columns), 2)

        # print(f"p_values\n{p_values}")

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
        self.assertEqual(p_values['Feature'].iloc[15], "PaymentMethod_Electronic Check")
        self.assertEqual(p_values['Feature'].iloc[16], "Gender_Male")
        self.assertEqual(p_values['Feature'].iloc[17], "Phone")
        self.assertEqual(p_values['Feature'].iloc[18], "Gender_Female")
        self.assertEqual(p_values['Feature'].iloc[19], "Marital_Never Married")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 20)

        # ************************************ SPLIT ************************************
        # now we need to split our data
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 20)
        self.assertEqual(the_f_df_train.shape, (6707, 20))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 20)
        self.assertEqual(the_f_df_test.shape, (2875, 20))

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

        # ************************************ SCALE ************************************
        # now that we've selected our features and split our data, we need to scale the data

        # create a StandardScaler
        the_scalar = StandardScaler()

        # scale the_f_df_train, the_f_df_test
        the_f_df_train = the_scalar.fit_transform(the_f_df_train)
        the_f_df_test = the_scalar.fit_transform(the_f_df_test)

        # scale the_t_var_train, the_t_var_test.
        # before we scaled the target data, we need to reshape it.
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # Now we have selected our features, split the dataset, scaled the dataset,
        # so we're ready to find our optimum K.  I'm first going to attempt to use

        train_score = {}
        test_score = {}
        n_neighbors = np.arange(2, 30, 1)

        # taken from https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7

        for neighbor in n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto')

            # if you don't call the_t_var_train.values.ravel(), you'll get an ugly warning.
            knn.fit(the_f_df_train, the_t_var_train.values.ravel())
            train_score[neighbor] = knn.score(the_f_df_train, the_t_var_train)
            test_score[neighbor] = knn.score(the_f_df_test, the_t_var_test)

        if exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"):
            os.remove(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")

        # generate a plot of results
        plt.clf()
        plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
        plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
        plt.xlabel("Number Of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN: Varying number of Neighbors")
        plt.legend()
        plt.xlim(0, 33)
        plt.ylim(0.60, 1.00)
        plt.grid()

        plt.savefig(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")
        plt.close()

        # run assertions on the graph
        self.assertTrue(exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"))

        # run assertions on test_score
        self.assertIsNotNone(test_score)
        self.assertIsInstance(test_score, dict)
        self.assertEqual(max(test_score.values()), 0.8890434782608696)

        # get the best k
        best_k = list({i for i in test_score if test_score[i] == max(test_score.values())})

        # run assertions on best k
        self.assertIsNotNone(best_k)
        self.assertIsInstance(best_k, list)
        self.assertEqual(len(best_k), 1)
        self.assertEqual(best_k[0], 27)

        # define the folds
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 30, 1), 'leaf_size': np.arange(2, 10, 1), 'p': (1, 2),
                      'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # print out the best params
        print(f"best params-->{knn_cv.best_params_}")

        # print out the best score
        # print('Best Score - KNN:', knn_cv.best_score_)

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8823610736032345)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # predict based on our test data
        y_pred = knn.predict(the_f_df_test)

        print('Accuracy Score - KNN:', metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred))
        print('Average Precision - KNN:', metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred))
        print('F1 Score - KNN:', metrics.f1_score(the_t_var_test.values.ravel(), y_pred))
        print('Precision - KNN:', metrics.precision_score(the_t_var_test.values.ravel(), y_pred))
        print('Recall - KNN:', metrics.recall_score(the_t_var_test.values.ravel(), y_pred))
        print('ROC Score - KNN:', metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred))

        # run assertions
        self.assertEqual(metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred), 0.8883478260869565)
        self.assertEqual(metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred), 0.6659114908444251)
        self.assertEqual(metrics.f1_score(the_t_var_test.values.ravel(), y_pred), 0.7692307692307693)
        self.assertEqual(metrics.precision_score(the_t_var_test.values.ravel(), y_pred), 0.823076923076923)
        self.assertEqual(metrics.recall_score(the_t_var_test.values.ravel(), y_pred), 0.7219973009446694)
        self.assertEqual(metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred), 0.8340539456925784)

    # proof of concept using MCD for outlier detection, along with lower p-values for feature selection
    def test_proof_of_concept_using_mcd_for_outlier_detection_lower_pvalues(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # get the base dataframe
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 39)
        self.assertEqual(len(the_df), 9582)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable columns
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 54)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions to make sure nothing went goofy
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df.columns), 53)
        self.assertEqual(len(the_features_df), 9582)

        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9582)

        # ************************************ SELECT ************************************
        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')  # k= features

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9582)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.01
        p_values = p_values[p_values['p_value'] < .01]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 16)
        self.assertEqual(len(p_values.columns), 2)

        # print(f"p_values\n{p_values}")

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
        self.assertEqual(p_values['Feature'].iloc[15], "PaymentMethod_Electronic Check")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9582)
        self.assertEqual(len(the_features_df.columns), 16)

        # ************************************ SPLIT ************************************
        # now we need to split our data
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6707)
        self.assertEqual(len(the_f_df_train.columns), 16)
        self.assertEqual(the_f_df_train.shape, (6707, 16))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2875)
        self.assertEqual(len(the_f_df_test.columns), 16)
        self.assertEqual(the_f_df_test.shape, (2875, 16))

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

        # ************************************ SCALE ************************************
        # now that we've selected our features and split our data, we need to scale the data

        # create a StandardScaler
        the_scalar = StandardScaler()

        # scale the_f_df_train, the_f_df_test
        the_f_df_train = the_scalar.fit_transform(the_f_df_train)
        the_f_df_test = the_scalar.fit_transform(the_f_df_test)

        # scale the_t_var_train, the_t_var_test.
        # before we scaled the target data, we need to reshape it.
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # Now we have selected our features, split the dataset, scaled the dataset,
        # so we're ready to find our optimum K.  I'm first going to attempt to use

        train_score = {}
        test_score = {}
        n_neighbors = np.arange(2, 30, 1)

        # taken from https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7

        for neighbor in n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto')

            # if you don't call the_t_var_train.values.ravel(), you'll get an ugly warning.
            knn.fit(the_f_df_train, the_t_var_train.values.ravel())
            train_score[neighbor] = knn.score(the_f_df_train, the_t_var_train)
            test_score[neighbor] = knn.score(the_f_df_test, the_t_var_test)

        if exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"):
            os.remove(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")

        # generate a plot of results
        plt.clf()
        plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
        plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
        plt.xlabel("Number Of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN: Varying number of Neighbors")
        plt.legend()
        plt.xlim(0, 33)
        plt.ylim(0.60, 1.00)
        plt.grid()

        plt.savefig(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")
        plt.close()

        # run assertions on the graph
        self.assertTrue(exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"))

        # run assertions on test_score
        self.assertIsNotNone(test_score)
        self.assertIsInstance(test_score, dict)
        self.assertEqual(max(test_score.values()), 0.8953043478260869)

        # get the best k
        best_k = list({i for i in test_score if test_score[i] == max(test_score.values())})

        # run assertions on best k
        self.assertIsNotNone(best_k)
        self.assertIsInstance(best_k, list)
        self.assertEqual(len(best_k), 1)
        self.assertEqual(best_k[0], 5)

        # define the folds
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 30, 1), 'leaf_size': np.arange(2, 10, 1), 'p': (1, 2),
                      'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # print out the best params
        print(f"best params-->{knn_cv.best_params_}")

        # print out the best score
        # print('Best Score - KNN:', knn_cv.best_score_)

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8869830442170634)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # predict based on our test data
        y_pred = knn.predict(the_f_df_test)

        print('Accuracy Score - KNN:', metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred))
        print('Average Precision - KNN:', metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred))
        print('F1 Score - KNN:', metrics.f1_score(the_t_var_test.values.ravel(), y_pred))
        print('Precision - KNN:', metrics.precision_score(the_t_var_test.values.ravel(), y_pred))
        print('Recall - KNN:', metrics.recall_score(the_t_var_test.values.ravel(), y_pred))
        print('ROC Score - KNN:', metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred))

        # run assertions
        self.assertEqual(metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred), 0.888)
        self.assertEqual(metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred), 0.6650469207739473)
        self.assertEqual(metrics.f1_score(the_t_var_test.values.ravel(), y_pred), 0.7690100430416069)
        self.assertEqual(metrics.precision_score(the_t_var_test.values.ravel(), y_pred), 0.8208269525267994)
        self.assertEqual(metrics.recall_score(the_t_var_test.values.ravel(), y_pred), 0.7233468286099866)
        self.assertEqual(metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred), 0.8342601059638499)

    # proof of concept using MCD for outlier detection, along with lower p-values for feature selection.
    # additionally changed the selectkbest scoring algorithm this time.
    def test_proof_of_concept_using_mcd_for_outlier_detection_lower_pvalues_remove_corr_features(self):
        # create Project_Assessment
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR,
                                report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list + ["Tenure"])
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)

        # get the base dataframe
        the_df = pa.analyzer.the_df

        # run assertions on what we think we have out of the gates
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns), 38)
        self.assertEqual(len(the_df), 9575)

        # create a variable encoder
        variable_encoder = Variable_Encoder(pa.analyzer.the_df)

        # encode the dataframe
        variable_encoder.encode_dataframe(drop_first=False)

        # get the variable columns
        the_variable_columns = variable_encoder.encoded_df.columns.to_list()

        # run assertions
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 53)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 52)
        self.assertFalse('Churn' in the_variable_columns)

        # retrieve the dataframe itself from the variable_encoder
        the_df = variable_encoder.get_encoded_dataframe()

        # get the column vector for the target variable
        the_target_var_series = the_df['Churn'].copy()
        the_features_df = the_df[the_variable_columns].copy()

        # run assertions to make sure nothing went goofy
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df.columns), 52)
        self.assertEqual(len(the_features_df), 9575)

        self.assertIsNotNone(the_target_var_series)
        self.assertIsInstance(the_target_var_series, Series)
        self.assertEqual(len(the_target_var_series), 9575)

        # ************************************ SELECT ************************************
        # Initialize the SelectKBest class and call fit_transform
        skbest = SelectKBest(score_func=f_classif, k='all')  # k= features

        # let the SelectKBest pick the features
        selected_features = skbest.fit_transform(the_features_df, the_target_var_series)

        # run assertions that we understand what we get back
        self.assertIsNotNone(selected_features)
        self.assertIsInstance(selected_features, numpy.ndarray)
        self.assertEqual(len(selected_features), 9575)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_features_df.columns,
                                 'p_value': skbest.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

        # run assertions on p_values
        self.assertIsNotNone(p_values)
        self.assertIsInstance(p_values, DataFrame)
        self.assertEqual(len(p_values), 14)
        self.assertEqual(len(p_values.columns), 2)

        # print(f"p_values\n{p_values}")

        # validate the features selected
        self.assertEqual(p_values['Feature'].iloc[0], "Bandwidth_GB_Year")
        self.assertEqual(p_values['Feature'].iloc[1], "MonthlyCharge")
        self.assertEqual(p_values['Feature'].iloc[2], "StreamingMovies")
        self.assertEqual(p_values['Feature'].iloc[3], "Contract_Month-to-month")
        self.assertEqual(p_values['Feature'].iloc[4], "StreamingTV")
        self.assertEqual(p_values['Feature'].iloc[5], "Contract_Two Year")
        self.assertEqual(p_values['Feature'].iloc[6], "Contract_One year")
        self.assertEqual(p_values['Feature'].iloc[7], "Multiple")
        self.assertEqual(p_values['Feature'].iloc[8], "InternetService_DSL")
        self.assertEqual(p_values['Feature'].iloc[9], "Techie")
        self.assertEqual(p_values['Feature'].iloc[10], "DeviceProtection")
        self.assertEqual(p_values['Feature'].iloc[11], "InternetService_Fiber Optic")
        self.assertEqual(p_values['Feature'].iloc[12], "OnlineBackup")
        self.assertEqual(p_values['Feature'].iloc[13], "InternetService_No response")

        # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
        # dataframe with just those columns.
        the_features_df = the_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_features_df
        self.assertIsNotNone(the_features_df)
        self.assertIsInstance(the_features_df, DataFrame)
        self.assertEqual(len(the_features_df), 9575)
        self.assertEqual(len(the_features_df.columns), 14)

        # ************************************ SPLIT ************************************
        # now we need to split our data
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(the_features_df,
                                                                                          the_target_var_series,
                                                                                          test_size=0.3,
                                                                                          random_state=12345)

        # run assertions on the_f_df_train
        self.assertIsNotNone(the_f_df_train)
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertEqual(len(the_f_df_train), 6702)
        self.assertEqual(len(the_f_df_train.columns), 14)
        self.assertEqual(the_f_df_train.shape, (6702, 14))

        # run assertions on the_f_df_test
        self.assertIsNotNone(the_f_df_test)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertEqual(len(the_f_df_test), 2873)
        self.assertEqual(len(the_f_df_test.columns), 14)
        self.assertEqual(the_f_df_test.shape, (2873, 14))

        # run assertions on the_t_var_train
        self.assertIsNotNone(the_t_var_train)
        self.assertIsInstance(the_t_var_train, Series)
        self.assertEqual(len(the_t_var_train), 6702)
        self.assertEqual(the_t_var_train.shape, (6702,))

        # run assertions on the_t_var_test
        self.assertIsNotNone(the_t_var_test)
        self.assertIsInstance(the_t_var_test, Series)
        self.assertEqual(len(the_t_var_test), 2873)
        self.assertEqual(the_t_var_test.shape, (2873,))

        # ************************************ SCALE ************************************
        # now that we've selected our features and split our data, we need to scale the data

        # create a StandardScaler
        the_scalar = StandardScaler()

        # scale the_f_df_train, the_f_df_test
        the_f_df_train = the_scalar.fit_transform(the_f_df_train)
        the_f_df_test = the_scalar.fit_transform(the_f_df_test)

        # scale the_t_var_train, the_t_var_test.
        # before we scaled the target data, we need to reshape it.
        the_t_var_train = the_t_var_train.to_frame().astype(int)
        the_t_var_test = the_t_var_test.to_frame().astype(int)

        # Now we have selected our features, split the dataset, scaled the dataset,
        # so we're ready to find our optimum K.  I'm first going to attempt to use

        train_score = {}
        test_score = {}
        n_neighbors = np.arange(2, 30, 1)

        # taken from https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7

        for neighbor in n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto')

            # if you don't call the_t_var_train.values.ravel(), you'll get an ugly warning.
            knn.fit(the_f_df_train, the_t_var_train.values.ravel())
            train_score[neighbor] = knn.score(the_f_df_train, the_t_var_train)
            test_score[neighbor] = knn.score(the_f_df_test, the_t_var_test)

        if exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"):
            os.remove(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")

        # generate a plot of results
        plt.clf()
        plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
        plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
        plt.xlabel("Number Of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN: Varying number of Neighbors")
        plt.legend()
        plt.xlim(0, 33)
        plt.ylim(0.60, 1.00)
        plt.grid()

        plt.savefig(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png")
        plt.close()

        # run assertions on the graph
        self.assertTrue(exists(self.OVERRIDE_PATH + "KNN_optimal_k_plot.png"))

        # run assertions on test_score
        self.assertIsNotNone(test_score)
        self.assertIsInstance(test_score, dict)
        self.assertEqual(max(test_score.values()), 0.8886181691611555)

        # get the best k
        best_k = list({i for i in test_score if test_score[i] == max(test_score.values())})

        # run assertions on best k
        self.assertIsNotNone(best_k)
        self.assertIsInstance(best_k, list)
        self.assertEqual(len(best_k), 1)
        self.assertEqual(best_k[0], 7)

        # define the folds
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)

        # define our parameters
        parameters = {'n_neighbors': np.arange(2, 30, 1), 'leaf_size': np.arange(2, 10, 1), 'p': (1, 2),
                      'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

        # define our classifier
        knn = KNeighborsClassifier(algorithm='auto')

        # instantiate our GridSearch
        knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

        # fit our ensemble
        knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

        # print out the best params
        print(f"best params-->{knn_cv.best_params_}")

        # print out the best score
        # print('Best Score - KNN:', knn_cv.best_score_)

        # run assertions
        self.assertEqual(knn_cv.best_score_, 0.8789921755873875)

        # create a new KNN with the best params
        knn = KNeighborsClassifier(**knn_cv.best_params_)

        # fit the data again
        knn.fit(the_f_df_train, the_t_var_train.values.ravel())

        # predict based on our test data
        y_pred = knn.predict(the_f_df_test)

        print('Accuracy Score - KNN:', metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred))
        print('Average Precision - KNN:', metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred))
        print('F1 Score - KNN:', metrics.f1_score(the_t_var_test.values.ravel(), y_pred))
        print('Precision - KNN:', metrics.precision_score(the_t_var_test.values.ravel(), y_pred))
        print('Recall - KNN:', metrics.recall_score(the_t_var_test.values.ravel(), y_pred))
        print('ROC Score - KNN:', metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred))

        # run assertions
        self.assertEqual(metrics.accuracy_score(the_t_var_test.values.ravel(), y_pred), 0.8927949878176122)
        self.assertEqual(metrics.average_precision_score(the_t_var_test.values.ravel(), y_pred), 0.6780057461723202)
        self.assertEqual(metrics.f1_score(the_t_var_test.values.ravel(), y_pred), 0.7852161785216178)
        self.assertEqual(metrics.precision_score(the_t_var_test.values.ravel(), y_pred), 0.8089080459770115)
        self.assertEqual(metrics.recall_score(the_t_var_test.values.ravel(), y_pred), 0.7628726287262872)
        self.assertEqual(metrics.roc_auc_score(the_t_var_test.values.ravel(), y_pred), 0.850288773379537)

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
        pa.clean_up_outliers(model_type=MT_KNN_CLASSIFICATION, max_p_value=0.001)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

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

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 38)
        self.assertFalse('Churn' in the_variable_columns)

        # get the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features = the_model.get_encoded_variables(the_target_column='Churn')

        # run assertions on the_encoded_features
        self.assertIsNotNone(the_encoded_features)
        self.assertIsInstance(the_encoded_features, list)
        self.assertEqual(len(the_encoded_features), 53)

        # invoke the method
        the_KNN_Model_Result = the_model.fit_a_model(the_target_column='Churn',
                                                     current_features=the_encoded_features,
                                                     model_type=MT_KNN_CLASSIFICATION)

        # run assertions on the_KNN_Model_Result
        self.assertIsNotNone(the_KNN_Model_Result)
        self.assertIsInstance(the_KNN_Model_Result, KNN_Model_Result)
        self.assertIsNotNone(the_KNN_Model_Result.get_model())
        self.assertIsInstance(the_KNN_Model_Result.get_model(), KNeighborsClassifier)

        self.assertEqual(the_KNN_Model_Result.get_accuracy(), 0.8973913043478261)
        self.assertEqual(the_KNN_Model_Result.get_avg_precision(), 0.6909695648722435)
        self.assertEqual(the_KNN_Model_Result.get_f1_score(), 0.7923997185080929)
        self.assertEqual(the_KNN_Model_Result.get_precision(), 0.8279411764705882)
        self.assertEqual(the_KNN_Model_Result.get_recall(), 0.7597840755735492)
        self.assertEqual(the_KNN_Model_Result.get_roc_score(), 0.8524787294456312)

    # test proof of concept for making sure only discrete and continuous features are scaled
    def test_fit_a_model_order(self):
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
        pa.calculate_internal_statistics(the_level=0.5)

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

        # get the variable columns
        the_variable_columns = the_df.columns.to_list()

        # run assertions on non-encoded features.
        self.assertIsNotNone(the_variable_columns)
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 39)
        self.assertTrue('Churn' in the_variable_columns)

        # remove the target variable
        the_variable_columns.remove('Churn')

        # run assertions
        self.assertIsInstance(the_variable_columns, list)
        self.assertEqual(len(the_variable_columns), 38)
        self.assertFalse('Churn' in the_variable_columns)

        # get the encoded variables, this is what has to be passed to 'current_features' argument.
        the_encoded_features = the_model.get_encoded_variables(the_target_column='Churn')

        # run assertions on the_encoded_features
        self.assertIsNotNone(the_encoded_features)
        self.assertIsInstance(the_encoded_features, list)
        self.assertEqual(len(the_encoded_features), 53)

        # validate the elements of the_encoded_features
        self.assertEqual(the_encoded_features[0], 'Population')
        self.assertEqual(the_encoded_features[1], 'TimeZone')
        self.assertEqual(the_encoded_features[2], 'Children')
        self.assertEqual(the_encoded_features[3], 'Age')
        self.assertEqual(the_encoded_features[4], 'Income')
        self.assertEqual(the_encoded_features[5], 'Outage_sec_perweek')
        self.assertEqual(the_encoded_features[6], 'Email')
        self.assertEqual(the_encoded_features[7], 'Contacts')
        self.assertEqual(the_encoded_features[8], 'Yearly_equip_failure')
        self.assertEqual(the_encoded_features[9], 'Techie')
        self.assertEqual(the_encoded_features[10], 'Port_modem')
        self.assertEqual(the_encoded_features[11], 'Tablet')
        self.assertEqual(the_encoded_features[12], 'Phone')
        self.assertEqual(the_encoded_features[13], 'Multiple')
        self.assertEqual(the_encoded_features[14], 'OnlineSecurity')
        self.assertEqual(the_encoded_features[15], 'OnlineBackup')
        self.assertEqual(the_encoded_features[16], 'DeviceProtection')
        self.assertEqual(the_encoded_features[17], 'TechSupport')
        self.assertEqual(the_encoded_features[18], 'StreamingTV')
        self.assertEqual(the_encoded_features[19], 'StreamingMovies')
        self.assertEqual(the_encoded_features[20], 'PaperlessBilling')
        self.assertEqual(the_encoded_features[21], 'Tenure')
        self.assertEqual(the_encoded_features[22], 'MonthlyCharge')
        self.assertEqual(the_encoded_features[23], 'Bandwidth_GB_Year')
        self.assertEqual(the_encoded_features[24], 'Timely_Response')
        self.assertEqual(the_encoded_features[25], 'Timely_Fixes')
        self.assertEqual(the_encoded_features[26], 'Timely_Replacements')
        self.assertEqual(the_encoded_features[27], 'Reliability')
        self.assertEqual(the_encoded_features[28], 'Options')
        self.assertEqual(the_encoded_features[29], 'Respectful_Response')
        self.assertEqual(the_encoded_features[30], 'Courteous_Exchange')
        self.assertEqual(the_encoded_features[31], 'Active_Listening')
        self.assertEqual(the_encoded_features[32], 'Area_Rural')
        self.assertEqual(the_encoded_features[33], 'Area_Suburban')
        self.assertEqual(the_encoded_features[34], 'Area_Urban')
        self.assertEqual(the_encoded_features[35], 'Marital_Divorced')
        self.assertEqual(the_encoded_features[36], 'Marital_Married')
        self.assertEqual(the_encoded_features[37], 'Marital_Never Married')
        self.assertEqual(the_encoded_features[38], 'Marital_Separated')
        self.assertEqual(the_encoded_features[39], 'Marital_Widowed')
        self.assertEqual(the_encoded_features[40], 'Gender_Female')
        self.assertEqual(the_encoded_features[41], 'Gender_Male')
        self.assertEqual(the_encoded_features[42], 'Gender_Nonbinary')
        self.assertEqual(the_encoded_features[43], 'Contract_Month-to-month')
        self.assertEqual(the_encoded_features[44], 'Contract_One year')
        self.assertEqual(the_encoded_features[45], 'Contract_Two Year')
        self.assertEqual(the_encoded_features[46], 'InternetService_DSL')
        self.assertEqual(the_encoded_features[47], 'InternetService_Fiber Optic')
        self.assertEqual(the_encoded_features[48], 'InternetService_No response')
        self.assertEqual(the_encoded_features[49], 'PaymentMethod_Bank Transfer(automatic)')
        self.assertEqual(the_encoded_features[50], 'PaymentMethod_Credit Card (automatic)')
        self.assertEqual(the_encoded_features[51], 'PaymentMethod_Electronic Check')
        self.assertEqual(the_encoded_features[52], 'PaymentMethod_Mailed Check')
        self.assertFalse('Churn' in the_encoded_features)

        # at this point, we know what the columns are on the_model, which is an instance of KNN_Model
        # and we know that the target variable 'Churn' is not on the_model.

        # set the target
        the_target_df = the_model.encoded_df['Churn'].astype(int)

        # define the complete set of current features, and cast to type int.
        the_current_features_df = the_model.encoded_df[the_encoded_features]  #.astype(int)

        # Initialize the SelectKBest class to identify relevance of all features
        sk_best = SelectKBest(score_func=f_classif, k='all')

        # let the SelectKBest calculate the p-values for all features.  We will not do anything with the
        # variable "selected_features".
        selected_features = sk_best.fit_transform(the_current_features_df, the_target_df)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': the_current_features_df.columns,
                                 'p_value': sk_best.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > 0.001
        p_values = p_values[p_values['p_value'] < .001]

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
        # dataframe with just those columns.  Only include the features from the_current_features_df that have a
        # value less than the required p-value.
        the_current_features_df = the_current_features_df[p_values['Feature'].tolist()].copy()

        # run assertions on the_current_features_df
        self.assertEqual(the_current_features_df.columns[0], "Tenure")
        self.assertEqual(the_model.encoded_df["Tenure"].dtype, float)

        self.assertEqual(the_current_features_df.columns[1], "Bandwidth_GB_Year")
        self.assertEqual(the_model.encoded_df["Bandwidth_GB_Year"].dtype, float)

        self.assertEqual(the_current_features_df.columns[2], "MonthlyCharge")
        self.assertEqual(the_model.encoded_df["MonthlyCharge"].dtype, float)

        self.assertEqual(the_current_features_df.columns[3], "StreamingMovies")
        self.assertEqual(the_model.encoded_df["StreamingMovies"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[4], "Contract_Month-to-month")
        self.assertEqual(the_model.encoded_df["Contract_Month-to-month"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[5], "StreamingTV")
        self.assertEqual(the_model.encoded_df["StreamingTV"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[6], "Contract_Two Year")
        self.assertEqual(the_model.encoded_df["Contract_Two Year"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[7], "Contract_One year")
        self.assertEqual(the_model.encoded_df["Contract_One year"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[8], "Multiple")
        self.assertEqual(the_model.encoded_df["Multiple"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[9], "InternetService_DSL")
        self.assertEqual(the_model.encoded_df["InternetService_DSL"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[10], "Techie")
        self.assertEqual(the_model.encoded_df["Techie"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[11], "InternetService_Fiber Optic")
        self.assertEqual(the_model.encoded_df["InternetService_Fiber Optic"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[12], "DeviceProtection")
        self.assertEqual(the_model.encoded_df["DeviceProtection"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[13], "OnlineBackup")
        self.assertEqual(the_model.encoded_df["OnlineBackup"].dtype, bool)

        self.assertEqual(the_current_features_df.columns[14], "InternetService_No response")
        self.assertEqual(the_model.encoded_df["InternetService_No response"].dtype, bool)

        # now we know the name of what we actually have in each column, and the ORIGINAL type of each column

        # now we need to split our data.  The the_current_features_df should only contain our pruned list of
        # features.  Providing a variable name mapping
        # x_train--> the_f_df_train = the training features (not including the target)
        # x_test--> the_f_df_test = the test features (not including the target)
        # y_train--> the_t_var_train = the training target variable
        # y_test-->the_t_var_test = the test target variable
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(
            the_current_features_df,
            the_target_df,
            test_size=0.3,
            random_state=12345)

        # run assertions on the resulting dataframes
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertTupleEqual(the_f_df_train.shape, (6707, 15))
        self.assertTupleEqual(the_f_df_test.shape, (2875, 15))

        # run assertions on the_f_df_train, to make sure we know what we have.
        self.assertEqual(the_f_df_train.columns[0], "Tenure")
        self.assertEqual(the_f_df_train["Tenure"].dtype, float)
        self.assertEqual(the_f_df_train.columns[1], "Bandwidth_GB_Year")
        self.assertEqual(the_f_df_train["Bandwidth_GB_Year"].dtype, float)
        self.assertEqual(the_f_df_train.columns[2], "MonthlyCharge")
        self.assertEqual(the_f_df_train["MonthlyCharge"].dtype, float)
        self.assertEqual(the_f_df_train.columns[3], "StreamingMovies")
        self.assertEqual(the_f_df_train["StreamingMovies"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[4], "Contract_Month-to-month")
        self.assertEqual(the_f_df_train["Contract_Month-to-month"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[5], "StreamingTV")
        self.assertEqual(the_f_df_train["StreamingTV"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[6], "Contract_Two Year")
        self.assertEqual(the_f_df_train["Contract_Two Year"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[7], "Contract_One year")
        self.assertEqual(the_f_df_train["Contract_One year"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[8], "Multiple")
        self.assertEqual(the_f_df_train["Multiple"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[9], "InternetService_DSL")
        self.assertEqual(the_f_df_train["InternetService_DSL"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[10], "Techie")
        self.assertEqual(the_f_df_train["Techie"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[11], "InternetService_Fiber Optic")
        self.assertEqual(the_f_df_train["InternetService_Fiber Optic"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[12], "DeviceProtection")
        self.assertEqual(the_f_df_train["DeviceProtection"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[13], "OnlineBackup")
        self.assertEqual(the_f_df_train["OnlineBackup"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[14], "InternetService_No response")
        self.assertEqual(the_f_df_train["InternetService_No response"].dtype, bool)

        # run assertions on the_f_df_test, to make sure we know what we have.
        self.assertEqual(the_f_df_test.columns[0], "Tenure")
        self.assertEqual(the_f_df_test["Tenure"].dtype, float)
        self.assertEqual(the_f_df_test.columns[1], "Bandwidth_GB_Year")
        self.assertEqual(the_f_df_test["Bandwidth_GB_Year"].dtype, float)
        self.assertEqual(the_f_df_test.columns[2], "MonthlyCharge")
        self.assertEqual(the_f_df_test["MonthlyCharge"].dtype, float)
        self.assertEqual(the_f_df_test.columns[3], "StreamingMovies")
        self.assertEqual(the_f_df_test["StreamingMovies"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[4], "Contract_Month-to-month")
        self.assertEqual(the_f_df_test["Contract_Month-to-month"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[5], "StreamingTV")
        self.assertEqual(the_f_df_test["StreamingTV"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[6], "Contract_Two Year")
        self.assertEqual(the_f_df_test["Contract_Two Year"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[7], "Contract_One year")
        self.assertEqual(the_f_df_test["Contract_One year"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[8], "Multiple")
        self.assertEqual(the_f_df_test["Multiple"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[9], "InternetService_DSL")
        self.assertEqual(the_f_df_test["InternetService_DSL"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[10], "Techie")
        self.assertEqual(the_f_df_test["Techie"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[11], "InternetService_Fiber Optic")
        self.assertEqual(the_f_df_test["InternetService_Fiber Optic"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[12], "DeviceProtection")
        self.assertEqual(the_f_df_test["DeviceProtection"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[13], "OnlineBackup")
        self.assertEqual(the_f_df_test["OnlineBackup"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[14], "InternetService_No response")
        self.assertEqual(the_f_df_test["InternetService_No response"].dtype, bool)

        # make a copy of the the_f_df_test
        the_f_df_test_orig = the_f_df_test.copy()

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we can scale, only INT64 and FLOAT.
        list_of_features_to_scale = the_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
                                                                                     FLOAT64_COLUMN_KEY])

        # run assertions on contents of list_of_features_to_scale
        self.assertIsNotNone(list_of_features_to_scale)
        self.assertIsInstance(list_of_features_to_scale, list)
        self.assertEqual(len(list_of_features_to_scale), 20)
        self.assertEqual(list_of_features_to_scale[0], 'Population')
        self.assertEqual(list_of_features_to_scale[1], 'TimeZone')
        self.assertEqual(list_of_features_to_scale[2], 'Children')
        self.assertEqual(list_of_features_to_scale[3], 'Age')
        self.assertEqual(list_of_features_to_scale[4], 'Email')
        self.assertEqual(list_of_features_to_scale[5], 'Contacts')
        self.assertEqual(list_of_features_to_scale[6], 'Yearly_equip_failure')
        self.assertEqual(list_of_features_to_scale[7], 'Timely_Response')
        self.assertEqual(list_of_features_to_scale[8], 'Timely_Fixes')
        self.assertEqual(list_of_features_to_scale[9], 'Timely_Replacements')
        self.assertEqual(list_of_features_to_scale[10], 'Reliability')
        self.assertEqual(list_of_features_to_scale[11], 'Options')
        self.assertEqual(list_of_features_to_scale[12], 'Respectful_Response')
        self.assertEqual(list_of_features_to_scale[13], 'Courteous_Exchange')
        self.assertEqual(list_of_features_to_scale[14], 'Active_Listening')
        self.assertEqual(list_of_features_to_scale[15], 'Income')
        self.assertEqual(list_of_features_to_scale[16], 'Outage_sec_perweek')
        self.assertEqual(list_of_features_to_scale[17], 'Tenure')
        self.assertEqual(list_of_features_to_scale[18], 'MonthlyCharge')
        self.assertEqual(list_of_features_to_scale[19], 'Bandwidth_GB_Year')

        # now we need to get the features we can scale
        the_f_df_train_to_scale = [i for i in the_f_df_train.columns.to_list()
                                   if i in list_of_features_to_scale]

        # run assertions on the_f_df_train_to_scale
        self.assertIsInstance(the_f_df_train_to_scale, list)
        self.assertEqual(len(the_f_df_train_to_scale), 3)
        self.assertEqual(the_f_df_train_to_scale[0], 'Tenure')
        self.assertEqual(str(the_model.encoded_df['Tenure'].dtype), 'float64')
        self.assertEqual(the_f_df_train_to_scale[1], 'Bandwidth_GB_Year')
        self.assertEqual(str(the_model.encoded_df['Bandwidth_GB_Year'].dtype), 'float64')
        self.assertEqual(the_f_df_train_to_scale[2], 'MonthlyCharge')
        self.assertEqual(str(the_model.encoded_df['MonthlyCharge'].dtype), 'float64')

        field_index_map = {}

        # I need to capture what column indexes in the_f_df_train the fields in the list the_f_df_train_to_scale
        # are located.
        for the_field in the_f_df_train_to_scale:
            # populate field_index_map
            field_index_map[the_field] = the_f_df_train.columns.get_loc(the_field)

            # run assertion
            self.assertIsInstance(field_index_map[the_field], int)

        # now we need to scale the_f_df_train, the_f_df_test only using the columns present in the list
        # the_f_df_train_to_scale
        # Note: The datatype changes from a DataFrame to a numpy.array after the call to fit_tranform().
        the_scaled_f_train_df = the_scalar.fit_transform(the_f_df_train[the_f_df_train_to_scale])
        the_scaled_f_test_df = the_scalar.fit_transform(the_f_df_test[the_f_df_train_to_scale])

        # run assertions on the type for the_scaled_f_train_df and the_scaled_f_test_df
        self.assertIsInstance(the_scaled_f_train_df, numpy.ndarray)
        self.assertIsInstance(the_scaled_f_test_df, numpy.ndarray)
        self.assertEqual(len(the_scaled_f_train_df), 6707)
        self.assertTupleEqual(the_scaled_f_train_df.shape, (6707, 3))
        self.assertEqual(len(the_scaled_f_test_df), 2875)
        self.assertTupleEqual(the_scaled_f_test_df.shape, (2875, 3))

        # now we need to cast the_scaled_f_train_df and the_scaled_f_test_df back to DataFrames
        the_scaled_f_train_df = pd.DataFrame(data=the_scaled_f_train_df, columns=the_f_df_train_to_scale)
        the_scaled_f_test_df = pd.DataFrame(data=the_scaled_f_test_df, columns=the_f_df_train_to_scale)

        # run assertions
        self.assertIsInstance(the_scaled_f_train_df, DataFrame)
        self.assertTupleEqual(the_scaled_f_train_df.shape, (6707, 3))
        self.assertIsInstance(the_scaled_f_test_df, DataFrame)
        self.assertTupleEqual(the_scaled_f_test_df.shape, (2875, 3))

        for the_field in the_f_df_train_to_scale:
            # update the original dataframe
            the_f_df_train[the_field] = the_scaled_f_train_df[the_field].values
            the_f_df_test[the_field] = the_scaled_f_test_df[the_field].values

        # run validations on the_f_df_train and the_f_df_test
        self.assertIsInstance(the_f_df_train, DataFrame)
        self.assertIsInstance(the_f_df_test, DataFrame)
        self.assertTupleEqual(the_f_df_train.shape, (6707, 15))
        self.assertTupleEqual(the_f_df_test.shape, (2875, 15))

        # run validations on the_f_df_train
        self.assertEqual(the_f_df_train.columns[0], "Tenure")
        self.assertEqual(the_f_df_train["Tenure"].dtype, float)
        self.assertEqual(the_f_df_train.columns[1], "Bandwidth_GB_Year")
        self.assertEqual(the_f_df_train["Bandwidth_GB_Year"].dtype, float)
        self.assertEqual(the_f_df_train.columns[2], "MonthlyCharge")
        self.assertEqual(the_f_df_train["MonthlyCharge"].dtype, float)
        self.assertEqual(the_f_df_train.columns[3], "StreamingMovies")
        self.assertEqual(the_f_df_train["StreamingMovies"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[4], "Contract_Month-to-month")
        self.assertEqual(the_f_df_train["Contract_Month-to-month"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[5], "StreamingTV")
        self.assertEqual(the_f_df_train["StreamingTV"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[6], "Contract_Two Year")
        self.assertEqual(the_f_df_train["Contract_Two Year"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[7], "Contract_One year")
        self.assertEqual(the_f_df_train["Contract_One year"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[8], "Multiple")
        self.assertEqual(the_f_df_train["Multiple"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[9], "InternetService_DSL")
        self.assertEqual(the_f_df_train["InternetService_DSL"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[10], "Techie")
        self.assertEqual(the_f_df_train["Techie"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[11], "InternetService_Fiber Optic")
        self.assertEqual(the_f_df_train["InternetService_Fiber Optic"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[12], "DeviceProtection")
        self.assertEqual(the_f_df_train["DeviceProtection"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[13], "OnlineBackup")
        self.assertEqual(the_f_df_train["OnlineBackup"].dtype, bool)
        self.assertEqual(the_f_df_train.columns[14], "InternetService_No response")
        self.assertEqual(the_f_df_train["InternetService_No response"].dtype, bool)

        # run assertions on the_f_df_test, to make sure we know what we have.
        self.assertEqual(the_f_df_test.columns[0], "Tenure")
        self.assertEqual(the_f_df_test["Tenure"].dtype, float)
        self.assertEqual(the_f_df_test.columns[1], "Bandwidth_GB_Year")
        self.assertEqual(the_f_df_test["Bandwidth_GB_Year"].dtype, float)
        self.assertEqual(the_f_df_test.columns[2], "MonthlyCharge")
        self.assertEqual(the_f_df_test["MonthlyCharge"].dtype, float)
        self.assertEqual(the_f_df_test.columns[3], "StreamingMovies")
        self.assertEqual(the_f_df_test["StreamingMovies"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[4], "Contract_Month-to-month")
        self.assertEqual(the_f_df_test["Contract_Month-to-month"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[5], "StreamingTV")
        self.assertEqual(the_f_df_test["StreamingTV"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[6], "Contract_Two Year")
        self.assertEqual(the_f_df_test["Contract_Two Year"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[7], "Contract_One year")
        self.assertEqual(the_f_df_test["Contract_One year"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[8], "Multiple")
        self.assertEqual(the_f_df_test["Multiple"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[9], "InternetService_DSL")
        self.assertEqual(the_f_df_test["InternetService_DSL"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[10], "Techie")
        self.assertEqual(the_f_df_test["Techie"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[11], "InternetService_Fiber Optic")
        self.assertEqual(the_f_df_test["InternetService_Fiber Optic"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[12], "DeviceProtection")
        self.assertEqual(the_f_df_test["DeviceProtection"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[13], "OnlineBackup")
        self.assertEqual(the_f_df_test["OnlineBackup"].dtype, bool)
        self.assertEqual(the_f_df_test.columns[14], "InternetService_No response")
        self.assertEqual(the_f_df_test["InternetService_No response"].dtype, bool)

