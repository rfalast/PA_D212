import logging
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from model.BaseModel import BaseModel
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.Variable_Encoder import Variable_Encoder
from model.analysis.models import ModelResultBase
from model.analysis.models.KNN_Model_Result import KNN_Model_Result
from model.analysis.models.Linear_Model_Result import Linear_Model_Result
from model.analysis.models.Logistic_Model_Result import Logistic_Model_Result
from model.analysis.models.Random_Forest_Model_Result import Random_Forest_Model_Result
from model.constants.BasicConstants import MT_OPTIONS, MT_LOGISTIC_REGRESSION, MT_KNN_CLASSIFICATION, \
    MT_RF_REGRESSION
from model.constants.DatasetConstants import INT64_COLUMN_KEY, FLOAT64_COLUMN_KEY
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_FINAL_MODEL, LM_STEP, LF_ELIM_REASON_VIF, \
    LF_ELIM_REASON_P_VALUE, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X_ORIGINAL
from util.Model_Result_Populator import Model_Result_Populator


class ModelBase(BaseModel):
    # init() method
    def __init__(self, dataset_analyzer: DatasetAnalyzer):
        # call superclass
        super().__init__()

        # perform validation
        if not self.is_valid(dataset_analyzer, DatasetAnalyzer):
            raise ValueError("dataset_analyzer is None or incorrect type.")

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # define internal variables
        self.dataset_analyzer = dataset_analyzer
        self.model_storage = {}
        self.variable_encoder = Variable_Encoder(dataset_analyzer.the_df)
        self.encoded_df = self.variable_encoder.get_encoded_dataframe()
        self.the_result = None
        self.model_type = None

    # getter method for the_result property.
    def get_the_result(self):
        return self.the_result

    # fit a model to specific arguments.  This method will not attempt to
    def fit_a_model(self, the_target_column: str, current_features: list, model_type: str) -> ModelResultBase:
        """
        Fit a model to a target variable (the_target_column) based on the features.
        The result is stored internally at self.the_log_lm_result
        defined in current_features.
        :param the_target_column:
        :param current_features:
        :param model_type: MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION, MT_KNN_CLASSIFICATION, or MT_RF_REGRESSION
        :return: ModelResultBase or extending class.
        """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise SyntaxError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise ValueError("the_target_column argument is not in dataframe.")
        elif not isinstance(current_features, list):
            raise SyntaxError("current_features argument is None or incorrect type.")

        # make sure everything in the_variable_columns is in self.encoded_df
        for the_column in current_features:
            if the_column not in self.encoded_df:
                raise ValueError("an element in the_variable_columns argument is not in dataframe.")

        # continue validation.
        if model_type not in MT_OPTIONS:
            raise ValueError("model_type is not a valid value.")

        # first, make sure the_target_column is not in the_variable_columns
        if the_target_column in current_features:
            # log that we are removing the_target_column from current_features
            self.logger.info(f"removing [{the_target_column}] from current_features{current_features}")

            # remove the_target_column from the_variable_columns
            current_features.remove(the_target_column)

        # variable declaration
        the_target_df = None
        the_current_features_df = None
        argument_dict = None
        mrp = Model_Result_Populator()

        # check if we have a logistic regression
        if model_type == MT_LOGISTIC_REGRESSION:
            # set the target
            the_target_df = self.encoded_df[the_target_column].astype(int)

            # set the current features
            the_current_features_df = self.encoded_df[current_features].astype(int)

            # get the constant from the_current_features_df
            x_con = sm.add_constant(the_current_features_df)

            # get the Logit
            logistic_regression = sm.Logit(the_target_df, x_con)

            # get the fitted model
            fitted_model = logistic_regression.fit()

            # create the argument_dict
            mrp.populate_storage(the_key="the_model", the_item=fitted_model)
            mrp.populate_storage(the_key="the_target_variable", the_item=the_target_column)
            mrp.populate_storage(the_key="the_variables_list", the_item=current_features)
            mrp.populate_storage(the_key="the_df", the_item=the_current_features_df)

            # create the result, and return it.
            self.the_result = Logistic_Model_Result(mrp.get_storage())

        # check if we are a KNN Model
        elif model_type == MT_KNN_CLASSIFICATION:
            # set the target
            the_target_df = self.encoded_df[the_target_column].astype(int)

            # define the complete set of current features, and cast to type int.
            the_current_features_df = self.encoded_df[current_features].astype(int)

            # eliminate all features with a p_value > 0.001
            p_values = self.get_selected_features(current_features_df=the_current_features_df,
                                                  the_target=the_target_df,
                                                  p_val_sig=0.001,
                                                  model_type=MT_KNN_CLASSIFICATION)

            # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
            # dataframe with just those columns.  Only include the features from the_current_features_df that have a
            # value less than the required p-value.
            the_current_features_df = the_current_features_df[p_values['Feature'].tolist()].copy()

            # make sure to update the_variables_list
            current_features = the_current_features_df.columns.to_list()

            # split and scale
            split_results = self.split_and_scale_features(current_features_df=the_current_features_df,
                                                          the_target=the_target_df,
                                                          test_size=0.3)

            # define variables
            the_f_df_train = split_results[X_TRAIN]
            the_f_df_test = split_results[X_TEST]
            the_t_var_train = split_results[Y_TRAIN]
            the_t_var_test = split_results[Y_TEST]
            the_f_df_test_orig = split_results[X_ORIGINAL]

            # reshape and recast the training and test target variables as int.
            the_t_var_train = the_t_var_train.to_frame().astype(int)
            the_t_var_test = the_t_var_test.to_frame().astype(int)

            # define the folds.  Set n_splits to 5 in final model.
            kf = KFold(n_splits=5, shuffle=True, random_state=12345)

            # define our initial search parameters for the classification.
            # n_neighbors of (2, 25, 1) and leaf_size of (2, 5, 1) for final model
            parameters = {'n_neighbors': np.arange(2, 25, 1), 'leaf_size': np.arange(2, 5, 1), 'p': (1, 2),
                          'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

            # define our classifier
            knn = KNeighborsClassifier(algorithm='auto')

            # fit the data before handing it over to the GridSearchCV.  You need to call .ravel() to get
            # an array, or else you'll get a warning that a column-vector y was passed.
            knn.fit(X=the_t_var_train, y=the_t_var_train.values.ravel())

            # instantiate our GridSearch
            knn_cv = GridSearchCV(knn, param_grid=parameters, cv=kf, verbose=1)

            # fit our ensemble model.
            knn_cv.fit(the_f_df_train, the_t_var_train.values.ravel())

            # create a new KNN with the best params
            knn = KNeighborsClassifier(**knn_cv.best_params_)

            # fit the data again
            knn.fit(the_f_df_train, the_t_var_train.values.ravel())

            # create the argument_dict
            argument_dict = dict()

            # populate argument_dict
            argument_dict['the_model'] = knn
            argument_dict['the_target_variable'] = the_target_column
            argument_dict['the_variables_list'] = current_features
            argument_dict['the_f_df_train'] = the_f_df_train
            argument_dict['the_f_df_test'] = the_f_df_test
            argument_dict['the_t_var_train'] = the_t_var_train
            argument_dict['the_t_var_test'] = the_t_var_test
            argument_dict['the_encoded_df'] = the_f_df_test_orig
            argument_dict['the_p_values'] = p_values
            argument_dict['gridsearch'] = knn_cv
            argument_dict['prepared_data'] = the_current_features_df
            argument_dict['cleaned_data'] = self.encoded_df
            argument_dict['the_df'] = self.encoded_df

            # create the result
            self.the_result = KNN_Model_Result(argument_dict=argument_dict)

        # check if we are a Random Forest Model
        elif model_type == MT_RF_REGRESSION:
            # log what we're doing
            self.logger.debug("performing Random Forest regression.")

            # initial variable declaration
            the_target_df = self.encoded_df[the_target_column]
            the_current_features_df = self.encoded_df[current_features]

            # eliminate all features with a p_value > 0.001
            p_values = self.get_selected_features(current_features_df=the_current_features_df,
                                                  the_target=the_target_df,
                                                  p_val_sig=0.001,
                                                  model_type=MT_RF_REGRESSION)

            # Thus, we now have a list of features that we need to use going forward, so we need to assemble a new
            # dataframe with just those columns.  Only include the features from the_current_features_df that have a
            # value less than the required p-value.
            the_current_features_df = the_current_features_df[p_values['Feature'].tolist()].copy()

            # make sure to update the_variables_list to match the reduced number of features based on the p value
            # significance.
            current_features = the_current_features_df.columns.to_list()

            # split and scale
            split_results = self.split_and_scale_features(current_features_df=the_current_features_df,
                                                          the_target=the_target_df,
                                                          test_size=0.2)

            # define variables
            the_f_df_train = split_results[X_TRAIN]
            the_f_df_test = split_results[X_TEST]
            the_t_var_train = split_results[Y_TRAIN]
            the_t_var_test = split_results[Y_TEST]
            the_f_df_test_orig = split_results[X_ORIGINAL]

            # create a base RandomForestRegressor to compare with.
            rfr = RandomForestRegressor(random_state=12345, oob_score=True)

            # call fit
            rfr.fit(the_f_df_train, the_t_var_train)

            # Create a parameter grid for GridSearchCV
            param_grid = {'n_estimators': [20, 100, 200, 300, 500, 1000], 'max_features': ['sqrt', 'log2'],
                          'max_depth': [None, 10, 30, 50], 'min_samples_split': [2, 10, 20],
                          'min_samples_leaf': [1, 2, 5, 10], 'bootstrap': [True]}

            # define the folds.  Set n_splits to 5 in final model.
            kf = KFold(n_splits=5, shuffle=True, random_state=12345)

            # Removes warnings in the current job
            warnings.filterwarnings("ignore")

            # Removes warnings in the spawned jobs
            os.environ['PYTHONWARNINGS'] = 'ignore'

            # # create a grid search
            grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0,
                                       scoring='neg_mean_squared_error')

            # Fit the grid search to the data
            grid_search.fit(X=the_f_df_train, y=the_t_var_train.values.ravel())

            # create the argument_dict
            argument_dict = dict()

            # populate the argument dict
            argument_dict['the_model'] = rfr
            argument_dict['the_target_variable'] = the_target_column
            argument_dict['the_variables_list'] = current_features
            argument_dict['the_f_df_train'] = the_f_df_train
            argument_dict['the_f_df_test'] = the_f_df_test
            argument_dict['the_t_var_train'] = the_t_var_train
            argument_dict['the_t_var_test'] = the_t_var_test
            argument_dict['the_encoded_df'] = the_f_df_test_orig
            argument_dict['the_p_values'] = p_values
            argument_dict['gridsearch'] = grid_search
            argument_dict['prepared_data'] = the_current_features_df
            argument_dict['cleaned_data'] = self.encoded_df

            # invoke the method
            self.the_result = Random_Forest_Model_Result(argument_dict=argument_dict)

        # we have a linear regression
        else:
            # variable declaration
            the_target_df = self.encoded_df[the_target_column].astype(float)
            the_current_features_df = self.encoded_df[current_features].astype(float)

            # add_constant to the_current_features_df
            the_current_features_df = sm.add_constant(the_current_features_df)

            # # create a model
            model = sm.OLS(the_target_df, the_current_features_df).fit()

            # create the argument_dict
            argument_dict = dict()

            # populate the argument_dict
            argument_dict['the_model'] = model
            argument_dict['the_target_variable'] = the_target_column
            argument_dict['the_variables_list'] = current_features
            argument_dict['the_df'] = the_current_features_df

            # create the result, and return it.
            self.the_result = Linear_Model_Result(argument_dict=argument_dict)

        # return the result
        return self.the_result

    # solve for target
    def solve_for_target(self, the_target_column, model_type, max_p_value=1.0, the_max_vif=10, suppress_console=True):
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise ValueError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise SyntaxError("the_target_column argument is not in dataframe.")
        elif model_type not in MT_OPTIONS:
            raise ValueError("model_type is not a valid value.")
        # validate that the p-value is a float <=1.00
        elif not isinstance(max_p_value, float):
            raise SyntaxError("max_p_value was None or incorrect type.")
        elif max_p_value <= 0 or max_p_value > 1.0:
            raise ValueError("max_p_value was greater than 1.0")
        # validate that the max_vif is a float
        elif not isinstance(the_max_vif, float):
            raise SyntaxError("max_vif was None or incorrect type.")
        elif the_max_vif < 5.0:
            raise ValueError("max_vif must be > 5.0")

        # log that we've been called
        self.logger.debug(f"A request to solve for [{the_target_column}] has been made.")

        # call reduce_a_model
        self.reduce_a_model(the_target_column=the_target_column,
                            current_features=self.encoded_df.columns.to_list(),
                            model_type=model_type,
                            max_p_value=max_p_value,
                            max_vif=the_max_vif,
                            suppress_console=suppress_console)

    # reduce a model
    def reduce_a_model(self, the_target_column, current_features, model_type,
                       max_p_value=1.0, max_vif=10.0, suppress_console=True) -> ModelResultBase:
        """
        Generate a linear model by starting with the entire dataset and removing one variable at a time
        based on the p_value.
        :param the_target_column: the target column
        :param current_features: list of current features
        :param model_type: MT_LOGISTIC_REGRESSION or MT_LINEAR_REGRESSION
        :param max_p_value: maximum p-value
        :param max_vif: maximum vif value
        :param suppress_console: True if you don't want console output
        :return: ModelResultBase
        """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise ValueError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise SyntaxError("the_target_column argument is not in dataframe.")
        elif not isinstance(current_features, list):
            raise ValueError("current_features argument is None or incorrect type.")

        # make sure everything in the_variable_columns is in self.encoded_df
        for the_column in current_features:
            if the_column not in self.encoded_df:
                raise SyntaxError("an element in the_variable_columns argument is not in dataframe.")

        # continue validation.
        if model_type not in MT_OPTIONS:
            raise ValueError("model_type is not a valid value.")
        # validate that max_p_value is a float <=1.00
        elif not isinstance(max_p_value, float):
            raise SyntaxError("max_p_value was None or incorrect type.")
        elif max_p_value < 0 or max_p_value > 1.0:
            raise ValueError("max_p_value must be in range (0,1)")
        # validate that the max_vif is a float
        elif not isinstance(max_vif, float):
            raise SyntaxError("max_vif was None or incorrect type.")
        elif max_vif < 5.0:
            raise ValueError("max_vif must be > 5.0")

        # first, make sure the_target_column is not in the_variable_columns
        if the_target_column in current_features:
            # log that we are removing the_target_column from current_features
            self.logger.info(f"removing [{the_target_column}] from current_features{current_features}")

            # remove the_target_column from the_variable_columns
            current_features.remove(the_target_column)

        # log that we've been called
        self.logger.debug(f"A request to reduce a model with max_p_value[{max_p_value}] has been made.")

        # variable declaration
        the_model_result = None
        the_feature = None
        loop_again = True
        current_features = current_features.copy()
        the_step = 1

        # the first step in our reduction technique is to ensure multi-collinearity assumption is not violated.
        # Thus, remove features until all remaining features have a VIF below max_vif
        while loop_again:

            assert the_feature not in current_features

            # get the_model_result
            the_model_result = self.fit_a_model(the_target_column=the_target_column,
                                                current_features=current_features,
                                                model_type=model_type)

            # check if the VIF is above the max_vif threshold
            if the_model_result.is_vif_above_threshold(max_allowable_vif=max_vif):
                # log that we need to remove a feature
                self.logger.debug(f"A feature was found to exceed the max_vif of [{max_vif}]")

                # get the feature with the max VIF
                the_feature = the_model_result.get_feature_with_max_vif()

                # log the feature with the max VIF
                self.logger.debug(f"removing feature [{the_feature}] to reduce VIF.")

                # add to internal storage a tuple with (reason, the_feature, the_linear_model_result)
                self.model_storage[the_step] = (LF_ELIM_REASON_VIF, the_feature, the_model_result)

                # dump to console for D208 PA requirement
                if not suppress_console:
                    self.log_model_summary_to_console(the_type=LM_STEP,
                                                      the_step=the_step,
                                                      feature_to_remove=the_feature,
                                                      reason_to_remove=LF_ELIM_REASON_VIF)

                # remove the feature from current_features
                current_features.remove(the_feature)

                # increment step
                the_step = the_step + 1
            else:
                # stop looping over VIF reduction.
                loop_again = False

        # reset loop_again for p-value reduction
        loop_again = True
        the_feature = None  # set back to None

        # loop until p-values are below the max_p_value
        while loop_again:
            assert the_feature not in current_features

            # get the_model_result
            the_model_result = self.fit_a_model(the_target_column=the_target_column,
                                                current_features=current_features,
                                                model_type=model_type)

            # check if there are p-values in excess of the max_p_value parameter
            if the_model_result.are_p_values_above_threshold(p_value=max_p_value):
                # set loop_again to True
                loop_again = True

                # identify the variable to remove
                the_feature = the_model_result.identify_parameter_based_on_p_value(p_value=max_p_value)

                # log what we're doing
                self.logger.debug(f"p-values found in excess of [{max_p_value}], removing [{the_feature}]")

                # add to internal storage (reason, the_feature, the_linear_model_result)
                self.model_storage[the_step] = (LF_ELIM_REASON_P_VALUE, the_feature, the_model_result)

                # dump to console for D208 PA requirement
                if not suppress_console:
                    self.log_model_summary_to_console(the_type=LM_STEP,
                                                      the_step=the_step,
                                                      feature_to_remove=the_feature,
                                                      reason_to_remove=LF_ELIM_REASON_P_VALUE)

                # remove the feature from current_features
                current_features.remove(the_feature)

                # increment the step
                the_step = the_step + 1
            else:
                loop_again = False

                # log that we're stopping
                self.logger.debug(f"no p-values exceeded threshold. Breaking out of loop")

        # return
        return the_model_result

    # dump the model summary to the console
    def log_model_summary_to_console(self, the_type: str, the_step=1, feature_to_remove=None, reason_to_remove=None):
        # perform argument validations
        if not isinstance(the_type, str):
            raise SyntaxError("the_type is None or incorrect type.")
        elif the_type not in [LM_INITIAL_MODEL, LM_FINAL_MODEL, LM_STEP]:
            raise ValueError("the_type argument is note recognized.")
        elif not isinstance(the_step, int):
            raise SyntaxError("the_step is None or incorrect type.")
        elif the_type == LM_STEP and the_step < 1:
            raise ValueError("the_step argument must be have domain [1, inf).")

        # determine the type
        if the_type == LM_INITIAL_MODEL:
            print("\nINITIAL MODEL RESULTS\n")
        elif the_type == LM_STEP:
            # check if the feature_to_remove is None
            if feature_to_remove is None:
                print(f"\nINCREMENTAL MODEL RESULTS - STEP {the_step}\n")
            else:
                print(f"\nINCREMENTAL MODEL RESULTS - STEP {the_step}, REMOVING [{feature_to_remove}], "
                      f"REASON [{reason_to_remove}]\n")
        else:
            print("\nFINAL MODEL RESULTS\n")

        # write out model.
        print(self.get_the_result().model.summary())

    # get encoded variables sans the target
    def get_encoded_variables(self, the_target_column) -> list:
        """
         return a list of encoded variables that does not include the_target_column
         :param the_target_column:
         :return: list
         """
        # perform argument validations
        if not isinstance(the_target_column, str):
            raise SyntaxError("the_target_column argument is None or incorrect type.")
        elif the_target_column not in self.encoded_df.columns:
            raise ValueError("the_target_column argument is not in dataframe.")

        # variable declaration
        the_result = self.encoded_df.columns.to_list()

        # remove the_target_column
        the_result.remove(the_target_column)

        # return
        return the_result

    # get the selected features dataframe
    def get_selected_features(self, current_features_df: DataFrame, the_target: Series,
                              p_val_sig: float, model_type: str) -> DataFrame:
        # run validations
        if not isinstance(current_features_df, DataFrame):
            raise AttributeError("current_features_df is None or incorrect type.")
        elif not isinstance(the_target, Series):
            raise AttributeError("the_target is None or incorrect type.")
        elif not isinstance(p_val_sig, float):
            raise AttributeError("p_val_sig is None or incorrect type.")
        elif p_val_sig <= 0 or p_val_sig > 1:
            raise AttributeError("p_val_sig must be in [0, 1].")
        elif model_type not in [MT_KNN_CLASSIFICATION, MT_RF_REGRESSION]:
            raise SyntaxError("model_type must be in [MT_KNN_CLASSIFICATION, MT_RF_REGRESSION].")

        # variable declaration
        p_values = None

        # Initialize the SelectKBest class to identify relevance of all features
        if model_type == MT_KNN_CLASSIFICATION:
            sk_best = SelectKBest(score_func=f_classif, k='all')
        else:
            sk_best = SelectKBest(score_func=f_regression, k='all')

        # let the SelectKBest calculate the p-values for all features.  We will not do anything with the
        # variable "selected_features".
        selected_features = sk_best.fit_transform(current_features_df, the_target)

        # Finding P-values to select statistically significant feature
        p_values = pd.DataFrame({'Feature': current_features_df.columns,
                                 'p_value': sk_best.pvalues_}).sort_values('p_value')

        # eliminate all features with a p_value > p_val_sig
        p_values = p_values[p_values['p_value'] < p_val_sig]

        # log what we have
        self.logger.debug(f"p_value_features->{p_values['Feature'].tolist()}")

        # return
        return p_values

    # split and scale
    def split_and_scale_features(self, current_features_df: DataFrame, the_target: Series, test_size: float) -> dict:
        # run validations
        if not isinstance(current_features_df, DataFrame):
            raise AttributeError("current_features_df is None or incorrect type.")
        elif not isinstance(the_target, Series):
            raise AttributeError("the_target is None or incorrect type.")
        elif not isinstance(test_size, float):
            raise AttributeError("test_size is None or incorrect type.")
        elif test_size <= 0 or test_size > 1:
            raise AttributeError("test_size must be in [0, 1].")

        # variable declaration
        the_result = {}

        # now we need to split our data.  The the_current_features_df should only contain our pruned list of
        # features.  Providing a variable name mapping
        # x_train--> the_f_df_train = the training features (not including the target)
        # x_test--> the_f_df_test = the test features (not including the target)
        # y_train--> the_t_var_train = the training target variable
        # y_test-->the_t_var_test = the test target variable
        the_f_df_train, the_f_df_test, the_t_var_train, the_t_var_test = train_test_split(
            current_features_df,
            the_target,
            test_size=test_size,
            random_state=12345)

        # make a copy of the the_f_df_test
        the_f_df_test_orig = the_f_df_test.copy()

        # create a StandardScaler
        the_scalar = StandardScaler()

        # get the list of features we want to scale
        list_of_features_to_scale = self.dataset_analyzer.retrieve_features_of_specific_type([INT64_COLUMN_KEY,
                                                                                              FLOAT64_COLUMN_KEY])

        # now we need to get the features we can scale.
        the_f_df_train_to_scale = [i for i in the_f_df_train.columns.to_list() if i in list_of_features_to_scale]

        # log the list of features to scale
        self.logger.debug(f"the features that will be scaled {the_f_df_train_to_scale}.")

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

        # assemble the final result
        the_result[X_TRAIN] = the_f_df_train
        the_result[X_TEST] = the_f_df_test
        the_result[Y_TRAIN] = the_t_var_train
        the_result[Y_TEST] = the_t_var_test
        the_result[X_ORIGINAL] = the_f_df_test_orig

        # return
        return the_result
