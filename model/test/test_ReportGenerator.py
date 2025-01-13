import os
import unittest

from os.path import exists
from pandas import DataFrame
from model.Project_Assessment import Project_Assessment
from model.ReportGenerator import ReportGenerator
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from model.analysis.PCA_Analysis import PCA_Analysis
from model.analysis.PlotGenerator import PlotGenerator
from model.constants.BasicConstants import ANALYSIS_TYPE, ANALYZE_DATASET_FULL, D_212_CHURN, ANALYZE_DATASET_INITIAL, \
    MT_LOGISTIC_REGRESSION, MT_LINEAR_REGRESSION, MT_KNN_CLASSIFICATION, MT_RF_REGRESSION
from model.constants.ModelConstants import LM_INITIAL_MODEL, LM_FINAL_MODEL
from util.CSV_loader import CSV_Loader


# test class for the ReportGenerator
class test_ReportGenerator(unittest.TestCase):
    # constants
    DEFAULT_PATH = "resources/Output/DATASET_ANALYSIS.xlsx"
    OVERRIDE_PATH = "../../resources/Output/"
    BAD_OVERRIDE_PATH = "/bar/foo/"
    FULL_OVERRIDE_PATH = "../../resources/Output/DATASET_ANALYSIS.xlsx"
    VALID_CSV_PATH = "../../resources/Input/churn_raw_data.csv"
    VALID_DATAFRAME_XLSX_FILE = "../../resources/Output/DATAFRAME_OUTPUT.xlsx"
    VALID_BASE_DIR = "/Users/robertfalast/PycharmProjects/PA_212/"
    DATASET_ANALYSIS_LOC = "../../resources/Output/DATASET_ANALYSIS.xlsx"
    INITIAL_DATASET_ANALYSIS_LOC = "../../resources/Output/INITIAL_DATASET_ANALYSIS.xlsx"

    VALID_COLUMN_DICT = {"PC1": "Age", "PC2": "Tenure", "PC3": "MonthlyCharge", "PC4": "Bandwidth_GB_Year",
                         "PC5": "Yearly_equip_failure", "PC6": "Contacts", "PC7": "Outage_sec_perweek",
                         "PC8": "Children", "PC9": "Income"}

    field_rename_dict = {"Item1": "Timely_Response", "Item2": "Timely_Fixes", "Item3": "Timely_Replacements",
                         "Item4": "Reliability", "Item5": "Options", "Item6": "Respectful_Response",
                         "Item7": "Courteous_Exchange", "Item8": "Active_Listening"}

    column_drop_list = ['Zip', 'Lat', 'Lng', 'Customer_id', 'Interaction', 'State', 'UID', 'County', 'Job', 'City']

    CHURN_KEY = D_212_CHURN

    def test_init(self):
        # instantiate a ReportGenerator
        r_gen = ReportGenerator()

        # run assertions
        self.assertIsNotNone(r_gen.output_path)
        self.assertEqual(r_gen.output_path, self.DEFAULT_PATH)

    # negative tests for the init method
    def test_init_negative(self):

        # we're going to call analyze_dataset() without having loaded it first
        with self.assertRaises(BaseException) as context:
            # instantiate a ReportGenerator
            r_gen = ReportGenerator(self.BAD_OVERRIDE_PATH)

            # validate the error message.
            self.assertTrue("The overriden path does not exist, or is not valid." in context.exception)

    # test the override final report option for init() method
    def test_init_with_override(self):
        # instantiate a ReportGenerator
        r_gen = ReportGenerator(self.OVERRIDE_PATH)

        # run assertions
        self.assertIsNotNone(r_gen.output_path)
        self.assertEqual(r_gen.output_path, self.FULL_OVERRIDE_PATH)
        self.assertEqual(r_gen.dataframe_path, self.VALID_DATAFRAME_XLSX_FILE)
        self.assertEqual(r_gen.report_path, self.OVERRIDE_PATH)

    # negative tests for generate_excel_report()
    def test_generate_excel_report_negative(self):
        # instantiate a ReportGenerator
        r_gen = ReportGenerator(self.OVERRIDE_PATH)

        # get the dataframe from the CSV file
        csv_l = CSV_Loader()

        # instantiate a DatasetAnalyzer
        d_analyzer = DatasetAnalyzer(csv_l.get_data_frame_from_csv(self.VALID_CSV_PATH))

        # refresh the model
        d_analyzer.run_complete_setup()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(d_analyzer, self.OVERRIDE_PATH)

        # override the output_path
        r_gen.output_path = None

        # make sure we handle None, None, None
        with self.assertRaises(IOError) as context:
            # invoke method.
            r_gen.generate_excel_report(the_dataset_analyzer=None,
                                        the_plot_generator=None,
                                        stat_generator=None,
                                        the_model_type=None)

        # validate the error message.
        self.assertEqual("The output path is None.", str(context.exception))

        # instantiate a ReportGenerator because the other one is a mess.
        r_gen = ReportGenerator()

        # make sure we handle DatasetAnalyzer, None, None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke method.
            r_gen.generate_excel_report(the_dataset_analyzer=d_analyzer,
                                        the_plot_generator=None,
                                        stat_generator=None,
                                        the_model_type=None)

        # validate the error message.
        self.assertEqual("The PlotGenerator argument was None or incorrect type.", str(context.exception))

        # create a ReportGenerator and PlotGenerator
        r_gen = ReportGenerator()
        the_pg = PlotGenerator(d_analyzer)

        # make sure we handle DatasetAnalyzer, PlotGenerator, None, ANALYSIS_TYPE[0]
        with self.assertRaises(SyntaxError) as context:
            # invoke method
            r_gen.generate_excel_report(the_dataset_analyzer=d_analyzer,
                                        the_plot_generator=the_pg,
                                        stat_generator=None,
                                        the_model_type=None,
                                        the_type=ANALYZE_DATASET_FULL)

        # validate the error message.
        self.assertEqual("The StatisticsGenerator argument was None or incorrect type.", str(context.exception))

    # test the generate_excel_report() method with FULL setup being run LOGISTIC model
    def test_generate_excel_report_full_logistic_model(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)
        pa.build_model(the_target_column='Churn', model_type=MT_LOGISTIC_REGRESSION, max_p_value=0.001)

        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # get required references
        r_gen = pa.r_gen   # ReportGenerator
        s_gen = pa.s_gen   # StatisticsGenerator

        # create a PlotGenerator
        p_gen = PlotGenerator(pa.analyzer, r_gen.report_path)

        # tell it to generate all the plots required
        p_gen.generate_all_dataset_plots(statistics_generator=s_gen,
                                         the_model_type=MT_LOGISTIC_REGRESSION,
                                         the_version=ANALYZE_DATASET_FULL)

        # generate heatmap
        p_gen.generate_correlation_heatmap(s_gen.get_list_of_correlations())

        # verify the output path variable
        self.assertEqual(r_gen.output_path, self.FULL_OVERRIDE_PATH)

        # invoke the method.
        r_gen.generate_excel_report(the_dataset_analyzer=pa.analyzer,
                                    the_plot_generator=p_gen,
                                    stat_generator=s_gen,
                                    the_model_type=MT_LOGISTIC_REGRESSION,
                                    the_type=ANALYSIS_TYPE[0])

        # run validations
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))

    # test the generate_excel_report() method with FULL setup being run for LINEAR model
    def test_generate_excel_report_full_linear_model(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # get the dataframe from the pa
        the_df = pa.df

        # run assertions on the initial dataframe
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns.to_list()), 39)
        self.assertTrue('MonthlyCharge' in the_df.columns.to_list())

        pa.build_model(the_target_column='MonthlyCharge', model_type=MT_LINEAR_REGRESSION, max_p_value=0.001)

        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # get required references
        r_gen = pa.r_gen   # ReportGenerator
        s_gen = pa.s_gen   # StatisticsGenerator

        # create a PlotGenerator
        p_gen = PlotGenerator(pa.analyzer, r_gen.report_path)

        # tell it to generate all the plots required
        p_gen.generate_all_dataset_plots(statistics_generator=s_gen,
                                         the_model_type=MT_LINEAR_REGRESSION,
                                         the_version=ANALYZE_DATASET_FULL)

        # generate heatmap
        p_gen.generate_correlation_heatmap(s_gen.get_list_of_correlations())

        # verify the output path variable
        self.assertEqual(r_gen.output_path, self.FULL_OVERRIDE_PATH)

        # invoke the method.
        r_gen.generate_excel_report(the_dataset_analyzer=pa.analyzer,
                                    the_plot_generator=p_gen,
                                    stat_generator=s_gen,
                                    the_model_type=MT_LINEAR_REGRESSION,
                                    the_type=ANALYZE_DATASET_FULL)

        # run validations
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))

    # test the generate_excel_report() method with FULL setup being run for KNN model
    def test_generate_excel_report_full_knn_model(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # get the dataframe from the pa
        the_df = pa.df

        # run assertions on the initial dataframe
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns.to_list()), 39)
        self.assertTrue('Churn' in the_df.columns.to_list())

        pa.build_model(the_target_column='Churn',
                       model_type=MT_KNN_CLASSIFICATION,
                       max_p_value=0.001)

        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # get required references
        r_gen = pa.r_gen   # ReportGenerator
        s_gen = pa.s_gen   # StatisticsGenerator

        # create a PlotGenerator
        p_gen = PlotGenerator(pa.analyzer, r_gen.report_path)

        # tell it to generate all the plots required
        p_gen.generate_all_dataset_plots(statistics_generator=s_gen,
                                         the_model_type=MT_KNN_CLASSIFICATION,
                                         the_version=ANALYZE_DATASET_FULL)

        # verify the output path variable
        self.assertEqual(r_gen.output_path, self.FULL_OVERRIDE_PATH)

        # invoke the method.
        r_gen.generate_excel_report(the_dataset_analyzer=pa.analyzer,
                                    the_plot_generator=p_gen,
                                    stat_generator=s_gen,
                                    the_model_type=MT_KNN_CLASSIFICATION,
                                    the_type=ANALYZE_DATASET_FULL)

        # run validations
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))

    # test the generate_excel_report() method with FULL setup being run for Random Forest model
    def test_generate_excel_report_full_random_forest_model(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # validate the linear_model storage is empty
        self.assertFalse(LM_INITIAL_MODEL in pa.analyzer.linear_model_storage)
        self.assertFalse(LM_FINAL_MODEL in pa.analyzer.linear_model_storage)

        # get the dataframe from the pa
        the_df = pa.df

        # run assertions on the initial dataframe
        self.assertIsNotNone(the_df)
        self.assertIsInstance(the_df, DataFrame)
        self.assertEqual(len(the_df.columns.to_list()), 39)
        self.assertTrue('Bandwidth_GB_Year' in the_df.columns.to_list())

        pa.build_model(the_target_column='Bandwidth_GB_Year',
                       model_type=MT_RF_REGRESSION,
                       max_p_value=0.001)

        # delete the output report if it previously exists
        if exists(self.DATASET_ANALYSIS_LOC):
            os.remove(self.DATASET_ANALYSIS_LOC)

        # get required references
        r_gen = pa.r_gen   # ReportGenerator
        s_gen = pa.s_gen   # StatisticsGenerator

        # create a PlotGenerator
        p_gen = PlotGenerator(pa.analyzer, r_gen.report_path)

        # tell it to generate all the plots required
        p_gen.generate_all_dataset_plots(statistics_generator=s_gen,
                                         the_model_type=MT_RF_REGRESSION,
                                         the_version=ANALYZE_DATASET_FULL)

        # verify the output path variable
        self.assertEqual(r_gen.output_path, self.FULL_OVERRIDE_PATH)

        # invoke the method.
        r_gen.generate_excel_report(the_dataset_analyzer=pa.analyzer,
                                    the_plot_generator=p_gen,
                                    stat_generator=s_gen,
                                    the_model_type=MT_RF_REGRESSION,
                                    the_type=ANALYZE_DATASET_FULL)

        # run validations
        self.assertTrue(exists(self.DATASET_ANALYSIS_LOC))

    # test the generate_excel_report() method with INITIAL setup being run
    def test_generate_excel_report_initial(self):
        # delete the output report if it previously exists
        if exists(self.INITIAL_DATASET_ANALYSIS_LOC):
            os.remove(self.INITIAL_DATASET_ANALYSIS_LOC)

        # instantiate a ReportGenerator
        r_gen = ReportGenerator(self.OVERRIDE_PATH)

        # get the dataframe from the CSV file
        csv_l = CSV_Loader()

        # instantiate a DatasetAnalyzer
        d_analyzer = DatasetAnalyzer(csv_l.get_data_frame_from_csv(self.VALID_CSV_PATH))

        # only refresh the data, not full setup
        d_analyzer.refresh_model()
        d_analyzer.extract_boolean()

        # instantiate a PlotGenerator
        the_pg = PlotGenerator(d_analyzer, self.OVERRIDE_PATH)

        # tell it to generate all the plots required
        the_pg.generate_all_dataset_plots(statistics_generator=None,
                                          the_model_type=MT_KNN_CLASSIFICATION,
                                          the_version=ANALYZE_DATASET_INITIAL)

        # invoke the method.
        r_gen.generate_excel_report(the_dataset_analyzer=d_analyzer,
                                    the_plot_generator=the_pg,
                                    stat_generator=None,
                                    the_model_type=MT_KNN_CLASSIFICATION,
                                    the_type=ANALYZE_DATASET_INITIAL)

    # negative tests for the export_dataframes_to_excel()
    def test_export_dataframes_to_excel_negative(self):
        # instantiate a ReportGenerator
        r_gen = ReportGenerator(self.OVERRIDE_PATH)

        # get the dataframe from the CSV file
        csv_l = CSV_Loader()

        # instantiate a DatasetAnalyzer
        d_analyzer = DatasetAnalyzer(csv_l.get_data_frame_from_csv(self.VALID_CSV_PATH))

        # make sure we can handle None, None
        with self.assertRaises(SyntaxError) as context:
            # call generate_excel_report()
            r_gen.export_dataframes_to_excel(None, None)

            # validate the error message.
            self.assertTrue("The dataset analyzer was None or incorrect type." in context.exception)

        # make sure we can handle empty DatasetAnalyzer, None
        with self.assertRaises(SyntaxError) as context:
            # call generate_excel_report() d_analyzer, None
            r_gen.export_dataframes_to_excel(d_analyzer, None)

            # validate the error message.
            self.assertTrue("Cannot export dataframe since it has not been created." in context.exception)

        # setup the dataframe, but do not perform normalization
        d_analyzer.run_complete_setup()

        # run assertions to make sure the test is valid.
        self.assertIsNotNone(d_analyzer.the_df)
        self.assertIsNone(d_analyzer.the_normal_df)

        # make sure we can handle DatasetAnalyzer partially populated, None
        with self.assertRaises(SyntaxError) as context:
            # call generate_excel_report() d_analyzer, None
            r_gen.export_dataframes_to_excel(d_analyzer, None)

            # validate the error message.
            self.assertTrue("Cannot export normalized dataframe since it has not been created." in context.exception)

        # normalize the dataset
        d_analyzer.normalize_dataset()
        self.assertIsNotNone(d_analyzer.the_normal_df)

        # make sure we can handle DatasetAnalyzer partially populated, None
        with self.assertRaises(SyntaxError) as context:
            # call generate_excel_report() d_analyzer, None
            r_gen.export_dataframes_to_excel(d_analyzer, None)

            # validate the error message.
            self.assertTrue("The PCA_Analysis was None or incorrect type." in context.exception)

    # test the export_dataframes_to_excel() method
    def test_export_dataframes_to_excel(self):
        # instantiate a ReportGenerator
        r_gen = ReportGenerator(self.OVERRIDE_PATH)

        # get the dataframe from the CSV file
        csv_l = CSV_Loader()

        # instantiate a DatasetAnalyzer
        d_analyzer = DatasetAnalyzer(csv_l.get_data_frame_from_csv(self.VALID_CSV_PATH))

        # populate the dataset analyzer
        d_analyzer.run_complete_setup()
        d_analyzer.normalize_dataset()

        # create PCA Analysis object
        self.pca_analysis = PCA_Analysis(d_analyzer, self.VALID_COLUMN_DICT)

        # perform analysis
        self.pca_analysis.perform_analysis()

        # invoke the function
        r_gen.export_dataframes_to_excel(d_analyzer, self.pca_analysis)

        # run assertions
        self.assertTrue(exists(self.VALID_DATAFRAME_XLSX_FILE))

    # test method for initialize_counter_storage()
    def test_initialize_counter_storage(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

    # negative tests for get_model_type_title()
    def test_get_model_type_title_negative(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get required references
        r_gen = pa.r_gen

        # run assertions
        self.assertIsNotNone(r_gen)

        # verify we handle None, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            r_gen.get_model_type_title(the_model_type=None, the_run_type=None)

        # validate the error message.
        self.assertEqual("the_model_type is None or incorrect option.", context.exception.msg)

        # verify we handle foo, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            r_gen.get_model_type_title(the_model_type="foo", the_run_type=None)

        # validate the error message.
        self.assertEqual("the_model_type is None or incorrect option.", context.exception.msg)

        # verify we handle MT_LINEAR_REGRESSION, None
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            r_gen.get_model_type_title(the_model_type=MT_LINEAR_REGRESSION, the_run_type=None)

        # validate the error message.
        self.assertEqual("the_run_type is None or incorrect option.", context.exception.msg)

        # verify we handle MT_LINEAR_REGRESSION, "foo"
        with self.assertRaises(SyntaxError) as context:
            # invoke the method
            r_gen.get_model_type_title(the_model_type=MT_LINEAR_REGRESSION, the_run_type="foo")

        # validate the error message.
        self.assertEqual("the_run_type is None or incorrect option.", context.exception.msg)

    # test method for get_model_type_title()
    def test_get_model_type_title(self):
        pa = Project_Assessment(base_directory=self.VALID_BASE_DIR, report_loc_override=self.OVERRIDE_PATH)

        pa.load_dataset(dataset_name_key=self.CHURN_KEY)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.update_column_names(self.field_rename_dict)
        pa.drop_column_from_dataset(self.column_drop_list)
        pa.analyze_dataset(ANALYZE_DATASET_FULL)
        pa.calculate_internal_statistics(the_level=0.5)

        # get required references
        r_gen = pa.r_gen

        # run assertions
        self.assertIsNotNone(r_gen)

        # test the_model_type=MT_LINEAR_REGRESSION, the_run_type=ANALYZE_DATASET_INITIAL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_LINEAR_REGRESSION,
                                                    the_run_type=ANALYZE_DATASET_INITIAL),
                         "INITIAL LINEAR MODEL RESULTS")

        # test the_model_type=MT_LINEAR_REGRESSION, the_run_type=ANALYZE_DATASET_FULL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_LINEAR_REGRESSION,
                                                    the_run_type=ANALYZE_DATASET_FULL),
                         "FINAL LINEAR MODEL RESULTS")

        # test the_model_type=MT_LOGISTIC_REGRESSION, the_run_type=ANALYZE_DATASET_INITIAL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_LOGISTIC_REGRESSION,
                                                    the_run_type=ANALYZE_DATASET_INITIAL),
                         "INITIAL LOGISTIC MODEL RESULTS")

        # test the_model_type=MT_LOGISTIC_REGRESSION, the_run_type=ANALYZE_DATASET_FULL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_LOGISTIC_REGRESSION,
                                                    the_run_type=ANALYZE_DATASET_FULL),
                         "FINAL LOGISTIC MODEL RESULTS")

        # test the_model_type=MT_KNN_CLASSIFICATION, the_run_type=ANALYZE_DATASET_INITIAL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_KNN_CLASSIFICATION,
                                                    the_run_type=ANALYZE_DATASET_INITIAL),
                         "KNN MODEL RESULT")

        # test the_model_type=MT_KNN_CLASSIFICATION, the_run_type=ANALYZE_DATASET_FULL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_KNN_CLASSIFICATION,
                                                    the_run_type=ANALYZE_DATASET_FULL),
                         "KNN MODEL RESULT")

        # test the_model_type=MT_RF_REGRESSION, the_run_type=ANALYZE_DATASET_INITIAL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_RF_REGRESSION,
                                                    the_run_type=ANALYZE_DATASET_INITIAL),
                         "RANDOM FOREST MODEL RESULT")

        # test the_model_type=MT_RF_REGRESSION, the_run_type=ANALYZE_DATASET_FULL
        self.assertEqual(r_gen.get_model_type_title(the_model_type=MT_RF_REGRESSION,
                                                    the_run_type=ANALYZE_DATASET_FULL),
                         "RANDOM FOREST MODEL RESULT")
