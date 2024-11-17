import unittest

from os.path import exists
from openpyxl.workbook import Workbook
from model.analysis.DatasetAnalyzer import DatasetAnalyzer
from util.CSV_loader import CSV_Loader
from util.ExcelManager import ExcelManager


# test class for ExcelManager
class test_ExcelManager(unittest.TestCase):
    # constants
    VALID_CSV_PATH = "../../resources/Input/churn_raw_data.csv"
    TEST_WB_PATH = "../../resources/Output/test.xlsx"
    VALID_WB_PATH = "../../resources/Output/test.xlsx"
    INVALID_WB_PATH = "foo.txt"
    VALID_INT_COL_NAME_1 = "Population"
    VALID_INT_COL_NAME_2 = "Contacts"
    VALID_FLT_COL_NAME_1 = "Income"
    VALID_FLT_COL_NAME_2 = "Tenure"
    VALID_TAB_NAME_1 = "tab 1"

    # test init method
    def test_init(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # run validations
        self.assertIsNotNone(excel_manager.wb_storage)

    # negative tests for the create_workbook() method
    def test_create_workbook_negative(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # None arguments for Converter()
            excel_manager.create_workbook(None)

            # validate the error message.
            self.assertTrue("The path was None or incorrect type." in context.exception)

    # tests for the create_workbook() method
    def test_create_workbook(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # invoke the function
        excel_manager.create_workbook(the_path=self.VALID_WB_PATH)

        # run assertions
        self.assertTrue(exists(self.VALID_WB_PATH))
        self.assertIsNotNone(excel_manager.wb_storage)
        self.assertEqual(len(excel_manager.wb_storage), 0)
        self.assertTrue(exists(self.VALID_WB_PATH))

    # negative tests for the close_workbook() method
    def test_close_workbook_negative(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.close_workbook(None)

            # validate the error message.
            self.assertTrue("The path was None or incorrect type." in context.exception)

        # we're going to pass in int
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.close_workbook(2)

            # validate the error message.
            self.assertTrue("The path was None or incorrect type." in context.exception)

        # we're going to pass in INVALID_WB_PATH, None
        with self.assertRaises(FileNotFoundError) as context:
            # run method
            excel_manager.close_workbook(self.INVALID_WB_PATH)

            # validate the error message.
            self.assertTrue(f"The path [{self.INVALID_WB_PATH}] does not exist." in context.exception)

    # test the close_workbook() method
    def test_close_workbook(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # create a workbook
        excel_manager.create_workbook(the_path=self.VALID_WB_PATH)

        # open the workbook
        excel_manager.open_workbook(self.VALID_WB_PATH)

        # make sure the workbook is in storage
        self.assertIsNotNone(excel_manager.wb_storage[self.VALID_WB_PATH])
        self.assertIsInstance(excel_manager.wb_storage[self.VALID_WB_PATH], Workbook)

        # close the workbook
        excel_manager.close_workbook(self.VALID_WB_PATH)

        # run assertions
        self.assertTrue(exists(self.VALID_WB_PATH))
        self.assertFalse(self.VALID_WB_PATH in excel_manager.wb_storage)

    # negative tests for the open_workbook() method
    def test_open_workbook_negative(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.open_workbook(None)

            # validate the error message.
            self.assertTrue("The path was None or incorrect type." in context.exception)

        # we're going to pass in int
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.open_workbook(4)

            # validate the error message.
            self.assertTrue("The path was None or incorrect type." in context.exception)

        # we're going to pass in INVALID_WB_PATH
        with self.assertRaises(FileNotFoundError) as context:
            # run method
            excel_manager.open_workbook(self.INVALID_WB_PATH)

            # validate the error message.
            self.assertTrue(f"The path [{self.INVALID_WB_PATH}] does not exist." in context.exception)

    # test the open_workbook() method
    def test_open_workbook(self):
        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # invoke the function
        the_wb = excel_manager.open_workbook(self.VALID_WB_PATH)

        # run assertions
        self.assertIsNotNone(the_wb)
        self.assertIsInstance(the_wb, Workbook)
        self.assertIsInstance(excel_manager.wb_storage[self.VALID_WB_PATH], Workbook)
        self.assertEqual(the_wb, excel_manager.wb_storage[self.VALID_WB_PATH])

    # negative tests for write_df_into_wb_tab() method
    def test_write_df_into_wb_tab_negative(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run setup
        dsa.run_complete_setup()

        # get valid df reference
        the_df = dsa.the_df

        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # we're going to pass in None, None, None
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.write_df_into_wb_tab(None, None, None)

            # validate the error message.
            self.assertTrue("The incoming DataFrame was None or incorrect type." in context.exception)

        # we're going to pass in the_df, None, None
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.write_df_into_wb_tab(the_df, None, None)

            # validate the error message.
            self.assertTrue("The incoming tab name was None or incorrect type." in context.exception)

        # we're going to pass in the_df, VALID_TAB_NAME_1, None
        with self.assertRaises(SyntaxError) as context:
            # run method
            excel_manager.write_df_into_wb_tab(the_df, self.VALID_TAB_NAME_1, None)

            # validate the error message.
            self.assertTrue("The incoming workbook was None or incorrect type." in context.exception)

    # test method for write_df_into_wb_tab()
    def test_write_df_into_wb_tab(self):
        # create an instance of the CSV_Loader
        cl = CSV_Loader()

        # get a dataframe
        df = cl.get_data_frame_from_csv(self.VALID_CSV_PATH)

        # instantiate a DatasetAnalyzer()
        dsa = DatasetAnalyzer(df)

        # run setup
        dsa.run_complete_setup()

        # get valid df reference
        the_df = dsa.the_df

        # create an instance of the ExcelManager
        excel_manager = ExcelManager()

        # create the workbook
        excel_manager.create_workbook(self.VALID_WB_PATH)

        # open the workbook
        the_wb = excel_manager.open_workbook(self.VALID_WB_PATH)

        # invoke the method
        excel_manager.write_df_into_wb_tab(the_df, self.VALID_TAB_NAME_1, the_wb)

        # close the workbook
        excel_manager.close_workbook(self.VALID_WB_PATH)

    # test method for increment_column()
    def test_increment_column(self):
        # create an instance of ExcelManager
        em = ExcelManager()

        # null hypothesis test
        self.assertEqual(em.increment_column("A", 0), "A")
        # simple increment of A-->F
        self.assertEqual(em.increment_column("A", 5), "F")
        # simple increment of JA-->JF
        self.assertEqual(em.increment_column("JA", 5), "JF")
        # simple increment of ZJA-->ZJF
        self.assertEqual(em.increment_column("ZJA", 5), "ZJF")
        # simple roll to next address
        self.assertEqual(em.increment_column("A", 30), "AF")
        # more complicated
        self.assertEqual(em.increment_column("BA", 30), "CF")
        # more complicated
        self.assertEqual(em.increment_column("QA", 30), "RF")
        # even more complicated
        self.assertEqual(em.increment_column("ZBA", 30), "ZCF")

    # positive tests for offset_excel_address()
    def test_offset_excel_address(self):
        # create an instance of ExcelManager
        em = ExcelManager()

        # do a null test
        self.assertEqual(em.offset_excel_address("A1", 0, 0), "A1")

        # do a test that capitalization doesn't matter
        self.assertEqual(em.offset_excel_address("a1", 0, 0), "A1")

        # do a row only adjustment
        self.assertEqual(em.offset_excel_address("A1", 0, 4), "A5")

        # do a large row only adjustment
        self.assertEqual(em.offset_excel_address("A1", 0, 1004), "A1005")

        # do a small column adjustment
        self.assertEqual(em.offset_excel_address("A1", 1, 0), "B1")

        # do a larger column adjustment, < 26
        self.assertEqual(em.offset_excel_address("A1", 12, 0), "M1")

        # do a column and row adjustment
        self.assertEqual(em.offset_excel_address("A1", 1, 1), "B2")

        # do a larger column adjustment, < 26, with large row adjustment
        self.assertEqual(em.offset_excel_address("A1", 12, 18), "M19")

        # now we're going crazy
        self.assertEqual(em.offset_excel_address("A1", 32, 18), "AH19")