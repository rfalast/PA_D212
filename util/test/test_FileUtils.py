import os
import unittest

from copy import copy
from os.path import exists
from pathlib import Path
from util.FileUtils import FileUtils


# test cases for FileUtils class
class test_FileUtils(unittest.TestCase):
    # constants
    EXISTING_DIRECTORY = "../../resources/Output/"
    NEW_DIRECTORY = "../../resources/Output/test/"
    EXPECTED_BASE_DIRECTORY = "/Users/robertfalast/PycharmProjects/PA_212/util"
    OS_OPTIONS = ['macOS', 'Windows', 'Darwin']
    DIRECTORY_SEPERATOR = {"macOS": '/', "Windows": '\\\\', "Darwin": '/'}
    WIN_PATH = '\\foo\\bar\\some\\dir'
    MAC_PATH = "/foo/bar/some/dir"

    WIN_PATH_TEST = "\\Users\\robertfalast\\PycharmProjects\\PA_212\\resources\\Input\\"
    WIN_PATH_TEST = "/Users/robertfalast/PycharmProjects/PA_212/resources/Input/"

    # test init method
    def test_init(self):
        # create object
        file_util = FileUtils()

        # run assertions
        self.assertIsNotNone(file_util)
        self.assertIsInstance(file_util, FileUtils)

    # negative tests for create_output_directories()
    def test_create_output_directories_negative(self):
        # create object
        file_util = FileUtils()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # None arguments for Converter()
            file_util.create_output_directories(None)

            # validate the error message.
            self.assertTrue("The path argument was None or incorrect type." in context.exception)

    # test methods for the create_output_directories() method
    def test_create_output_directories(self):
        # create object
        file_util = FileUtils()

        # make sure an existing directory is there
        self.assertTrue(exists(self.EXISTING_DIRECTORY))

        # invoke the method
        file_util.create_output_directories(self.EXISTING_DIRECTORY)

        # make sure an existing directory is still there
        self.assertTrue(exists(self.EXISTING_DIRECTORY))

        # invoke on a new directory that didn't previously exist
        self.assertFalse(exists(self.NEW_DIRECTORY))
        file_util.create_output_directories(self.NEW_DIRECTORY)
        self.assertTrue(exists(self.NEW_DIRECTORY))

    # negative tests for the remove_output_directory() method
    def test_remove_output_directory_negative(self):
        # create object
        file_util = FileUtils()

        # we're going to pass in None
        with self.assertRaises(SyntaxError) as context:
            # None arguments for Converter()
            file_util.remove_output_directory(None)

            # validate the error message.
            self.assertTrue("The path argument was None or incorrect type." in context.exception)

    # test cases for the remove_output_directory() method
    def test_remove_output_directory(self):
        # create object
        file_util = FileUtils()

        # check if the directory exists to determine if we need to create it first
        if not exists(self.NEW_DIRECTORY):
            file_util.create_output_directories(self.NEW_DIRECTORY)

        # run assertions
        self.assertTrue(exists(self.NEW_DIRECTORY))

        # delete directory
        file_util.remove_output_directory(self.NEW_DIRECTORY)

        # verify it is gone
        self.assertFalse(exists(self.NEW_DIRECTORY))

    # test method for get_base_directory() method
    def test_get_base_directory(self):
        file_utils = FileUtils()

        # run assertions
        self.assertIsNotNone(file_utils.get_base_directory())
        self.assertIsInstance(file_utils.get_base_directory(), str)
        self.assertEqual(file_utils.get_base_directory(), self.EXPECTED_BASE_DIRECTORY)

    # test method for find_os_type() method
    def test_find_os_type(self):
        # create object
        file_util = FileUtils()

        # invoke the function
        os_returned = file_util.find_os_type()

        # run assertions
        self.assertIsNotNone(os_returned)
        self.assertIsInstance(os_returned, str)
        self.assertTrue(os_returned in self.OS_OPTIONS)

    # test method for get_directory_seperator()
    def test_get_directory_seperator(self):
        # create object
        file_util = FileUtils()

        # invoke the function
        seperator = file_util.get_directory_seperator()

        # run assertions
        self.assertIsNotNone(seperator)
        self.assertIsInstance(seperator, str)
        self.assertEqual(seperator, self.DIRECTORY_SEPERATOR[file_util.os_type])

    # test method for correct_path_for_this_os()
    def test_correct_path_for_this_os(self):
        base_dir = FileUtils.get_base_directory()

        data_folder = Path("/../resources/Input/")

        file_to_open = data_folder / "test.txt"

        # make sure we can create a file.
        if not exists(file_to_open):
            # create the file.
            f = open("../../resources/Input/test.txt", "w+")

            # write to file
            f.write("my test.")

            # close the file
            f.close()

        # verify that the file is there
        self.assertTrue(exists("../../resources/Input/test.txt"))

        # define the original path.  In this case, use a windows path
        original_win_path = copy(self.WIN_PATH_TEST)

        # create a FileUtils instances
        f_util = FileUtils()

        # invoke the method
        # Note, this test always fails.
        updated_path = f_util.correct_path_for_this_os(the_path=original_win_path)

        # run assertions.
        self.assertIsNotNone(updated_path)
        self.assertIsInstance(updated_path, str)


