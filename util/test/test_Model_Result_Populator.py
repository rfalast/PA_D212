import unittest

from util.Model_Result_Populator import Model_Result_Populator


class test_Model_Result_Populator(unittest.TestCase):

    # negative test for the populate_storage() method
    def test_populate_storage_negative(self):
        # instantiate an instance of Model_Result_Populator
        mrp = Model_Result_Populator()

        # check that storage is empty
        self.assertEqual(len(mrp.storage), 0)

        # make sure we throw an AttributeError for None argument
        with self.assertRaises(AttributeError) as context:
            # add None
            mrp.populate_storage(the_key=None, the_item=None)

        # validate the error message.
        self.assertTrue("the_key is None." in context.exception.args)

        # make sure we throw an AttributeError for None argument
        with self.assertRaises(AttributeError) as context:
            # add None
            mrp.populate_storage(the_key="foo", the_item=None)

        # validate the error message.
        self.assertTrue("the_item is None." in context.exception.args)

    # test the populate_storage() method
    def test_populate_storage(self):
        # instantiate an instance of Model_Result_Populator
        mrp = Model_Result_Populator()

        # check that storage is empty
        self.assertEqual(len(mrp.storage), 0)

        # add an item
        mrp.populate_storage(the_key="foo", the_item="bar")

        # run assertions
        self.assertEqual(len(mrp.storage), 1)
        self.assertTrue("foo" in mrp.storage)

        # add another item
        mrp.populate_storage(the_key="foo_2", the_item="bar_2")
        self.assertEqual(len(mrp.storage), 2)
        self.assertTrue("foo" in mrp.storage)
        self.assertTrue("foo_2" in mrp.storage)

    # negative test for remove_from_storage()
    def test_remove_from_storage_negative(self):
        # instantiate an instance of Model_Result_Populator
        mrp = Model_Result_Populator()

        # check that storage is empty
        self.assertEqual(len(mrp.storage), 0)

        # add an item
        mrp.populate_storage(the_key="foo", the_item="bar")

        # run assertions
        self.assertEqual(len(mrp.storage), 1)
        self.assertTrue("foo" in mrp.storage)

        # invoke method
        mrp.remove_from_storage(the_key="foo_1")

        # run assertions
        self.assertEqual(len(mrp.storage), 1)
        self.assertTrue("foo" in mrp.storage)

    # test the remove_from_storage() method
    def test_remove_from_storage(self):
        # instantiate an instance of Model_Result_Populator
        mrp = Model_Result_Populator()

        # check that storage is empty
        self.assertEqual(len(mrp.storage), 0)

        # add an item
        mrp.populate_storage(the_key="foo", the_item="bar")

        # run assertions
        self.assertEqual(len(mrp.storage), 1)
        self.assertTrue("foo" in mrp.storage)

        # invoke method
        mrp.remove_from_storage(the_key="foo")

        # run assertions
        self.assertEqual(len(mrp.storage), 0)
        self.assertFalse("foo" in mrp.storage)

        # now add two items
        mrp.populate_storage(the_key="foo", the_item="bar")
        mrp.populate_storage(the_key="foo_2", the_item="bar_2")

        # run assertions
        self.assertEqual(len(mrp.storage), 2)
        self.assertTrue("foo" in mrp.storage)
        self.assertTrue("foo_2" in mrp.storage)

        # invoke the method again
        # invoke method
        mrp.remove_from_storage(the_key="foo")
        self.assertFalse("foo" in mrp.storage)
        self.assertTrue("foo_2" in mrp.storage)

    # test get_storage() method
    def test_get_storage(self):
        # instantiate an instance of Model_Result_Populator
        mrp = Model_Result_Populator()

        # check that storage is empty
        self.assertEqual(len(mrp.storage), 0)

        # add an item
        mrp.populate_storage(the_key="foo", the_item="bar")

        # invoke the method
        self.assertEqual(mrp.get_storage(), {"foo": "bar"})

        # add another item
        mrp.populate_storage(the_key="foo_2", the_item="bar_2")

        # run assertions
        self.assertEqual(len(mrp.storage), 2)
        self.assertTrue("foo" in mrp.storage)
        self.assertTrue("foo_2" in mrp.storage)

        # invoke the method
        self.assertEqual(mrp.get_storage(), {"foo": "bar", "foo_2": "bar_2"})
