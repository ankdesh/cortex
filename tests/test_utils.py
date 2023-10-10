# Importing the unittest module to create test cases
import unittest
from cortex.utils import set_value

# Create a test class inheriting from unittest.TestCase
class TestSetNestedKeyValue(unittest.TestCase):

    def test_depth_one(self):
        """Test updating a key at depth 1"""
        d = {'a': 1, 'b': 2}
        self.assertTrue(set_value(d, 'a', 10))
        self.assertEqual(d['a'], 10)

    def test_depth_two(self):
        """Test updating a key at depth 2"""
        d = {'a': 1, 'b': {'c': 2}}
        self.assertTrue(set_value(d, 'c', 20))
        self.assertEqual(d['b']['c'], 20)

    def test_depth_three(self):
        """Test updating a key at depth 3"""
        d = {'a': 1, 'b': {'c': {'d': 2}}}
        self.assertTrue(set_value(d, 'd', 20))
        self.assertEqual(d['b']['c']['d'], 20)

    def test_non_existent_key(self):
        """Test updating a non-existent key"""
        d = {'a': 1, 'b': 2}
        self.assertFalse(set_value(d, 'c', 10))

    def test_empty_dict(self):
        """Test updating a key in an empty dictionary"""
        d = {}
        self.assertFalse(set_value(d, 'a', 10))

# Execute the unit tests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestSetNestedKeyValue))
