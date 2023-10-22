import unittest

from cortex.ludwig.input_features import LudwigInputFeatures

class TestLudwigInputFeatures(unittest.TestCase):

    def setUp(self):
        """Initialize an instance for testing."""
        self.feature = LudwigInputFeatures(name="TestFeature", type_="number")
        
    def test_init(self):
        """Test the __init__ method."""
        self.assertEqual(self.feature._name, "TestFeature")
        self.assertEqual(self.feature._type, "number")
        self.assertIsNone(self.feature._preprocessing)
        self.assertIsNone(self.feature._encoder)