import unittest

from cortex.utils import set_value  
from cortex.ludwig.number_feature import NumberInputFeature, encoder_defaults, encoder_validators
from cortex.ludwig.number_feature import NumberOutputFeature
from cortex.node import Node

# Creating a unittest class for NumberInputFeature
class TestNumberInputFeature(unittest.TestCase):

    def setUp(self):
        """Initialize a NumberInputFeature instance for testing."""
        self.feature = NumberInputFeature(name="TestFeature")
    
    def test_change_encoder_type_valid(self):
        """Test change_encoder_type method for a valid encoder type."""
        result = self.feature.change_encoder_type("dense")
        self.assertTrue(result)
        
        # Test if the encoder defaults are set correctly
        self.assertEqual(self.feature._encoder, encoder_defaults["dense"])
        
        # Test if the validation function is set correctly
        self.assertEqual(self.feature._is_valid_encoder_key, encoder_validators["dense"])
    
    def test_change_encoder_type_invalid(self):
        """Test change_encoder_type method for an invalid encoder type."""
        result = self.feature.change_encoder_type("invalid_encoder")
        self.assertFalse(result)
        
    def test_set_encoder_value_valid(self):
        """Test set_encoder_value method for valid key-value pair."""
        self.feature.change_encoder_type("dense")
        
        # Test a valid key-value pair
        result = self.feature.set_encoder_value("dropout", 0.5)
        self.assertTrue(result)
        self.assertEqual(self.feature._encoder["dropout"], 0.5)
    
    def test_set_encoder_value_invalid(self):
        """Test set_encoder_value method for invalid key-value pair."""
        self.feature.change_encoder_type("dense")
        
        # Test an invalid key-value pair
        result = self.feature.set_encoder_value("dropout", "invalid_value")
        self.assertFalse(result)
        

class TestNumberOutputFeature(unittest.TestCase):

    def setUp(self):
        # Initialize NumberOutputFeature object
        self.feature = NumberOutputFeature (name='feature1')
        
    def test_set_decoder_value(self):
        """Test setting decoder value."""
        self.assertEqual(self.feature._decoder['type'], 'regressor')
        self.assertTrue(self.feature.set_decoder_value('type', 'regressor'))
        

    def test_change_loss_type(self):
        """Test setting loss value."""
        self.assertTrue(self.feature.change_loss_type('huber'))
        self.assertEqual(self.feature._loss['type'], 'huber')