import unittest

from cortex.utils import set_value
from cortex.ludwig.number_feature import NumberInputFeature, encoder_defaults, encoder_validators
from cortex.ludwig.number_feature import NumberOutputFeature
from cortex.ludwig.list_input_features_node import ListInputFeatures
from cortex.ludwig.list_output_features_node import ListOutputFeatures
from cortex.utils import dict_to_yaml_file
#from ludwig.api import LudwigModel

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


class GenerateSimpleTest(unittest.TestCase):

    def setUp(self):
        # Create a input feature list node and an ouput feature list node
        # for flowers dataset

        self.ip_feature = NumberInputFeature("sepal_length_cm")
        self.ip_feature.change_encoder_type("dense")
        self.ip_feature.set_encoder_value("dropout", 0.75)

        self.op_feature = NumberOutputFeature("petal_width_cm")
        self.op_feature.set_decoder_value("dropout", 0.75)

    def test_generated_config(self):
        """Generate config file for flower dataset and train using ludwig"""

        list_input_features = ListInputFeatures()
        list_input_features.add_feature(self.ip_feature)

        list_output_features = ListOutputFeatures()
        list_output_features.add_feature(self.op_feature)

        config_list = {"input_features":list_input_features.get_list_input_features(),
                       "output_features":list_output_features.get_list_output_features()}

        dict_to_yaml_file (config_list,"cortex/generated/config_petal.yaml")
        
        # TODO : Enable it ?
        # model = LudwigModel(config="/tmp/config_petal.yaml")
        #train_stats, _, _ = model.train(dataset="tests/ludwig/sample_inputs/dataset.csv")

        # If no exception was made till now, test passes
        self.assertTrue(True)
