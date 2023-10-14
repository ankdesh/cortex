import unittest
from unittest.mock import Mock
from cortex.ludwig.ludwig_input_features import LudwigInputFeatures
from cortex.utils import set_value  

class TestLudwigInputFeatures(unittest.TestCase):

    def setUp(self):
        # Mock validation functions
        self.is_valid_preprocessing_key = Mock(return_value=True)
        self.is_valid_encoder_key = Mock(return_value=True)

        # Initialize LudwigInputFeatures object
        self.feature = LudwigInputFeatures(
            name='feature1',
            type_='numerical',
            preprocessing={'norm': 'z-score'},
            encoder={'type': 'dense'},
            is_valid_preprocessing_key=self.is_valid_preprocessing_key,
            is_valid_encoder_key=self.is_valid_encoder_key
        )

    def test_initialization(self):
        """Test if initialization works correctly."""
        self.assertEqual(self.feature._name, 'feature1')
        self.assertEqual(self.feature._type, 'numerical')
        self.assertEqual(self.feature._preprocessing, {'norm': 'z-score'})
        self.assertEqual(self.feature._encoder, {'type': 'dense'})

    def test_set_preprocessing_value(self):
        """Test setting preprocessing value."""
        self.assertTrue(self.feature.set_preprocessing_value('norm', 'min-max'))
        self.assertEqual(self.feature._preprocessing['norm'], 'min-max')

    def test_set_encoder_value(self):
        """Test setting encoder value."""
        self.assertTrue(self.feature.set_encoder_value('type', 'conv'))
        self.assertEqual(self.feature._encoder['type'], 'conv')

    def test_to_dict(self):
        """Test to_dict method."""
        expected_dict = {
            'name': 'feature1',
            'type': 'numerical',
            'preprocessing': {'norm': 'z-score'},
            'encoder': {'type': 'dense'}
        }
        self.assertEqual(self.feature.to_dict(), expected_dict)