import unittest
from cortex.input_feature_ludwig import InputFeatureConfig, InputFeatureLudwig  
from cortex.root import RootNode

class TestInputFeatureConfig(unittest.TestCase):

    def setUp(self):
        self.config = InputFeatureConfig(name="text", type_feature="text")

    def test_initialization(self):
        self.assertEqual(self.config.get_config()["name"], "text")
        self.assertEqual(self.config.get_config()["type"], "text")
        self.assertNotIn("level", self.config.get_config())
        self.assertNotIn("preprocessing", self.config.get_config())
        self.assertNotIn("encoder", self.config.get_config())

    def test_set_level(self):
        self.config.set_level("word")
        self.assertEqual(self.config.get_config()["level"], "word")

    def test_set_preprocessing(self):
        self.config.set_preprocessing("space")
        self.assertEqual(self.config.get_config()["preprocessing"]["word_tokenizer"], "space")

    def test_set_encoder(self):
        self.config.set_encoder("bert", "mean", True)
        self.assertEqual(self.config.get_config()["encoder"]["type"], "bert")
        self.assertEqual(self.config.get_config()["encoder"]["reduce_output"], "mean")
        self.assertEqual(self.config.get_config()["encoder"]["trainable"], True)

# Unit tests for InputFeatureLudwig
class TestInputFeatureLudwig(unittest.TestCase):

    def setUp(self):
        # Create a fully configured InputFeatureConfig object
        self.inp_features = InputFeatureConfig(name="age", type_feature="numerical")
        self.inp_features.set_level("high")
        self.inp_features.set_preprocessing(word_tokenizer="space")
        self.inp_features.set_encoder(type_encoder="rnn", reduce_output="sum", trainable=False)

        # Create an InputFeatureLudwig object with the above config
        self.ludwig_node = InputFeatureLudwig(self.inp_features)

    def test_emit_entry(self):
        # Test if the emit_entry function returns the correct string
        expected_output = """input_features = {'encoder': {'reduce_output': 'sum', 'trainable': False, 'type': 'rnn'},
 'level': 'high',
 'name': 'age',
 'preprocessing': {'word_tokenizer': 'space'},
 'type': 'numerical'}"""
        
        self.assertEqual(self.ludwig_node.emit_entry().strip(), expected_output.strip())