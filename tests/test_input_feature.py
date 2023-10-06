import unittest
from cortex.input_feature_ludwig import InputFeatureConfig  

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

