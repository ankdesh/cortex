"""
input_feature_ludwig.py
This module defines a Input Feature node for ludwig based pipeline
"""


class InputFeatureConfig:
    """Class to create and manage the configuration for input features.
    
    This class allows setting of values for a specific configuration dict
    and provides a function to return the configured dict.
    """

    def __init__(self, name: str, type_feature: str):
        """Initialize the mandatory fields.
        
        Args:
            name (str): Name of the feature.
            type_feature (str): Type of the feature.
        """
        self._config = {
            "name": name,
            "type": type_feature
        }

    def set_level(self, level: str):
        """Set the level for the feature.

        Args:
            level (str): Level of the feature.
        """
        self._config["level"] = level

    def set_preprocessing(self, word_tokenizer: str):
        """Set the preprocessing details.

        Args:
            word_tokenizer (str): Tokenizer type for word preprocessing.
        """
        if "preprocessing" not in self._config:
            self._config["preprocessing"] = {}
        self._config["preprocessing"]["word_tokenizer"] = word_tokenizer

    def set_encoder(self, type_encoder: str, reduce_output=None, trainable=True):
        """Set the encoder details.
        
        Args:
            type_encoder (str): Type of encoder.
            reduce_output (Optional): Reduction type for encoder output.
            trainable (bool, default=True): If encoder is trainable.
        """
        if "encoder" not in self._config:
            self._config["encoder"] = {}
        self._config["encoder"]["type"] = type_encoder
        self._config["encoder"]["reduce_output"] = reduce_output
        self._config["encoder"]["trainable"] = trainable

    def get_config(self) -> dict:
        """Get the configured dictionary.
        
        Returns:
            dict: The configured dictionary.
        """
        return self._config
