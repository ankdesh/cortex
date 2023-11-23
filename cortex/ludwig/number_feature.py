"""
number_feature.py
This module defines a Input and Output Feature for Number format
"""

from cortex.ludwig.input_features import LudwigInputFeatures
from cortex.ludwig.output_features import LudwigOutputFeatures
from cortex.ludwig.number_feature_attributes import encoder_defaults, encoder_validators
from cortex.ludwig.number_feature_attributes import preprocessing_default, preprocessing_validator
from cortex.ludwig.number_feature_attributes import decoder_validator, decoder_defaults, loss_defaults, loss_validators

from abc import ABC, abstractmethod
from copy import deepcopy


class NumberInputFeature(LudwigInputFeatures):
    """
    This class implements the node for input Number Feature for ludwig pipeline
    """

    def __init__(self, name):
        """Init"""
        LudwigInputFeatures.__init__(self, name=name, type_="number")
        self._preprocessing = deepcopy(preprocessing_default)
        self._is_valid_preprocessing_key = preprocessing_validator
        self._encoder = deepcopy(encoder_defaults["passthrough"])
        self._is_valid_encoder_key = encoder_validators["passthrough"]

    def change_encoder_type(self, encoder_type: str) -> bool:
        """
        Change the encoder type and corresponding parameters
        """
        if encoder_type in ["passthrough", "dense"]:
            self._encoder = encoder_defaults[encoder_type]
            self._is_valid_encoder_key = encoder_validators[encoder_type]
            return True
        else:
            return False


class NumberOutputFeature(LudwigOutputFeatures):
    """
    This class implements the node for output Number Feature for Ludwig pipeline.
    """

    def __init__(self, name: str) -> None:
        """
        Parameters:
        - name (str): The name of the feature.
        """
        super().__init__(name=name, type_="number")

        self._loss = deepcopy(loss_defaults["mean_squared_error"])
        self._is_valid_loss_key = loss_validators["mean_squared_error"]

        self._decoder = deepcopy(decoder_defaults)
        self._is_valid_decoder_key = decoder_validator

    def change_loss_type(self, loss_type: str) -> bool:
        """
        Change the loss type and corresponding parameters.

        Parameters:
        - loss_type (str): The type of the loss.

        Returns:
        - bool: True if the decoder type is successfully changed, otherwise False.
        """
        if loss_type in [
                'mean_squared_error', 'mean_absolute_error',
                'mean_absolute_percentage_error', 'root_mean_squared_error',
                'root_mean_squared_percentage_error', 'huber'
        ]:
            # Replace with your decoder options
            self._loss = loss_defaults[loss_type]
            # Replace with your decoder validator
            self._is_valid_loss_key = loss_defaults[loss_type]
            return True
        else:
            return False
