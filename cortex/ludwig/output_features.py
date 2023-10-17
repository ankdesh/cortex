"""
output_features.py
This module defines the base class for output feature nodes for ludwig based pipeline
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
from cortex.utils import set_value


class LudwigOutputFeatures(ABC):
    """
    Abstract Base Class for Ludwig Output Features.

    Attributes:
    - _name (str): Name of the feature
    - _type (str): Type of the feature
    - _loss (Dict): Dictionary for loss options
    - _decoder (Dict): Dictionary for decoder options
    """

    def __init__(self,
                 name: str,
                 type_: str
                 ) -> None:
        """
        Initialize LudwigOutputFeatures with given parameters.

        Parameters:
        - name (str): The name of the feature.
        - type_ (str): The type of the feature.
        """
        self._name = name
        self._type = type_
        
        self._loss = None  # To be updated based on loss options
        self._is_valid_loss_key = None
        
        self._decoder = None  # To be updated based on decoder_type
        self._is_valid_decoder_key = None

    def set_loss_value(self, key: str, value: Any) -> bool:
        """
        Set a loss key-value pair if valid.

        Parameters:
        - key (str): The key for the loss option.
        - value (Any): The value for the loss option.

        Returns:
        - bool: True if the key-value pair is valid, otherwise False.
        """
        if self._is_valid_loss_key(key, value):
            return set_value(self._loss, key, value)
        return False

    def set_decoder_value(self, key: str, value: Any) -> bool:
        """
        Set a decoder key-value pair if valid.

        Parameters:
        - key (str): The key for the decoder option.
        - value (Any): The value for the decoder option.

        Returns:
        - bool: True if the key-value pair is valid, otherwise False.
        """
        if self._is_valid_decoder_key(key, value):
            return set_value(self._decoder, key, value)
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the object.

        Returns:
        - Dict[str, Any]: Dictionary representation of the object.
        """
        return {
            'name': self._name,
            'type': self._type,
            'loss': self._loss,
            'decoder': self._decoder
        }
