"""
ludwig_features.py
This module defines the base class for input, output Feature nodes for ludwig based pipeline
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
from cortex.utils import set_value

class LudwigInputFeatures(ABC):
    """
    Abstract Base Class for Ludwig Input Features.

    Attributes:
    - _name (str): Name of the feature
    - _type (str): Type of the feature
    - _preprocessing (Dict): Dictionary for preprocessing options
    - _encoder (Dict): Dictionary for encoder options
    """

    def __init__(self,
                 name: str,
                 type_: str,
                 preprocessing: Dict[str, Any],
                 encoder: Dict[str, Any],
                 is_valid_preprocessing_key: Callable[[str, Any], bool],
                 is_valid_encoder_key: Callable[[str, Any], bool]
                 ) -> None:
        """Initialize LudwigInputFeatures with given parameters."""
        self._name = name
        self._type = type_
        self._preprocessing = preprocessing
        self._encoder = encoder
        self._is_valid_preprocessing_key = is_valid_preprocessing_key
        self._is_valid_encoder_key = is_valid_encoder_key

    def set_preprocessing_value(self, key: str, value: Any) -> bool:
        """Set a preprocessing key-value pair if valid."""
        if self._is_valid_preprocessing_key(key, value):
            return set_value(self._preprocessing, key, value)
        return False

    def set_encoder_value(self, key: str, value: Any) -> bool:
        """Set an encoder key-value pair if valid."""
        if self._is_valid_encoder_key(key, value):
            return set_value(self._encoder, key, value)
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the object."""
        return {
            'name': self._name,
            'type': self._type,
            'preprocessing': self._preprocessing,
            'encoder': self._encoder
        }
