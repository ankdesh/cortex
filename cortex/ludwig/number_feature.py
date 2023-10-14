"""
number_feature.py
This module defines a Input and Output Feature for Number format
"""

import pprint
from cortex.ludwig.input_features import LudwigInputFeatures

from abc import ABC, abstractmethod

class NumberInputFeature(LudwigInputFeatures):
    """
    This class implements the node for input Number Feature for ludwig pipeline
    """

    def __init__(self,name):
        """Initialize both parent classes."""
        LudwigInputFeatures.__init__(self, name=name,
                                     type_= "number"
                                     )
        self._preprocessing = number_input_preprocessing_default
        self._is_valid_preprocessing_key = number_input_validate_preprocessing
        self._encoder = encoder_defaults["passthrough"]
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

# Default preprocessing dictionary
number_input_preprocessing_default = {
    "missing_value_strategy": "fill_with_const",
    "normalization": "zscore",
    "outlier_strategy": "null",
    "fill_value": 0.0,
    "outlier_threshold": 3.0
}

def number_input_validate_preprocessing(key, value):
    """
    Validate a key-value pair for preprocessing settings based on the given description.
    
    Parameters:
    - key (str): The key representing the preprocessing setting.
    - value (any): The value to be set for the preprocessing setting.
    
    Returns:
    - bool: True if the key-value pair is valid, False otherwise.
    - str: A message indicating the result of the validation.
    """
    # Validate 'missing_value_strategy'
    if key == "missing_value_strategy":
        options = ["fill_with_const", "fill_with_mode", "bfill", "ffill", "drop_row", "fill_with_mean"]
        if value in options:
            return True
        else:
            return False
    
    # Validate 'normalization'
    elif key == "normalization":
        options = ["zscore", "minmax", "log1p", "iq", "null"]
        if value in options:
            return True
        else:
            return False
    
    # Validate 'outlier_strategy'
    elif key == "outlier_strategy":
        options = ["fill_with_const", "fill_with_mode", "bfill", "ffill", "drop_row", "fill_with_mean", "null"]
        if value in options:
            return True
        else:
            return False
    
    # Validate 'fill_value'
    elif key == "fill_value":
        if isinstance(value, (int, float)):
            return True
        else:
            return False
    
    # Validate 'outlier_threshold'
    elif key == "outlier_threshold":
        if isinstance(value, (int, float)):
            return True
        else:
            return False
    
    else:
        return False


# Default encoder dictionaries
encoder_defaults = {
    "passthrough": {
        "type": "passthrough"
    },
    "dense": {
        "type": "dense",
        "dropout": 0.0,
        "output_size": 256,
        "norm": None,
        "num_layers": 1,
        "activation": "relu",
        "use_bias": True,
        "bias_initializer": "zeros",
        "weights_initializer": "xavier_uniform",
        "norm_params": None,
        "fc_layers": None
    }
}

# Validation functions
def validate_passthrough(key, value):
    """Always return True as passthrough encoder has no additional parameters."""
    return True

def validate_dense(key, value):
    """Validate key-value pair for dense encoder."""
    if key in ["dropout", "output_size", "num_layers"]:
        return isinstance(value, (int, float))
    elif key == "norm":
        return value in [None, "batch", "layer", "ghost"]
    elif key == "activation":
        return value in ["elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh", "softmax", None]
    elif key == "use_bias":
        return isinstance(value, bool)
    elif key in ["bias_initializer", "weights_initializer"]:
        return value in ["uniform", "normal", "constant", "ones", "zeros", "eye", "dirac", 
                         "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal",
                         "orthogonal", "sparse", "identity"]
    elif key == "norm_params":
        return True  # Assuming it's a dictionary or None, add more checks if needed
    elif key == "fc_layers":
        return True  # Assuming it's a list of dictionaries or None, add more checks if needed
    else:
        return False

# Dictionary of validation functions
encoder_validators = {
    "passthrough": validate_passthrough,
    "dense": validate_dense
}
