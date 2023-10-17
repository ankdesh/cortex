"""
number_feature_attributes.py
This module contains the attributes and validation functions for 
feaures with Number format
"""

from typing import Callable, Dict, Any

# Default preprocessing dictionary
preprocessing_default = {
    "missing_value_strategy": "fill_with_const",
    "normalization": "zscore",
    "outlier_strategy": "null",
    "fill_value": 0.0,
    "outlier_threshold": 3.0
}


def preprocessing_validator(key, value):
    """
    Validate a key-value pair for preprocessing settings based on the given description.

    Parameters:
    - key (str): The key representing the preprocessing setting.
    - value (any): The value to be set for the preprocessing setting.

    Returns:
    - bool: True if the key-value pair is valid, False otherwise.
    """
    # Validate 'missing_value_strategy'
    if key == "missing_value_strategy":
        options = [
            "fill_with_const", "fill_with_mode", "bfill", "ffill", "drop_row",
            "fill_with_mean"
        ]
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
        options = [
            "fill_with_const", "fill_with_mode", "bfill", "ffill", "drop_row",
            "fill_with_mean", "null"
        ]
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
        return value in [
            "elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh",
            "softmax", None
        ]
    elif key == "use_bias":
        return isinstance(value, bool)
    elif key in ["bias_initializer", "weights_initializer"]:
        return value in [
            "uniform", "normal", "constant", "ones", "zeros", "eye", "dirac",
            "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal", "orthogonal", "sparse", "identity"
        ]
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

# Dictionary for default values of the decoder
decoder_defaults = {
    'type': 'regressor',
    'num_fc_layers': 0,
    'fc_output_size': 256,
    'fc_norm': None,
    'fc_dropout': 0.0,
    'fc_activation': 'relu',
    'fc_layers': None,
    'fc_use_bias': True,
    'fc_weights_initializer': 'xavier_uniform',
    'fc_bias_initializer': 'zeros',
    'fc_norm_params': None,
    'use_bias': True,
    'weights_initializer': 'xavier_uniform',
    'bias_initializer': 'zeros'
}


def decoder_validator(key: str, value: Any) -> bool:
    """
    Validate a key-value pair for the decoder based on the provided defaults.

    Parameters:
    - key (str): The key for the decoder option.
    - value (Any): The value for the decoder option.

    Returns:
    - bool: True if the key-value pair is valid, otherwise False.
    """
    if key not in decoder_defaults:
        return False

    if key == 'type':
        return value == 'regressor'

    if key == 'num_fc_layers':
        return isinstance(value, int) and value >= 0

    if key == 'fc_output_size':
        return isinstance(value, int) and value > 0

    if key == 'fc_norm':
        return value in [None, 'batch', 'layer', 'ghost']

    if key == 'fc_dropout':
        return isinstance(value, float) and 0 <= value <= 1

    if key == 'fc_activation':
        return value in [
            'elu', 'leakyRelu', 'logSigmoid', 'relu', 'sigmoid', 'tanh',
            'softmax', None
        ]

    if key == 'fc_layers':
        return value is None or isinstance(value, list)

    if key in ['fc_use_bias', 'use_bias']:
        return isinstance(value, bool)

    if key in ['fc_weights_initializer', 'weights_initializer']:
        return value in [
            'uniform', 'normal', 'constant', 'ones', 'zeros', 'eye', 'dirac',
            'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
            'kaiming_normal', 'orthogonal', 'sparse', 'identity'
        ]

    if key in ['fc_bias_initializer', 'bias_initializer']:
        return value in [
            'uniform', 'normal', 'constant', 'ones', 'zeros', 'eye', 'dirac',
            'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
            'kaiming_normal', 'orthogonal', 'sparse', 'identity'
        ]

    if key == 'fc_norm_params':
        return value is None or isinstance(value, dict)

    return False

# Dictionary for default values of the loss types
loss_defaults = {
    'mean_squared_error': {'type': 'mean_squared_error', 'weight': 1.0},
    'mean_absolute_error': {'type': 'mean_absolute_error', 'weight': 1.0},
    'mean_absolute_percentage_error': {'type': 'mean_absolute_percentage_error', 'weight': 1.0},
    'root_mean_squared_error': {'type': 'root_mean_squared_error', 'weight': 1.0},
    'root_mean_squared_percentage_error': {'type': 'root_mean_squared_percentage_error', 'weight': 1.0},
    'huber': {'type': 'huber', 'weight': 1.0, 'delta': 1.0}
}

def weight_validator(key: str, value: Any) -> bool:
    """Validate 'weight' key for loss."""
    return key == 'weight' and isinstance(value, (float, int))

def delta_validator(key: str, value: Any) -> bool:
    """Validate 'delta' key for Huber loss."""
    return key == 'delta' and isinstance(value, (float, int))

def type_validator(key: str, value: Any) -> bool:
    """Validate 'type' key for loss."""
    return key == 'type' and value in loss_defaults.keys()

# Dictionary of functions to validate key-value pairs for each loss type
loss_validators = {
    'mean_squared_error': weight_validator,
    'mean_absolute_error': weight_validator,
    'mean_absolute_percentage_error': weight_validator,
    'root_mean_squared_error': weight_validator,
    'root_mean_squared_percentage_error': weight_validator,
    'huber': lambda k, v: weight_validator(k, v) or delta_validator(k, v) or type_validator(k, v)
}
