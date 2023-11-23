"""
combiner_attributes.py
This module contains the attributes and validation functions for 
feaures with Number format
"""
from typing import Any

#  Dictionary for default values of the combiner
combiner_defaults = {
    'type': 'concat',
    'dropout': 0.0,
    'num_fc_layers': 0,
    'output_size': 256,
    'norm': None,
    'activation': 'relu',
    'flatten_inputs': False,
    'residual': False,
    'use_bias': True,
    'bias_initializer': 'zeros',
    'weights_initializer': 'xavier_uniform',
    'norm_params': None,
    'fc_layers': None
}


def combiner_validator(key: str, value: Any) -> bool:
    """
    Validate a key-value pair for the combiner based on the provided defaults.

    Parameters:
    - key (str): The key for the combiner option.
    - value (Any): The value for the combiner option.

    Returns:
    - bool: True if the key-value pair is valid, otherwise False.
    """
    if key not in combiner_defaults:
        return False

    if key == 'type':
        return value == 'concat'

    if key == 'dropout':
        return isinstance(value, float) and 0 <= value <= 1

    if key == 'num_fc_layers':
        return isinstance(value, int) and value >= 0

    if key == 'output_size':
        return isinstance(value, int) and value > 0

    if key == 'norm':
        return value in [None, 'batch', 'layer', 'ghost']

    if key == 'activation':
        return value in [
            'elu', 'leakyRelu', 'logSigmoid', 'relu', 'sigmoid', 'tanh',
            'softmax', None
        ]

    if key in ['flatten_inputs', 'residual', 'use_bias']:
        return isinstance(value, bool)

    if key in ['bias_initializer', 'weights_initializer']:
        return value in [
            'uniform', 'normal', 'constant', 'ones', 'zeros', 'eye', 'dirac',
            'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
            'kaiming_normal', 'orthogonal', 'sparse', 'identity'
        ] or (isinstance(value, dict) and 'type' in value)

    if key == 'norm_params':
        return value is None or isinstance(value, dict)

    if key == 'fc_layers':
        return value is None or isinstance(value, list) 

    return False
