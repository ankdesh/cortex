"""
combiner.py
This module defines combiner for ludwig config
"""
import yaml
from typing import List
from typing import Callable, Dict, Any
from copy import deepcopy

from cortex.utils import set_value
from cortex.node import Node
from cortex.ludwig.combiner_attributes import combiner_defaults, combiner_validator



class Combiner(Node):
    """
    This class implements the combiner  which can be 
    emitted to ludwig config file list of input features. 
    """

    def __init__(self):
        super().__init__()
        self._combiner_values = deepcopy(combiner_defaults)
        self._is_valid_combiner_key = combiner_validator

    def set_value(self, key: str, value: Any) -> bool:
        """Set a combiner key-value pair if valid."""
        if self._is_valid_combiner_key(key, value):
            return set_value(self._combiner_values, key, value)
        return False

    def get_combiner_values_dict(self) -> Dict:
        """Return the combiner values""" 
        return self._combiner_values

    def emit_entry(self) -> str:
        str_combiner = yaml.dump(self._combiner_values)
        return str_combiner

    def emit_exit(self) -> str:
        return ""
