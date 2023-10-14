"""
number_feature.py
This module defines a Input and Output Feature for Number format
"""

import pprint
from cortex.node import Node
from cortex.ludwig.ludwig_input_features import LudwigInputFeatures

from abc import ABC, abstractmethod

class NumberInputFeature(LudwigInputFeatures, Node):
    """
    Class that inherits from LudwigFeatures and Node.
    
    This class represents a node that can have both features from LudwigFeatures
    and the properties of an AST Node. It's named NumberFeature to represent its 
    utility for number-based features in an AST.
    """

    def __init__(self):
        """Initialize both parent classes."""
        LudwigInputFeatures.__init__(self)
        Node.__init__(self)

    def _is_valid(self, key, value) -> bool:
        pass

    def emit_entry(self) -> str:
        """
        Generate code for the entry to this block.
        
        :return: Code as a string for the entry to this block.
        """
        # Example implementation for NumberFeature's entry
        return self._indent_space() + "<Entering NumberFeature>\n"

    def emit_exit(self) -> str:
        """
        Generate code for the exit from this block.
        
        :return: Code as a string for the exit from this block.
        """
        # Example implementation for NumberFeature's exit
        return self._indent_space() + "<Exiting NumberFeature>\n"

