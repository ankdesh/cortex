"""
list_input_features_node.py
This module defines the list of input features for ludwig config
"""
import yaml
from typing import List

from cortex.node import Node 
from cortex.ludwig.output_features import LudwigOutputFeatures

class ListOutputFeatures(Node):
    """
    This class implements the list of features which can be 
    emitted to ludwig config file list of input features. 
    """ 
    def __init__(self):
        super().__init__()
        self._list_output_features = []

    def add_feature(self, feature: LudwigOutputFeatures) -> None:
        """Add the Input feature to list"""
        self._list_output_features.append(feature.to_dict())

    def get_list_output_features(self) -> List:
        """Returns the of input features """
        return self._list_output_features    

    def emit_entry(self) -> str:
        str_output_features = yaml.dump(self._list_output_features)
        return str_output_features

    def emit_exit(self) -> str:
        return ""