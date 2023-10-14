"""
list_input_features_node.py
This module defines the list of input features for ludwig config
"""
import yaml
from typing import List

from cortex.node import Node 
from cortex.ludwig.input_features import LudwigInputFeatures
from cortex.ludwig.number_feature import NumberInputFeature 

class ListInputFeatures(Node):
    """
    This class implements the list of features which can be 
    emitted to ludwig config file list of input features. 
    """ 
    def __init__(self):
        super().__init__()
        self._list_input_features = []

    def add_feature(self, feature: LudwigInputFeatures) -> None:
        """Add the Input feature to list"""
        self._list_input_features.append(feature.to_dict())

    def get_list_input_features(self) -> List:
        """Returns the of input features """
        return self._list_input_features    

    def emit_entry(self) -> str:
        str_input_features = yaml.dump(self._list_input_features)
        return str_input_features

    def emit_exit(self) -> str:
        return ""