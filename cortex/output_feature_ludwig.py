"""
output_feature_ludwig.py
This module defines an Output Feature node for ludwig based pipeline
"""

import pprint
from cortex.node import Node  # Replace this with your actual import for Node

class OutputFeatureConfig:
    """Class to create and manage the configuration for output features.
    
    This class allows setting of values for a specific configuration dict
    and provides a function to return the configured dict.
    """

    def __init__(self, name: str, type_feature: str):
        """Initialize the mandatory fields.
        
        Args:
            name (str): Name of the feature.
            type_feature (str): Type of the feature.
        """
        self._config = {
            "name": name,
            "type": type_feature
        }

    def set_decoder(self, num_fc_layers: int, output_size: int):
        """Set the decoder details for the feature.

        Args:
            num_fc_layers (int): Number of fully connected layers.
            output_size (int): Output size.
        """
        self._config["decoder"] = {
            "num_fc_layers": num_fc_layers,
            "output_size": output_size
        }

    def set_dependencies(self, dependencies: list):
        """Set the dependencies for the feature.

        Args:
            dependencies (list): List of dependencies.
        """
        self._config["dependencies"] = dependencies

    def get_config(self) -> dict:
        """Get the configured dictionary.
        
        Returns:
            dict: The configured dictionary.
        """
        return self._config


class OutputFeatureLudwig(Node):
    """Represents an OutputFeatureLudwig node in AST"""

    def __init__(self, out_features: OutputFeatureConfig):
        """Initialize the OutputFeatureLudwig node.
        
        Args:
            out_features (OutputFeatureConfig): Configuration for the output features.
        """
        super().__init__()  # Call parent class's __init__ method
        self._out_features = out_features
        self._code_template = "output_features = " + pprint.pformat(out_features.get_config())

    def emit_entry(self) -> str:
        """Implementation of abstract method from parent class."""
        return self._indent_space() + self._code_template + "\n"

    def emit_exit(self) -> str:
        """Implementation of abstract method from parent class."""
        return ""
