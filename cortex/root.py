"""
root_node.py
This module defines the node to be used as root node and generate general 
text expected in a file including Copyright notice
TODO: Empty for now
"""

from cortex.node import Node

# Root Node implementing the Node
class RootNode(Node):
    """Implementation of the Node class for root node."""

    def emit_entry(self) -> str:
        """To include python file headers.
        Empty for now"""
        return "\n"

    def emit_exit(self) -> str:
        """Implementation of abstract method from parent class."""
        return ""
