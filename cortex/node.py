"""
node.py
This module defines a base Node to be used in representing AST. Traversal of AST
and emiting code will lead to generation of entire pipeline
"""

from abc import ABC, abstractmethod

class Node(ABC):
    """Represents a Node in AST"""

    indent_step = 4  # num spaces for each indentation level
    
    def __init__(self):
        self.depth = 0
        self.children = []

    def add_child(self, node: 'Node') -> 'Node':
        """
        Adds the node to children list and sets the appropriate depth of child 
        node. Returns the current node for chaining.
        """
        assert isinstance(node, Node)
        self.children.append(node)
        node.depth = self.depth + 1 
        return self

    def _indent_space(self) -> str:
        """Returns a string of spaces for indentation."""
        return ' ' * (self.depth * Node.indent_step)

    @abstractmethod
    def emit_entry(self) -> str:
        """Returns the generated code for entry to this block as a string."""
        pass

    @abstractmethod
    def emit_exit(self) -> str:
        """Returns the generated code for exit from this block as a string."""
        pass

    def emit_depth_first_order(self) -> str:
        """
        Visits the AST tree with this node as root and returns the code
        generated by depth-first order traversal.
        """
        code = ''
        for child in self.children:
            code += child.emit_entry()
            code += child.emit_depth_first_order()
            code += child.emit_exit()
        return code

