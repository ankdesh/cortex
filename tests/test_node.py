# test_node.py
import unittest
from cortex.node import Node  

class Parantheses(Node):
    # Implements a simple language of braces "{}" for testing 
    # the implementation of Node

    def _indent_space(self) -> str:
        return ' ' * (self._depth * Node.indent_step)

    def emit_entry(self) -> str:
        return f"{self._indent_space()}{{\n"

    def emit_exit(self) -> str:
        return f"{self._indent_space()}}}\n"
    
class TestNode(unittest.TestCase):
    def test(self):
        """
        Test function1 from module1.
        """
        nodes = Parantheses()
        
        nodes.add_child(Parantheses()).add_child(Parantheses())
        nodes._children[1].add_child(Parantheses()).add_child(Parantheses())
        nodes._children[1]._children[1].add_child(Parantheses()) 

        right_result = """{
    {
    }
    {
        {
        }
        {
            {
            }
        }
    }
}
"""

        assert (nodes.emit_depth_first_order() == right_result)

        pass  

