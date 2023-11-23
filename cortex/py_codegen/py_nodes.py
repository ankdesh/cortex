"""
py_nodes.py
This module defines the node for Python code generation
"""

from cortex.node import Node  

class Py_Statement(Node):
    # Implements the Python statements 
    def __init__(self, statement: str):
        super().__init__()
        self._statement = statement
    
    def emit_entry(self) -> str:
        return self._indent_space() + self._statement + "\n" 

    def emit_exit(self) -> str:
        return ""


class Py_loop(Node):
    # Implements a Python loop 
    def __init__(self, iterator: str,  iterable: str):
        super().__init__()
        self._iterator = iterator
        self._iterable = iterable

    def emit_entry(self) -> str:
        return f"{self._indent_space()}for {self._iterator} in {self._iterable}:" + "\n"

    def emit_exit(self) -> str:
        return ""