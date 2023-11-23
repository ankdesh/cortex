import unittest

from cortex.py_codegen import py_nodes

class TestPyLoopCodeGen(unittest.TestCase):
    def test(self):
        """
        Test python loop codegen
        """
        loop_node = py_nodes.Py_loop(iterator="x", iterable="range(5)")
        
        statement_node = py_nodes.Py_Statement("print (x)")

        loop_node.add_child(statement_node)

        res = """for x in range(5):
    print (x)
"""
                
        self.assertEqual(res, loop_node.emit_depth_first_order())
