from cortex.ludwig.list_output_features_node import ListOutputFeatures
from cortex.ludwig.number_feature import NumberOutputFeature
from cortex.utils import dict_to_yaml_file

import unittest
import yaml


class TestListOutputFeatures(unittest.TestCase):
    """
    Unit tests for ListInputFeatures class.
    """
    
    def test_number_features(self):
        """
        A helper function to create two input features 
        """
        feature0 = NumberOutputFeature("feature0")
        feature1 = NumberOutputFeature("feature1")

        # Modify some stuff in ip_feature1
        feature1.change_loss_type("huber")
        feature1.set_decoder_value("dropout", 0.75)

        # Create Number feature list
        list_output_features = ListOutputFeatures()
        list_output_features.add_feature(feature0)
        list_output_features.add_feature(feature1)

        write_file = list_output_features.get_list_output_features()
        dict_to_yaml_file (write_file,"/tmp/out.yaml")
        
        read_input_features = None
        with open("/tmp/out.yaml") as read_file: 
            read_input_features = yaml.safe_load(read_file)

        self.assertTrue (len(read_input_features) > 0)
    