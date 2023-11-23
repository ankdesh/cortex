from cortex.ludwig.list_input_features_node import ListInputFeatures
from cortex.ludwig.number_feature import NumberInputFeature
from cortex.utils import dict_to_yaml_file

import unittest
import yaml


class TestListInputFeatures(unittest.TestCase):
    """
    Unit tests for ListInputFeatures class.
    """
    
    def test_number_features(self):
        """
        A helper function to create two input features 
        """
        ip_feature0 = NumberInputFeature("ip_feature0")
        ip_feature1 = NumberInputFeature("ip_feature1")

        # Modify some stuff in ip_feature1
        ip_feature1.change_encoder_type("dense")
        ip_feature1.set_encoder_value("dropout", 0.75)
        ip_feature1.set_preprocessing_value("normalization","minmax")

        # Create Number feature list
        list_input_features = ListInputFeatures()
        list_input_features.add_feature(ip_feature0)
        list_input_features.add_feature(ip_feature1)

        write_file = list_input_features.get_list_input_features()
        dict_to_yaml_file (write_file,"/tmp/in.yaml")
        
        read_input_features = None
        with open("/tmp/in.yaml") as read_file: 
            read_input_features = yaml.safe_load(read_file)

        self.assertTrue (len(read_input_features) > 0)
    