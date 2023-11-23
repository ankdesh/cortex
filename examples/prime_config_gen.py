"""
prime_config_gen.py
This module generates the config for prime dataset and trains it.
"""

from typing import List, Dict
from cortex.ludwig.list_input_features_node import ListInputFeatures
from cortex.ludwig.list_output_features_node import ListOutputFeatures
from cortex.ludwig.number_feature import NumberInputFeature, NumberOutputFeature
from cortex.ludwig.combiner import Combiner
from cortex.utils import dict_to_yaml_file

def get_inp_features() -> List:
    list_input_features = ListInputFeatures()
    for inp_feature in ['param_1', 'param_10', 'param_2', 'param_3', 'param_4', 'param_5', 'param_6', 'param_7', 'param_8', 'param_9']:
        ip_feature = NumberInputFeature(inp_feature)
        list_input_features.add_feature(ip_feature)

    return list_input_features.get_list_input_features()

def get_out_features() -> List:
    list_output_features = ListOutputFeatures()
    for out_feature in ['runtime']:
        ip_feature = NumberOutputFeature(out_feature)
        list_output_features.add_feature(ip_feature)

    return list_output_features.get_list_output_features()

def get_combiner() -> Dict:
    combiner = Combiner()
    combiner.set_value("num_fc_layers", 5)
    return combiner.get_combiner_values_dict()

if __name__ == "__main__":

    config_list = {"input_features":get_inp_features(),
                    "output_features":get_out_features(),
                    "combiner": get_combiner()}

    dict_to_yaml_file (config_list,"cortex/generated/config_prime.yaml")
    
    # TODO : Enable it ?
    # model = LudwigModel(config="/tmp/config_petal.yaml")
    #train_stats, _, _ = model.train(dataset="tests/ludwig/sample_inputs/dataset.csv")

    # If no exception was made till now, test passes
