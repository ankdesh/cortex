"""
model_size_recipe.py 
This module defines a simple recipe of finding right number of layers
in combiner.
"""

from prime_config_gen import get_inp_features, get_out_features, get_combiner
from cortex.utils import dict_to_yaml_file

def gen_prime_config ():

    for i in range(5):
        combiner = get_combiner()
        combiner.set_value("num_fc_layers",i) 

        config_list = {"input_features":get_inp_features().get_list_input_features(),
                        "output_features":get_out_features().get_list_output_features(),
                        "combiner": combiner.get_combiner_values_dict()}

        dict_to_yaml_file (config_list,f"cortex/generated/config_prime_{i}.yaml")


if __name__ == "__main__":
    gen_prime_config()
