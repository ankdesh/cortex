import yaml


def set_value(d, key, value):
    """
    Set the value of a key in a nested dictionary, irrespective of its depth.

    Parameters:
    - d (dict): The nested dictionary
    - key (str): The key to set
    - value (Any): The value to set for the key

    Returns:
    - bool: True if the key was found and set, False otherwise
    """
    # Check if the key exists at the current depth
    if key in d:
        d[key] = value
        return True

    # If the key doesn't exist at the current depth, go deeper
    for k in d:
        if isinstance(d[k], dict):
            if set_value(d[k], key, value):
                return True

    # If the key was not found at any depth
    return False


def dict_to_yaml_file(obj, file_path):
    """
    Dump a python obj to a YAML file.

    Parameters:
    - Python object (inp): The object to convert
    - file_path (str): The path where the YAML file will be saved

    Returns:
    - None: The function writes the YAML file to the disk
    """
    # Open the file in write mode
    with open(file_path, 'w') as yaml_file:
        # Dump the obj to the YAML file
        yaml.dump(obj, yaml_file)
