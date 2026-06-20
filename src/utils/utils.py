"""
Just some utility functions
"""

import yaml

def flat_config(config : dict) -> dict:
    """Flattens a nested config file"""
    flat_config = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                flat_config[key] = value
        else:
            flat_config[section] = params

    return flat_config