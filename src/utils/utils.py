"""
Just some utility functions
"""

import yaml
import csv

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

class MetricsCollector:
    """A class to collect metrics during training"""
    def __init__(self):
        self.metrics = []

    def add_metric(self, value: dict):
        """Adds a metric to the collector"""
        self.metrics.append(value)

    def get_metrics(self) -> dict:
        """Returns the collected metrics"""
        return self.metrics
    
    def save_metrics(self, where: str):
        """Saves the collected metrics to a csv file"""
        with open(where, 'w') as f:
            
            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)