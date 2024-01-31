import csv
import numpy as np
import os
from typing import Any
import yaml

Array= np.array

def read_embd(csv_file)-> Array:
    '''
    Output:
        ids: Array, database ids .
        Embeddings: Array, database embeddings.
    '''
    ids= []
    embeddings= []
    with open(csv_file, mode='r') as file:
        reader= csv.reader(file)
        for row in reader:
            embedding= row[:-1]
            ids.append(row[-1])
            embeddings.append(embedding)
    return np.array(ids), np.array(embeddings)


def load_configs(configs: Any) -> dict:
        """load configs from yaml file

        Args:
            config_path (any): path to yaml file or dict
        """
       
        if (isinstance(configs, str)) and (configs.endswith(".yaml")):
            with open(
                os.path.join(
                    os.path.dirname(__file__),
                    configs,
                ),
                encoding="utf-8",
            ) as file:
                try:
                    configs = yaml.safe_load(file)["school"]
                except (ValueError, KeyError) as exc:
                    raise ValueError(" WRONG MODEL CONFIG NAME!") from exc
        else:
            raise ValueError("CONFIGS MUST BE DICT OR YAML FILE")
        
        return configs