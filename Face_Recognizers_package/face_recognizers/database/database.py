""" Main Database wrapper module

    @author: Hermas
    @copyright: Tahaluf UAE 2023
"""
from typing import Any
import logging
import numpy as np
import yaml

from face_recognizers.database.basic import basic_db


class DataBase:
    """Database class for face recognition models"""

    def __init__(self, configs, db_name: str = "basic") -> None:
        self.db_name = db_name
        self.logger = logging.getLogger(__name__)
        self._load_configs(configs)
        self._db_obj = self._load_database_obj()

    def _load_configs(self, configs: Any) -> dict:
        """load configs from yaml file

        Args:
            config_path (any): path to yaml file or dict
        """
        if isinstance(configs, dict):
            self.logger.debug("LOADING CONFIGS FROM DICT ..")
            self.configs = configs
            self.logger.info("CONFIGS LOADED")
        elif (isinstance(configs, str)) and (configs.endswith(".yaml")):
            with open(configs, encoding="utf-8") as file:
                self.logger.debug("LOADING CONFIGS FROM FILE ..")
                try:
                    self.configs = yaml.safe_load(file)[self.db_name]
                    self.logger.info("CONFIGS LOADED")
                except (ValueError, KeyError) as exc:
                    raise ValueError(" WRONG MODEL CONFIG NAME!") from exc
        else:
            raise ValueError("CONFIGS MUST BE DICT OR YAML FILE")

    def _load_database_obj(self) -> object:
        """Loads the database object"""
        if self.db_name == "basic":
            return basic_db.DataBase(self.logger, self.configs)
        raise NotImplementedError

    def generate_database(self, model) -> None:
        """generates & saves subjects face embeddings"""
        self._db_obj.generate_database(model)

    def load_database(self) -> None:
        """loads subjects face embeddings"""
        self._db_obj.load_database()

    def get_id(self, feats: np.ndarray, thresh: float) -> str:
        """get face id from his embedding"""
        return self._db_obj.get_id(feats, thresh)
