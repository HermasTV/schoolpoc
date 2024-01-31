"""
Recognition models loader

@authors:   Mahmoud Ewaisha
            Hermas
            Anwar Alsheikh
@lisence: Tahaluf 2023
"""
import os
from typing import Any, List, Tuple, Dict,Union
import logging
import yaml
import numpy as np
import mlflow
from botocore.exceptions import ClientError


from face_recognizers.models.arcface_onnx import arcface
from face_recognizers.models.adaface_onnx import adaface
from face_recognizers.models.mobilefacenet_onnx import mobilefacenet


class Recognizer:
    """Wrapper class for recognition model.
    Although having it does not make sense with an ABC"""

    def __init__(self, model_name: str, configs: str = "./configs_fr.yaml") -> None:
        self.model_name = model_name
        self.logger = logging.getLogger("FR package")

        self._load_configs(configs)
        self._download_mlflow_model()
        self._load_model(model_name)

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
            with open(
                os.path.join(
                    os.path.dirname(__file__),
                    configs,
                ),
                encoding="utf-8",
            ) as file:
                self.logger.debug("LOADING CONFIGS FROM FILE ..")
                try:
                    self.configs = yaml.safe_load(file)[self.model_name]
                    self.logger.info("CONFIGS LOADED")
                except (ValueError, KeyError) as exc:
                    raise ValueError(" WRONG MODEL CONFIG NAME!") from exc
        else:
            raise ValueError("CONFIGS MUST BE DICT OR YAML FILE")

    def _set_env_vars(self, configs: dict) -> None:
        """set env vars

        Args:
            configs (dict): configs dict
        """
        self.logger.debug("SETTING ENV VARS")
        for key, value in configs.items():
            os.environ[key] = value

    def _download_mlflow_model(self) -> None:
        """download mlflow model"""
        # create model directory if not exists
        self.logger.info("DOWNLOADING %s FROM MLFLOW", self.model_name)
        # check if onnx model file exsists
        if os.path.exists(f"./{self.model_name}/model.onnx"):
            self.logger.info("MODEL ALREADY EXISTS")
            self.configs["model"] = f"./{self.model_name}/model.onnx"
            return
        os.makedirs(self.model_name, exist_ok=True)
        # load mlflow configs relative
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "./configs_mlflow.yaml",
            ),
            encoding="utf-8",
        ) as file:
            mlcfg = yaml.safe_load(file)
        self._set_env_vars(mlcfg["env"])
        mlflow.set_tracking_uri(mlcfg["mlflow_uri"])
        model_name = mlcfg[self.model_name]["model_name"]
        model_mlflow_uri = f"models:/{model_name}/{mlcfg[self.model_name]['version']}"
        try:
            mlflow.onnx.load_model(model_mlflow_uri, f"./{self.model_name}")
            self.configs["model"] = f"./{self.model_name}/model.onnx"
            self.logger.info("MODEL DOWNLOADED SUCCESSFULLY")
        except ClientError:
            self.logger.error(
                """ERROR DOWNLOADING MODEL FROM MLFLOW,Credientials Issue;
            Make sure you have sourced the env file"""
            )
            self.logger.info("USING LOCAL MODEL")

    def _load_model(self, model_name: str):
        """Select and return the class corresponding to model name
        Args:
            model_name (str): The recognition model name
        """
        try:
            if model_name == "large":
                self.model = arcface.ArcfaceOnnx(self.logger, self.configs)
            elif model_name == "pedestrian":
                self.model = adaface.AdafaceOnnx(self.logger, self.configs)
            elif model_name == "small":
                self.model = mobilefacenet.MBFOnnx(self.logger, self.configs)
            dummy_input = np.random.rand(1, 3, 112, 112).astype(np.float32)
            self.model.predict(dummy_input)
            self.logger.info("Model '%s' loaded successfully", model_name)
        except Exception as exception:
            self.logger.error("Error loading model with name '%s'", model_name)
            raise exception

    def pre_process(self, img) -> np.ndarray:
        """image pre-processing"""

        return self.model.pre_process(img)

    def predict(self, img: np.ndarray) -> np.ndarray:

        """performs inference and returns image embeddings"""

        preprocessed_image = self.pre_process(img)

        return self.model.predict(preprocessed_image)

    def set_device(self, device: str):
        """set device
        Args:
            device (str): device name
        """
        return self.model.set_device(device)

    def get_device(self):
        """get device
        Returns:
        str : device name
        """
        return self.model.get_device()

    def set_thresh(self, thresh: float):
        """set det_thresh
        Args:
            thresh (float): threshold
        """
        return self.model.set_thresh(thresh)

    def get_thresh(self):
        """get threshold
        Returns:
        float : thresh
        """
        return self.model.get_thresh()

