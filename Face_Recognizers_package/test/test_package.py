""" Package tests"""

import os
import cv2
from face_recognizers import Recognizer
from face_recognizers import DataBase

MODELS = ["large", "small", "pedestrian"]
DB_CONFIGS_PATH = "./test/assets/db_configs.yaml"
TEST_IMG_PATH = "./test/assets/test_messi_cropped.jpg"
RESULTS_PATH = "./test/results"

os.makedirs(RESULTS_PATH, exist_ok=True)


def test_fr_models_loading():
    """Test FR models loading"""
    for model in MODELS:
        Recognizer(model)


def test_db_loading():
    """Test DB Object loading"""
    DataBase(DB_CONFIGS_PATH)


def test_face_encode():
    """Test face encoding"""
    for model in MODELS:
        recognizer = Recognizer(model)
        img = cv2.imread(TEST_IMG_PATH)
        img = cv2.resize(img, (112, 112))
        face = recognizer.predict(img)
        assert face is not None
        assert face.shape == (1, 512)


def test_database_geenration():
    """Test database generation,loading and test example"""
    for model in MODELS:
        recognizer = Recognizer(model)
        database = DataBase(DB_CONFIGS_PATH)
        database.generate_database(recognizer)
        img = cv2.imread(TEST_IMG_PATH)
        face = recognizer.predict(img)
        assert face is not None
        print(recognizer.configs["thresh"])
        face_id = database.get_id(face, recognizer.configs["thresh"])
        assert face_id == "Messi"


def test_exsisting_database_recognition():
    """load exsisting database and test example"""
    model = "pedestrian"
    recognizer = Recognizer(model)
    database = DataBase(DB_CONFIGS_PATH)
    database.load_database()
    img = cv2.imread(TEST_IMG_PATH)
    img = cv2.resize(img, (112, 112))
    face = recognizer.predict(img)
    assert face is not None
    face_id = database.get_id(face, recognizer.configs["thresh"])
    assert face_id == "Messi"
    # delete_model()


def delete_model():
    "Just deleting the downloaded model"
    # remove model_name folder
    for model in MODELS:
        os.system(f"rm -rf ./{model}")
