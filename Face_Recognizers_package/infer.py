"""test"""
from face_recognizers import Recognizer

CONFIG_PATH = "path to FR yaml configs"
detector = Recognizer("pedestrian")
print(detector)
