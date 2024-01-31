""" setup file
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", encoding="utf-8") as file:
    requirements = file.readlines()

setup(
    name="face-recognizers",
    version="0.3",
    include_package_data=True,
    author="Mahmoud Ewaisha",
    author_email="m.ewaisha.ext@tahaluf.ae",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    license="Tahaluf UAE",
    packages=find_packages(exclude=(["arcface", "adaface", "mobilefacenet_onnx"])),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence ",
    ],
    keywords="",
)
