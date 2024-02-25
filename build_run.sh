#!/bin/bash

# Set the IMAGES directory path
IMAGES_DIR="IMAGES/"

echo "Converting ONNX models to TensorRT .plan models if not already converted ..."

# Before conversion make sure that the plan files are not present in the model repository 
# convert Face Detection ONNX model to tensorrt .plan model
if [ ! -f ./model_repository/SCRTRT_2/0/model.plan ]; then
    echo "Face Detection model not found, converting it to .plan"
    /usr/src/tensorrt/bin/trtexec --onnx=./onnx-models/scrfd_2.5.onnx --saveEngine=./model_repository/SCRTRT_2/0/model.plan 
fi

# convert Face Recognition ONNX model to tensorrt .plan model
if [ ! -f ./model_repository/arcface/0/model.plan ]; then
    echo "Face Recognition model not found, converting it to .plan"
    /usr/src/tensorrt/bin/trtexec --onnx=./onnx-models/model.onnx --saveEngine=./model_repository/arcface/0/model.plan 
fi

echo "Generating Database csv file IF not existed for the Face Recognition model ..."
if [ ! -f ./arcface-r50-trt.csv ]; then
    echo "Database csv file not found, generating it ..."
    python3 generate_database.py "$IMAGES_DIR" arcface-r50-trt.csv
fi

echo "Starting the Triton Inference Server ..."
tritonserver --model-repository=./model_repository;

# Check if the first argument is 'run-client'
if [ "$1" = "run-client" ]; then
    echo "Running client.py ..."
    python3 client.py
fi

# RUN THE APP if the second argument is 'run-dashboard'
if [ "$2" = "run-dashboard" ]; then
    echo "Running dashboard.py ..."
    python3 dashboard.py
fi

# End of file

