# !/bin/bash
#
echo "Converting ONNX models to TensorRT .plan models if not already converted ... "



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
# run the triton server
echo "Starting the Triton Inference Server ..."
tritonserver --model-repository=./model_repository
