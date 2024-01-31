#!/bin/bash

docker run --gpus=1 --rm --net=host -it \
-v ${PWD}/model_repository:/models \
-v ${PWD}/SCRFD/models/:/onnx-models \
nvcr.io/nvidia/tritonserver:23.09-py3 /bin/bash -c "tritonserver --model-repository=/models; /bin/bash"
