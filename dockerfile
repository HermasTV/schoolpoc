FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Set the working directory in the container
WORKDIR /app

# Copy the model_repository directory into the container
COPY model_repository /models

# Copy the SCRFD/models directory into the container
COPY onnx-models /onnx-models

# copy the models converter script into the container
COPY convert_run_models.sh /app/convert_run_models.sh

# run the models converter script
RUN chmod 777 /app/convert_run_models.sh

# Run the convert_run_models.sh script in terminal when the container launches
CMD ["/bin/bash", "-c", "/app/convert_run_models.sh"]

# to build the docker image run the following command
# docker build -t tritonserver:23.09-py3 .
# to run the docker image run the following command
# docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 tritonserver:23.09-py3
