FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Set the working directory in the container
WORKDIR /app

# Copy the model_repository directory into the container
COPY model_repository /models

# Copy the SCRFD/models directory into the container
COPY SCRFD/models /onnx-models

# Run the tritonserver command when the container launches
CMD ["tritonserver", "--model-repository=/models"]