FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Set the working directory in the container
WORKDIR /app

# copy all files to the working directory
COPY . /app

# add universe repository
RUN add-apt-repository universe 

RUN apt-get update && apt-get install ffmpeg  -y

# install requirements 
RUN pip install --ignore-installed -r /app/requirements.txt

# run the models converter script
RUN chmod 777 /app/convert_run_models.sh

# Run the convert_run_models.sh script in terminal when the container launches
CMD ["/bin/bash", "-c", "/app/convert_run_models.sh"]

# to build the docker image run the following command
# docker build -t tritonserver:23.09-py3 .
# to run the docker image run the following command and mount model_repository to the container
# docker run --name school --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/:/app/ tritonserver:23.09-py3
# then you can run the following command to run client
# docker exec -it school python3 client.py