
import os
import cv2
import csv
import time
import datetime
import numpy as np
import pdb
import pytz
from sklearn.metrics.pairwise import cosine_similarity

from typing import Any
import traceback
Array= np.array

from utils import face_utils
import utils.utils as utils_ 
from models import Models
from streams import CameraStream


from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import faiss

class school:

    def __init__(self,models: object)-> None:
        '''
        Args:
            csv_file: string, path of the database file.
        Output:
            none
        '''
        self.models = models
        self.config = utils_.load_configs("../config.yaml")
        print("Models loaded successfully !")
        self.csv= self.config[self.config["recognizer"]]["DB"]
        self.ids, self.faces= utils_.read_embd(self.csv)
        self.num_students= set()
        self.known= {} # why is this here?
        self.unknown= {}
        self.__init_streams()
        self.login= set()
        self.logout= set()
        self.index = faiss.IndexFlatL2(512)
        self.faces = self.faces.astype('float32')
        # print magnitudes of a sample face embedding
        
        print("faces shape", self.faces.shape)
        # use gpu
        # self.res = faiss.StandardGpuResources()
        # self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        self.index.add(self.faces)
        
    def __init_streams(self)-> None:
        # streams_data is dict of streams dicts 
        # {name : {path: , location: }}
        streams_data = self.config["streams"]
        self.streams = [CameraStream(src=streams_data[stream]["path"],
                                      name=stream, location=streams_data[stream]["location"],
                                      visualize=True)
                                        for stream in streams_data]
    
    def find(self, embedding)-> Any:
        '''
        Args:
            embeddings: Array, frame detected face embeddings.
        Output:
            id: string, detected face id.
            similarity score: float, least distance of face embeddings.

        '''
        # Ensure the input embedding is 2D (1, num_features)
        embedding = embedding.reshape(1, -1)

        # Compute cosine similarity between the input embedding and all known embeddings
        similarity = cosine_similarity(embedding, self.faces)

        # Find the index of the most similar face
        ind = np.argmax(similarity)

        # Return the ID and the corresponding similarity score
        return self.ids[ind], similarity[0, ind]
    
    def find_faiss(self, embedding)-> Any:
        '''
        Args:
            embeddings: Array, frame detected face embeddings.
        Output:
            id: string, detected face id.
            similarity score: float, least distance of face embeddings.

        '''
        # Ensure the input embedding is 2D (1, num_features)
        embedding = embedding.reshape(1, -1).astype('float32')

        # Compute cosine similarity between the input embedding and all known embeddings
        _,student = self.index.search(embedding, 1)
        # retun only the index of the most similar face
        return student[0][0]
    
    
    def embed(self,stream : CameraStream, stream_id : int):

        detections = self.models.detector.predict(stream.frame)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # perform face recognition for each face in each stream
            embeddings = executor.map(self.align_and_embedd, [stream]*len(detections[1]), detections[0],detections[1], [stream_id]*len(detections[1]))

        return embeddings

        
    def align_and_embedd(self,stream : CameraStream, detections,landmarks , s : int):
        """
        Args:
            frame: Array, input frame to be proccessed 
            landmarks: Array, detected face landmarks.
            s: int, stream id
        Output:
            none
        """
        # print("s : ", s)
        # align face
        aligned_frame= self.models.aligner.align(img=stream.frame,
                                                landmarks= np.array(landmarks))
        # embed face
        embeds = self.models.recognizer.predict(aligned_frame)
        # Find face id
        student_number = self.find_faiss(embeds)

        # calculate similarity between embeds and self.faces[studentId]
        sim = cosine_similarity(embeds.reshape(1, -1), self.faces[student_number].reshape(1, -1))
        sim = sim[0][0]
        studentId = self.ids[student_number]
        # keep only alphabetical characters and first occrance of "-"
        studentId = "-".join(studentId.split("-")[0:2])
        # print(studentId)
        # check if face is known
        if sim>= 0.5:
            
            #  visualize the face and the name
            # print(studentId, sim)
            if self.config["debug"]:
                # print("debug mode ...")
                face_utils.boundboxes(stream.frame, detections, studentId, sim)
            # add face to known faces
            if studentId in self.streams[s].presence:
                self.streams[s].presence[studentId] += 1
            else:
                self.streams[s].presence[studentId] = 1
            
            if self.streams[s].presence[studentId] >= self.config["consecutive_frames"] and \
                            ((studentId not in self.login and self.streams[s].location=="entrance") or \
                            (studentId not in self.logout and self.streams[s].location=="exit")):
                if self.streams[s].location == 'entrance':
                    self.login.add(studentId)
                else:
                    self.logout.add(studentId)               
                
                timezone = pytz.timezone('Africa/Cairo')
                date= datetime.datetime.now(tz=timezone)
                # time in hours and minutes and seconds
                current_datetime = date.strftime("%H:%M:%S")
                # date in day-month-year
                current_date = date.strftime("%d-%m-%Y")
                face_img_path = f"./logs/faces/{current_date}/{studentId}_{self.streams[s].name}.jpg"
                with open(f"./logs/{current_date}.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([str(current_datetime), str(studentId), str(self.streams[s].name), face_img_path])
                # save image to logs file in /logs/faces/date/studentId_streamId.jpg
                # first create the directory if it doesn't exist
                os.makedirs(f"./logs/faces/{current_date}", exist_ok=True)
                cv2.imwrite(face_img_path, aligned_frame)
        else:
            
            self.unknown["unknown"+ str(len(self.unknown))]= embeds
            if self.config["debug"]:
                # print("debug mode ...")
                face_utils.boundboxes(stream.frame, detections, 'u', sim)
        # if self.config["debug"]:
        #     face_utils.dashboard(stream.frame, stream.frame.shape[1], self.login, self.logout)

    
    def fetch_frame(self, stream_id):
        '''
        Args:
            stream_id: int, id of stream to be processed.
        Output:
            none
        '''
        # fetch stream
        stream = self.streams[stream_id]
        # fetch frame from stream
        stream.update_frame()
        # check if stream is stopped
        if stream.is_stopped():
            return None
        return stream.frame
    
    def run_1_stream(self, stream_id):
        '''
        Args:
            stream_id: int, id of stream to be processed.
        Output:
            none
        '''
        # fetch stream
        # print("statring stream", stream_id)
        stream = self.streams[stream_id]
        while True:
            # fetch frame from stream
            stream.update_frame()
            # check if stream is stopped
            if stream.is_stopped():
                return None
            # Run FD for Single Stream
            if stream.frame_count % self.config["skip_frame"]==0:
                self.embed(stream,stream_id)
                # write frame to output
            if self.config["debug"]:
                
                stream.write_frame()