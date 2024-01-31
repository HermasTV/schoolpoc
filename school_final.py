
import cv2
import csv
import time
import datetime
import numpy as np
import pdb

from sklearn.metrics.pairwise import cosine_similarity

from typing import Any
import traceback
Array= np.array

import utils.face_utils as utils
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
        self.faces_normalized = self.faces / np.linalg.norm(self.faces, axis=1,keepdims=True)
        self.faces_normalized = self.faces_normalized.astype('float32')
        
        print("faces shape", self.faces.shape)
        # use gpu
        # self.res = faiss.StandardGpuResources()
        # self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        self.index.add(self.faces_normalized)
        
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
    
    
    def embed(self,frame: np.ndarray):

        detections = self.models.detector.predict(frame)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # perform face recognition for each face in each stream
            embeddings = executor.map(self.align_and_embedd, [frame]*len(detections[1]), detections[1], range(len(detections[1])))

        return embeddings

        
    def align_and_embedd(self, frame: np.ndarray, landmarks , s : int):
        """
        Args:
            frame: Array, input frame to be proccessed 
            landmarks: Array, detected face landmarks.
            s: int, stream id
        Output:
            none
        """
        # align face
        aligned_frame= self.models.aligner.align(img=frame,
                                                landmarks= np.array(landmarks))
        # embed face
        embeds = self.models.recognizer.predict(aligned_frame)
        # Find face id
        student_number = self.find_faiss(embeds)
        # print(self.faces[studentId].shape)
        # print(embeds.shape)
        # calculate similarity between embeds and self.faces[studentId]
        sim = cosine_similarity(embeds.reshape(1, -1), self.faces[student_number].reshape(1, -1))
        sim = sim[0][0]
        studentId = self.ids[student_number]
        # check if face is known
        if sim>= 0.5:
            # add face to known faces
            # print("known", studentId, sim)
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
                current_datetime= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("./assets/logs.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([str(current_datetime), str(studentId), str(self.streams[s].name)])

        else:
            
            self.unknown["unknown"+ str(len(self.unknown))]= embeds
        
    
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
                self.embed(stream.frame)
            # self.embed(stream.frame)