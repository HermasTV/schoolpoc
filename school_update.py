'''school pipeline class
    @authors:   Eslam Abdelrahman
                Mohamed Salah
                Anwar Alsheikh
    @lisence: Tahaluf 2023
'''

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
        

    def __init_streams(self)-> None:
        # streams_data is dict of streams dicts 
        # {name : {path: , location: }}
        streams_data = self.config["streams"]
        self.streams = [CameraStream(src=streams_data[stream]["path"],
                                      name=stream, location=streams_data[stream]["location"],
                                      visualize=True)
                                        for stream in streams_data]

    def embed(self,frame: Array)-> Array:
        '''
        Args:
            frame: Array, input frame.
        Output:
            Embeddings: Array, output face embeddings.
            Bboxs: Array, detected boundboxes.
            Landmarks: Array, detected landmarks.
        '''
        
        detections= self.models.detector.predict(frame)
        # print(detections[0].shape)
        # print("detections", detections)
        aligned_frames= []
        
        for i in range(len(detections[0])):
            aligned_frame= self.models.aligner.align(img=frame, landmarks= np.array(detections[1][i]))
            aligned_frames.append(aligned_frame)
            # Save aligned frames
            # cv2.imwrite(f"./assets/{i}.jpg", aligned_frame)
        embddings = []
        if aligned_frames:
            
            for aligned_frame in aligned_frames:
                aligned_frame= aligned_frame.reshape(1,3,112, 112).astype(np.float32)
                # print(aligned_frame)
                embdding = self.models.recognizer.predict(aligned_frame)[0].copy()
                # print("embed shape : ", embdding.shape)
                embddings.append(embdding)
        embddings = np.array(embddings)

        return embddings, detections[0]
    
    def find(self, embeddings)-> Any:
        '''
        Args:
            embeddings: Array, frame detected face embeddings.
        Output:
            id: string, detected face id.
            similarity score: float, least distance of face embeddings.

        '''
        embeddings= embeddings.reshape(1, -1)
        similarity= cosine_similarity(embeddings, self.faces)
        embeddings= embeddings.reshape(1, -1)
        ind= np.argsort(similarity)[0][-1]
        return self.ids[ind], np.sort(similarity)[0][-1]

    def process_stream(self,stream:CameraStream)-> Array:
        '''
        Args:
            frame: Array, input frame to be proccessed 
            
        Output:
            frame: Array, output frame with detected bboxes and student id.
        '''

        # load frame 
        stream.update_frame()
        if stream.is_stopped():
            return
        thresh= self.config["similarity_thresh"]

        embeddings, bboxs= self.embed(stream.frame)

        temp_presence= {}
        for i in range(embeddings.shape[0]):
            # print(i)
            embedding = embeddings[i]
            studentId, sim= self.find(embedding)
            if sim>= thresh:
                if self.config["debug"]:
                    print(f'student id is {studentId}, i am sure {int(sim*100)} %')
                    utils.boundboxes(stream.frame, bboxs[i], studentId, sim)
                if studentId in stream.presence:
                    temp_presence[studentId] = stream.presence[studentId]+1
                else:
                    temp_presence[studentId] = 1
                

                if temp_presence[studentId] >= self.config["consecutive_frames"] and \
                                ((studentId not in self.login and stream.location=="entrance") or \
                                (studentId not in self.logout and stream.location=="exit")):
                    if stream.location == 'entrance':
                        self.login.add(studentId)
                    else:
                        self.logout.add(studentId)
                    current_datetime= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("./assets/logs.csv", mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([str(current_datetime), str(studentId), str(stream.name)])

            else:
                self.unknown["unknown"+ str(len(self.unknown))]= embedding
                if self.config["debug"]:
                    utils.boundboxes(stream.frame, bboxs[i], 'u', sim)
        stream.presence= temp_presence.copy()
        if self.config["debug"]:
            utils.dashboard(stream.frame, stream.frame.shape[0], self.login, self.logout)
            
    def streams_run(self)-> None:
        '''
        Args:
            none
            
        Output:
            none
        '''
        while True:
            for stream in self.streams:
                self.process_stream(stream)
                if self.config["debug"]:
                    stream.write_frame()
            if all ([stream.is_stopped() for stream in self.streams]):
                break

    def process_streams_parallel(self: list)-> None:
        '''
        Args:
            streams: list, list of streams to be processed in parallel.
        
        Output:
            none
        '''
        while True:
            batch = []
            for stream in self.streams:
                stream.update_frame()
                batch.append(stream.frame)
            if any ([stream.is_stopped() for stream in self.streams]):
                break
            # use FD model to detect faces in batch
            batch = np.array(batch)
            
            detections = self.models.detector.predict(batch)

            # print(len(detections))
            
            # process each stream
            
            for s in range(len(detections)):
                # print("processing stream: ", s,"frame: ", self.streams[s].frame_count)
                #prepare aligned frames for each stream to be processed by FR model
                aligned_frames= []
                dets_stream = detections[s]
                # print("found faces: ", len(dets_stream[0]))
                if len(dets_stream[0]) == 0:
                    self.streams[s].write_frame()
                    continue
                for j in range(len(dets_stream[0])):
                    aligned_frame= self.models.aligner.align(img=self.streams[s].frame, landmarks= np.array(dets_stream[1][j]))
                    aligned_frames.append(aligned_frame)
                    # save aligned frames using random number
                    # cv2.imwrite(f"./results/{self.streams[s].frame_count}_{j}.jpg", aligned_frame)
                # convert aligned frames to array
                aligned_frames = np.array(aligned_frames)
                # prepare embeddings for each stream to be processed by FR model
                # reshape aligned frames to be compatible with FR model
                # print("aligned frames shape: ", aligned_frames.shape)
                aligned_frames = aligned_frames.reshape(aligned_frames.shape[0], 3, 112, 112).astype(np.float32)
                frame_embeds = []
                for i in range(aligned_frames.shape[0]):
                    frame_embeds.append(self.models.recognizer.predict(aligned_frames[i]).copy())
                # frame_embeds = self.models.recognizer.predict(aligned_frames).copy()
                frame_embeds = np.array(frame_embeds)
                # expand dims
                
                frame_embeds = frame_embeds.reshape(aligned_frames.shape[0], -1)
                # print("embed shape : ", frame_embeds.shape)
                
                # process each face in stream
                for i in range(frame_embeds.shape[0]):
                    # print(i)
                    embedding = frame_embeds[i]
                    studentId, sim= self.find(embedding)
                    if sim>= 0.5:
                        if self.config["debug"]:
                            print(f'student id is {studentId}, i am sure {int(sim*100)} %')
                            # print(dets_stream[])
                            utils.boundboxes(self.streams[s].frame, dets_stream[0][i], studentId, sim)
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
                        self.unknown["unknown"+ str(len(self.unknown))]= embedding
                        if self.config["debug"]:
                            utils.boundboxes(self.streams[s].frame, dets_stream[0][i], 'u', sim)
                if self.config["debug"]:
                    self.streams[s].write_frame()