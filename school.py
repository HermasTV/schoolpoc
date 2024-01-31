from SCRTrT import faceDetectionSCRFD as FD
from arcTrT import faceRecognizer as FR
import concurrent.futures
import os
import cv2
import glob 
import time
import utils_e as utils_
import numpy as np
import pdb
from sklearn.metrics.pairwise import cosine_similarity
import csv
import datetime
from typing import Any


class school:

    def __init__(self, csv_file: str) -> None:
        self.face_detector = FD("configs/SCR_TRT.yaml",version="two")
        self.face_recognizer = FR("configs/arc.yaml",version="trt")
        self.ids, self.faces= utils_.read_embd(csv_file)
        self.config = utils_.load_configs("config.yaml")["school"]
        self.num_students= set()
        self.known= {}
        self.unknown= {}
        self.presence = {}


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
    

    def embed(self,frame: np.ndarray):

        # pdb.set_trace()
        detections = self.face_detector.predict(frame[..., ::-1].copy())
        # pdb.set_trace()

        aligned_frames= []
        for i in range(len(detections[0])):
            aligned_frame= utils_.align(img=frame, landmarks= np.array(detections[0][i]["landmarks"]))
            aligned_frames.append(aligned_frame)
        
        embddings = []
        if aligned_frames:
            for aligned_frame in aligned_frames:
                aligned_frame= aligned_frame.reshape(1,3,112, 112)
                embdding = self.face_recognizer.predict(aligned_frame)[0].copy()
                embddings.append(embdding)
        embddings = np.array(embddings)
        return embddings
    

    def run(self, frame: np.ndarray)-> np.ndarray:
        '''
        Args:
            frame: Array, input frame to be proccessed 
            
        Output:
            frame: Array, output frame with detected bboxes and student id.
        '''
        thresh= self.config["similarity_thresh"]
        face_visuals_times= []
        find_times= []
        ids = []

        embeddings= self.embed(frame)
        for i, embedding in enumerate(embeddings):
            start= time.time()
            studentId, sim= self.find(embedding)
            find_times.append(time.time()- start)
            # print(sim)
            if sim>= thresh:
                print(f'student id is {studentId}, i am sure {int(sim*100)} %')
                ids.append(studentId)
                
                start= time.time()
                
                # utils.boundboxes(frame, bboxs[i], studentId, sim)
                # face_visuals_times.append(time.time()- start)
            
            else:
                self.unknown["unknown"+ str(len(self.unknown))]= embedding
                start= time.time()
                # try:
                #     utils.boundboxes(frame, bboxs[i], 'u', sim)
                # except:
                #     print('Error Frame')
                #     # pdb.set_trace()    
                face_visuals_times.append(time.time()-start)
            
            return frame,ids


    def stream(self,video_path: str, output: str, camID: str)-> None:
        '''
        Args:
            video_path: str, input video stream to be analyzed
            output: str, output path for the processed video
            camId: str, cameraid that indicated the type of the camera "enterance, exit" 
            
        Output:
            frame: Array, output frame with detected bboxes and student id.
        '''
         
        frame_count= 0
        login= set()
        logout= set()
        # pdb.set_trace()
        consecutive_frames= self.config["consecutive_frames"]
        presence = {}
        prev_frame = {}
        # pdb.set_trace()

        keys_to_delete = set()
        video= cv2.VideoCapture(video_path) 
        width= int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_codec= cv2.VideoWriter_fourcc(*'mp4v')
        out= cv2.VideoWriter(output+'/demo_out.mp4', output_codec, 30, (width, height))

        if not video.isOpened():
            print("Error opening video file")

        frame_writing_times= []
        visuals_times= []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            try:
                frame_,ids = self.run(frame)
            except:
                # print("bad out")
                continue

            for id in ids:
                if id in list(presence.keys()):
                    presence[id]+= 1
                else:
                    presence[id]= 1

            if prev_frame:
                for id in list(prev_frame.keys()):
                    if id not in ids :
                        presence[id]=0
                        keys_to_delete.add(id)

                for id, consecutive_count in presence.items():
                    if consecutive_count >= consecutive_frames and (id,camID) not in (login or logout)  :
                        if camID in (["enter0","enter1"]) :
                            login.add((id,camID))
                        else:
                            logout.add((id,camID))
                        current_datetime= datetime.datetime.now().strftime("%H:%M:%S")
                        with open("./assets/logs.csv", mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([str(current_datetime),  str(id), str(camID)])
            
            for id_to_delete in list(keys_to_delete):
                del presence[id_to_delete]
            keys_to_delete.clear()
            # print(frame_count, prev_frame)
            prev_frame=presence.copy()
            # start= time.time()
            # utils.dashboard(frame_, width, login, logout)
            # visuals_times.append(time.time()-start)
            # start= time.time()
            # out.write(frame_)
            # frame_writing_times.append(time.time()-start)
            # if frame_count%30==0:
            #     # print(f" fps for stream {camID} is {1/np.mean(visuals_times)}")
            frame_count+= 1

        print(f"stream {camID} finished ") 



def main():
    demo = school("arcface-r50-trt.csv")
    test_directory = "./tests"

    input_directory = f"{test_directory}/inputs"
    output_directory = f"{test_directory}/output_client"

    if os.path.isdir(output_directory):
        os.system(f"rm -dr {output_directory}")
    else:
        pass
    os.mkdir(output_directory)

    test_files = glob.glob(f"{input_directory}/*.*")[0]
    # pdb.set_trace()
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_one = executor.submit(demo.stream,"./streams/Rand-1.mp4",output_directory,"enter0")
        print("running fun1")
        future_two = executor.submit(demo.stream,"./streams/Rand-3.mp4",output_directory,"enter1")
        print("running fun2")
        future_three = executor.submit(demo.stream,"./streams/Rand-2.mp4",output_directory,"exit0")
        print("running fun3")
        future_four = executor.submit(demo.stream,"./streams/Rand-4.mp4",output_directory,"exit1")
        print("running fun4")

    end = time.time()
    print("Total Time taken: ", end - start)


main()