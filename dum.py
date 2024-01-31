import cv2
import os
import numpy as np
import pandas as pd
# from tqdm import tqdm
from models import Models
import csv

# use this file to generate embeddings for database with the following structure:
# 4B-Data/
# ├── id1
# │   ├── id1_1.jpg
# │   ├── id1_2.jpg
# │   └── ...
# ├── id2
# │   ├── id2_1.jpg
# │   ├── id2_2.jpg
# │   └── ...
# └── ...

if __name__ == "__main__":
    
    # path to database
    database_path = "/data/Biometri-dev/Hermas/school_poc/FaceSolution_API/school/assets/datasets/4B-Data/"
    models = Models()
    # create csv file to save embeddings
    csv_file = "dummy.csv"


    for id_folder in os.listdir(database_path):
        id_path = os.path.join(database_path, id_folder)
        for img in (os.listdir(id_path)):
            img_path = os.path.join(id_path, img)
            if ".jpg" not in img_path:
                continue
            # print(img_path)
            frame = cv2.imread(img_path)
            # face detection
            # print("img shape: ", frame.shape)
            detections = models.detector.predict(frame)
            # face alignment
            for i in range(len(detections[0])):
                if(detections[0][i][4]> 0.6): 
                    aligned_frame= models.aligner.align(img=frame, landmarks= np.array(detections[1][i]))
                    # face recognition
                    embdding = models.recognizer.predict(aligned_frame)[0].copy()
                    # convert to list
                    embdding = embdding.tolist()
                    # convert embedding to list of strings
                    embdding = [str(i) for i in embdding]
                    
                    # add id to embedding
                    embdding.append(id_folder)
                    # save embedding
                    with open(csv_file, mode='a') as file:
                        writer = csv.writer(file)
                        writer.writerow(embdding)
                        # print(embdding.shape)
                else:
                    continue
    
    


            