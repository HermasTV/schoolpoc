""" Main Database module

    @author: Mahmoud Ewaisha
             Hermas
    @copyright: Tahaluf UAE 2023
"""
import os
import os.path as osp
import csv
from typing import List
import imageio
import numpy as np
from scipy.spatial.distance import cdist


class DataBase:
    """Simple Database class for face recognition models"""

    def __init__(self, logger, configs) -> None:
        self.cfg = configs
        self.logger = logger
        self.db_path = None
        self.database = None

    def generate_database(self, model) -> None:
        """generates & saves subjects face embeddings"""
        ids, subjects_embeddings = [], []
        subjs_path = self.cfg["subjs_path"]
        assert subjs_path, "Please specify the path to database images in the config"
        assert self.cfg["db_path"], "Please specify path to save the DB in the config"

        for subj_id in os.listdir(subjs_path):
            if osp.isfile(osp.join(subjs_path, subj_id)):
                continue
            ids.append(subj_id)
            subject_embeddings = []
            for img in os.listdir(osp.join(subjs_path, subj_id)):
                img = imageio.imread(osp.join(subjs_path, subj_id, img))
                # predict the embedding
                subject_embeddings.append(model.predict(img))
                # flip the image and predict again
                subject_embeddings.append(model.predict(img[:, ::-1, :]))
            subjects_embeddings.append(subject_embeddings)
        subjects_embeddings = np.array(subjects_embeddings)

        averaged_subjects_embeddings = []
        # average the embeddings
        for subject_embeddings in subjects_embeddings:
            average_embedding = np.mean(np.asarray(subject_embeddings), axis=0)
            averaged_subjects_embeddings.append(average_embedding)
        averaged_subjects_embeddings_arr = np.array(averaged_subjects_embeddings)[
            :, 0, :
        ]

        # check if the path is a csv file
        if ".csv" in self.cfg["db_path"]:
            # take the path to the directory
            self.cfg["db_path"] = os.path.dirname(self.cfg["db_path"])

        assert os.path.isdir(
            self.cfg["db_save_path"]
        ), "Please specify a valid path to save the DB in the config"

        self.db_path = os.path.join(
            self.cfg["db_save_path"], f"database_{model.model_name}.csv"
        )
        self.save_database(self.db_path, ids, averaged_subjects_embeddings_arr)
        self.load_database()

    def load_database(self) -> None:
        """loads subjects face embeddings
        Returns:
            List[str]: list of string ids of subjects
            np.ndarray: numpy array of subjects face embeddings
        """
        if not self.db_path:
            self.db_path = self.cfg["db_path"]
        assert (
            self.db_path
        ), "No database loaded as no file was specified in the config."
        assert os.path.exists(self.db_path), (
            "No database loaded as specified file does not exist. "
            "Run generate_database() method to generate and load the DB."
        )

        rows, ids = [], []
        with open(self.db_path, encoding="utf-8") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                ids.append(row[0])
                rows.append(row[1:])
        embeddings_db = np.array([np.array(i).astype("float32") for i in rows]).astype(
            "float32"
        )
        self.database = ids, embeddings_db

    def get_id(self, feats: np.ndarray, thresh: float) -> str:
        """returns subject ID"""
        assert hasattr(self, "database"), "No database file specified"

        ids, embeddings_db = self.database

        distances = cdist(feats, embeddings_db, "cosine")
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        self.logger.info(min_distance)
        if min_distance < thresh:
            identity = ids[min_index]
        else:
            identity = str(-1)

        return identity

    @staticmethod
    def save_database(
        save_path: str,
        ids: List[str],
        averaged_subjects_embeddings_arr: np.ndarray,
    ) -> None:
        """Save embeddings to csv
        Args:
            save_path (str): path to save the csv
            ids (List[str]): list of subjects' IDs
            averaged_subjects_embeddings_arr (np.ndarray): subjects' embeddings
        """
        with open(f"{save_path}", "w", encoding="utf-8") as csv_file:
            header = "id"
            for i in range(averaged_subjects_embeddings_arr.shape[-1]):
                header += f",embedding_{i}"
            header += "\n"
            csv_file.write(header)
            for i in range(0, averaged_subjects_embeddings_arr.shape[0]):
                embeddings_str = (
                    ids[i]
                    + ","
                    + np.array2string(
                        averaged_subjects_embeddings_arr[i], separator=","
                    )
                    .replace("[", "")
                    .replace("]", "")
                    .replace("\n", "")
                    .replace(" ", "")
                    .strip()
                    .rstrip()
                    .lstrip()
                    + "\n"
                )
                csv_file.write(embeddings_str)
