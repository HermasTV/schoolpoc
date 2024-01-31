# Installation

### Using setup.py
```script
python setup.py install
```
### Using pip
```shell
pip install .
```

# Supported models 
1. large : (arcface)
2. pedestrian : (adaface)
3. small : (mobilefacenet)

# Test Package
You can go to the package directory and simply open a terminal and call pytest:
```CMD
pytest
```

# Usage 
The model takes cropped face image and returns the encoding of the face,
usually its linear victor of 512 elements.

## Recognition model 

### configs structure
You can use the `configs_fr.yaml` file as reference, and here's a sample structure if you want to create large model (arcface) and load it to cpu with distance threshold of 0.75 :

```yaml
large:
    # dont have to provide model path unless u faced issue in downloading  
    model:  "./model.onnx" 
    device : "cpu" 
    thresh : 0.75
```

### Loading The Model
By Default the model will be loaded by downloading the onnx model from MLFLOW,  
If an issue occured, The model will load the onnx file from the path given in the configs file.

You can Create Model Object By passing the configs file path or configs Dictionary :  

### Load with Configs Path
```python
from face_recognizers import Recognizer

CONFIG_PATH = "path to FR yaml configs"
detector = Recognizer("large", CONFIG_PATH)
```

### Load with Configs dictionary
```python
from face_recognizers import Recognizer

CONFIG_PATH = "path to FR yaml configs"
with open(CONFIG_PATH) as file:
    cfg = yaml.safe_load(file)
detector = Recognizer("small", cfg)
```

### Prediction 
The prediction fucntion encode the face with a liner vector of length 512. 

```python 
from face_recognizers import Recognizer

recognizer = Recognizer(model, FR_CONFIGS_PATH)
img = cv2.imread(TEST_IMG_PATH)
face_encoding = recognizer.predict(img)
```


## Database :

### configs structure

There are one type of db which is `basic` and here is the structure of it : 

`configs_db.yaml` :
```yaml
basic:
    # in case of you dont have db_file 
    subjs_path: "db_images_dir/" # path to images of cropped face
    db_save_path: "db_dir/" # path to dir where to save db.csv file

    # in case you have db file
    db_path: "database.csv" # path to file
```

### the structure of the original data : 

If you want to create data from scratch please follow the following template : 

```
data/
    id1/
        img1.jpg
        img2.jpg
        img3.jpg
    id2/
        img1.jpg
        img2.jpg
        img3.jpg
    .
    .
    etc
```

### DataBase generation : 
Make sure that the images are face cropped, if not, use the facedetection package to detect faces and crop, with 1.3 ratio as follows : 

#### crop faces script
```python
import os
import os.path as osp
import cv2
from face_detectors import Detector
from face_detectors.utils import face_utils

# FD object
detector = Detector("large","./configs.yaml")

DB_DIR = "path to database images"
DB_CROP_DIR = "Path to cropped images"
os.makedirs(DB_CROP_DIR, exist_ok=True)

for subj_id in os.listdir(DB_DIR):
    os.makedirs(osp.join(DB_CROP_DIR, subj_id), exist_ok=True)
    if osp.isfile(osp.join(DB_DIR, subj_id)):
        continue
    subject_embeddings = []
    for img in os.listdir(osp.join(DB_DIR, subj_id)):
        img_path = osp.join(DB_DIR, subj_id, img)
        img_cv = cv2.imread(img_path)
        res = detector.predict(img_path)
        face, bbox_new = face_utils.crop_face(img_cv,res["bboxs"][0],1.3)
        cv2.imwrite(osp.join(DB_CROP_DIR, subj_id, img), face)
```

#### How to generate db
After cropping the face images, you should load the face recognition model you will be using, then pass it to the generation function of the Database model.

```python
from face_recognizers import Recognizer
from face_recognizers import DataBase

recognizer = Recognizer("modelName", FR_CONFIGS_PATH)
database = DataBase(DB_CONFIGS_PATH)
database.generate_database(recognizer)
```

### Existing Database Loadning
as simple as creating an object of the database and call the load function

```python
from face_recognizers import DataBase

database = DataBase(DB_CONFIGS_PATH)
database.load_database()
```

### Test if image in Database 
if you have an image of a cropped face and you want to know if its in the data base or not, you pass it to recognition model to encode it and then use the get_id fucntion from the database handler


```python
from face_recognizers import Recognizer
from face_recognizers import DataBase

model = "pedestrian"
# load face recognition model
recognizer = Recognizer(model, FR_CONFIGS_PATH)
# load database assuming its aleady exists
database = DataBase(DB_CONFIGS_PATH)
database.load_database()
# load cropped face image
img = cv2.imread(TEST_IMG_PATH)
# encode face
face_encode = recognizer.predict(img)
# pass it to get_id and use the recognition model threshold
face_id = database.get_id(face_encode, recognizer.configs["thresh"])
```
