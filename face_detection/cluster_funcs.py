import io
import os
import math
from PIL import Image
from sklearn.cluster import DBSCAN
from imutils import build_montages, paths
import numpy as np
import os
import pickle
import cv2
import shutil
import time
from tqdm import tqdm
from moviepy.editor import *
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from google.cloud import videointelligence_v1 as videointelligence

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "application_default_credentials.json"

IMGS_PER_FACE = 10

''' Common utilities '''
'''
Credits: AndyP at StackOverflow
The ResizeUtils provides resizing function to keep the aspect ratio intact

class ResizeUtils:
    # Given a target height, adjust the image by calculating the width and resize
    def rescale_by_height(self, image, target_height, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_height` (preserving aspect ratio)."""
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return cv2.resize(image, (w, target_height), interpolation=method)

    # Given a target width, adjust the image by calculating the height and resize
    def rescale_by_width(self, image, target_width, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_width` (preserving aspect ratio)."""
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return cv2.resize(image, (target_width, h), interpolation=method)
'''

def detect_faces(path):
    """Detects faces in a video."""
    
    client = videointelligence.VideoIntelligenceServiceClient()

    if not path.startswith("gs"):
        with io.open(path, "rb") as f:
            input_content = f.read()
    else:
        input_content = path
    
    # Configure the request
    config = videointelligence.FaceDetectionConfig(
        include_bounding_boxes=True, include_attributes=False
    )
    context = videointelligence.VideoContext(face_detection_config=config)

    # Start the asynchronous request
    operation = client.annotate_video(
        request={
            "features": [videointelligence.Feature.FACE_DETECTION],
            "input_uri": input_content,
            "video_context": context,
        }
    )

    print("\nProcessing video for face detection annotations.")
    result = operation.result(timeout=10000)

    print("\nFinished processing.\n")

    # Retrieve the first result, because a single video was processed.
    return result.annotation_results[0]

def extract_faces(annotation_result,movie_name,path,output):
    """ Extract faces pictures to a folder """
    if os.path.exists(output):
        shutil.rmtree(output)
        time.sleep(0.5)
    os.mkdir(output)
    faces_dir = os.path.join(output,"Faces")
    os.mkdir(faces_dir)
    all_dir = os.path.join(faces_dir,"all")
    os.mkdir(all_dir)
    
    # TODO movie = UPLOAD FROM GOOGLE CLOUD
    movie = VideoFileClip(path)
    i = 0
    
    for annotation in tqdm(annotation_result.face_detection_annotations, desc = "faces extracting"):
        for track in annotation.tracks:
            if track.confidence > 0.8:
                # Each segment includes timestamped faces that include
                # characteristics of the face detected.
                objects = track.timestamped_objects
                imgs_n = len(objects)
                if(imgs_n) <= IMGS_PER_FACE:
                    step = 1
                else:
                    step = math.floor(imgs_n/IMGS_PER_FACE)
                for obji in range(0, imgs_n, step):
                    # Iterate over the pictures
                    obj = objects[obji]
                    i+=1
                    frame = os.path.join(output,"image"+str(i)+"full.jpg")
                    movie.save_frame(frame, t=(obj.time_offset.seconds + obj.time_offset.microseconds / 1e6), withmask=False)
                    box = obj.normalized_bounding_box
                    
                    # Opens a image in RGB mode
                    im = Image.open(frame)
                    
                    # Size of the image in pixels (size of original image)
                    width, height = im.size
                
                    face_path = os.path.join(all_dir,str(i)+".jpg")
                    
                    crop_im = im.crop((box.left*width, box.top*height, box.right*width, box.bottom*height))
                    
                    #TODO keep the same ratios and sizes
                    """ def AutoResize(self, frame):
            resizeUtils = ResizeUtils()

            height, width, _ = frame.shape

            if height > 500:
                frame = resizeUtils.rescale_by_height(frame, 500)
                self.AutoResize(frame)
            
            if width > 700:
                frame = resizeUtils.rescale_by_width(frame, 700)
                self.AutoResize(frame)
            
            return frame """
                    
                    crop_im.save(face_path)
                    os.remove(frame)
                    
    return os.path.join(output,"Faces")
"""
# Option - using mediapipe landmarks

def landmark(faces_dir):
    #Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    dict = {}
    
    # for every image, detect landmarks and save to python dict
    for face in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, face)
        id = os.path.splitext(face)[0]
        image = mp.Image.create_from_file(img_path)
        detection_result = detector.detect(image)
        dict[id] = (img_path,[landmark for landmark in detection_result.face_landmarks])
        
    return dict
"""
        
def cluster(embeddings):

    # Credits: Arian's pyimagesearch for the clustering code
    # Here we are using the sklearn.DBSCAN functioanlity
    # cluster all the facial embeddings to get clusters 
    # representing distinct people
    
    """
    
    Option - using mediapipe landmarks
    
    for id in dict:
        flat_landmarks = []
        if len(dict[id][1]) != 0:
            for i in range(478):
                flat_landmarks.append(dict[id][1][0][i].x)
                flat_landmarks.append(dict[id][1][0][i].y)
                flat_landmarks.append(dict[id][1][0][i].z)
            landmarks_mat.append(flat_landmarks)
        else:
            for i in range(478):
                flat_landmarks.append(0)
                flat_landmarks.append(0)
                flat_landmarks.append(0)
            landmarks_mat.append(flat_landmarks)
            print("couldn`t find face in image number ", id)
    #print(landmarks_mat)
    """
    
    # cluster the embeddings
    clt = DBSCAN(eps=0.8, metric="euclidean", min_samples=IMGS_PER_FACE+1)
    clt.fit(embeddings)

    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("unique faces: {}".format(numUniqueFaces))

    return clt.labels_

def split_images(labels,faces_path):
    opened_labels = {}
    os.mkdir(os.path.join(faces_path,"clustered"))
    faces = os.listdir(os.path.join(faces_path,"all"))
    for i in range(len(faces)):
        label = labels[i]
        if label not in opened_labels:
            opened_labels[label] = 0
            os.mkdir(os.path.join(faces_path,"clustered",str(label)))
        opened_labels[label] += 1
        src = os.path.join(faces_path,"all",faces[i])
        dst = os.path.join(faces_path,"clustered",str(label),faces[i])
        shutil.copy(src, dst)