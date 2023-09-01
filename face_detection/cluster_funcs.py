import io
import os
import math
import time
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\USER\\AppData\\Roaming\\gcloud\\application_default_credentials.json"

IMGS_PER_FACE = 5

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
    
    """ GOOGLE CLOUD VIDEO ANNOTATION API """
    
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
    if not path.startswith("gs"):
        operation = client.annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION],
                "input_content": input_content,
                "video_context": context,
            }
        )
    else: 
        operation = client.annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION],
                "input_uri": input_content,
                "video_context": context,
            }
        )
        

    print("\nProcessing video for face detection annotations.")
    result = operation.result(timeout=1000000)

    print("\nFinished processing.\n")

    # Retrieve the first result, because a single video was processed.
    return result.annotation_results[0]
    
def extract_faces(annotation_result,movie_name,path,output):
    """ Extract faces pictures to a folder """
    if not os.path.exists(output):
        os.mkdir(output)
    faces_dir = os.path.join(output,"Faces")
    if os.path.exists(faces_dir):
        shutil.rmtree(faces_dir)
        time.sleep(0.5)
    os.mkdir(faces_dir)
    all_dir = os.path.join(faces_dir,"all")
    if os.path.exists(all_dir):
        shutil.rmtree(all_dir)
        time.sleep(0.5)
    os.mkdir(all_dir)
    
    # TODO movie = UPLOAD FROM GOOGLE CLOUD
    #movie = VideoFileClip("D:\\OneDrive\\OneDrive - mail.tau.ac.il\\python\\face_detection\\12.Y34r5.4.5l4v3.13.br.sdm0v13sp01nt.c0m.mp4")
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
                    #if i < 5:
                        #print(obj.time_offset.seconds + obj.time_offset.microseconds / 1e6)
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

def mp_detect_and_extract(path,output):
    
    if os.path.exists(output):
        shutil.rmtree(output)
        time.sleep(0.5)
    os.mkdir(output)
    faces_dir = os.path.join(output,"Faces")
    os.mkdir(faces_dir)
    all_dir = os.path.join(faces_dir,"all")
    os.mkdir(all_dir)
    
    print("opened main dir")
    
    # Create an FaceDetector object.
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
        running_mode=VisionRunningMode.VIDEO, min_detection_confidence=0.2,min_suppression_threshold=0)
    
    print("created detector")
    
    #load the video file
    movie = cv2.VideoCapture(path)
    frame_index = 0
    total_faces = 0
    video_file_fps = movie.get(cv2.CAP_PROP_FPS)
    
    print("loaded movie ", movie, " fps = ", video_file_fps)
    with FaceDetector.create_from_options(options) as detector:
        while movie.isOpened:
            frame_index +=1
            ret, frame = movie.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
            frame_timestamp_ms = int(1000 * frame_index / video_file_fps)
            # Perform face detection on the provided single image.
            # The face detector must be created with the video mode.
            face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            face_index = 0
            print("detected ", len(face_detector_result.detections), " faces in frame number ", frame_index)
            for detection in face_detector_result.detections:
                face_index += 1
                total_faces += 1
                bbox = detection.bounding_box
                # Cropping an image
                cropped_image = frame[bbox.origin_y:bbox.origin_y + bbox.height, bbox.origin_x:bbox.origin_x + bbox.width]
                face_path = os.path.join(all_dir,str(frame_index)+"_"+str(face_index)+".jpg")
                # Save the cropped image
                cv2.imwrite(face_path, cropped_image)
                
                #print("frame: ", frame_index, " face: ", face_index, " saved, bbox: ", bbox, " cofidence: ", detection.categories[0].score)
        
    print(total_faces, " faces extracted")
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
    clt = DBSCAN(eps=0.75, metric="euclidean", min_samples=IMGS_PER_FACE+1)
    clt = clt.fit(embeddings)

    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("unique faces: {}".format(numUniqueFaces))

    return clt.labels_

def split_images(labels,faces_path):
    opened_labels = {}
    clusters_path = os.path.join(faces_path,"clustered")
    if os.path.exists(clusters_path):
        shutil.rmtree(clusters_path)
        time.sleep(0.5)
    os.mkdir(clusters_path)
    faces = os.listdir(os.path.join(faces_path,"all"))
    
    for i in tqdm(range(len(faces)), desc = "faces clustering"):
        label = labels[i]
        if label not in opened_labels:
            opened_labels[label] = 0
            os.mkdir(os.path.join(faces_path,"clustered",str(label)))
        opened_labels[label] += 1
        src = os.path.join(faces_path,"all",faces[i])
        dst = os.path.join(faces_path,"clustered",str(label),faces[i])
        shutil.copy(src, dst)
        
def cluster_naming(characters_embeddings, embeddings, output):
    faces_path = os.path.join(output,"Faces")
    characters_path = os.path.join(output,"characters")
    clusters_path = os.path.join(faces_path,"clustered")
    characters = os.listdir(characters_path)
    for cluster in os.listdir(clusters_path):
        if cluster != "-1":
            cluster_path = os.path.join(clusters_path,cluster)
            best_dist = np.inf
            for character_i in range(len(characters)):
                avg_dist = 0
                character_embedding = characters_embeddings[character_i]
                character_name = os.path.splitext(characters[character_i])[0]
                if avg_dist == 0:
                    print("embedding for character ", character_name, ":", character_embedding)
                for img_i in tqdm(range(len(os.listdir(cluster_path))), desc = "calculating distance for cluster "+str(cluster)+" from character "+str(character_name)):
                    if avg_dist == 0:
                        print("embedding for: ", img_i, ":", embeddings[img_i])
                    avg_dist += np.linalg.norm(embeddings[img_i]-character_embedding)
                avg_dist = avg_dist/len(os.listdir(cluster_path))
                print("avg dist: ", avg_dist)
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_character = character_name
                    print("!!! NEW BEST CHARACTER")
            os.rename(cluster_path,os.path.join(clusters_path,cluster+" "+best_character))
            print("CHANGED CLUSTER ", cluster, " TO: ", best_character)
            
                    