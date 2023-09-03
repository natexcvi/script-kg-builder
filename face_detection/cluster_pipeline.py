""" Inspired from https://github.com/cppxaxa/FaceRecognitionPipeline_GeeksForGeeks"""

from cluster_funcs import *
from face_embedding import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracting faces from video and cluster it by person')
    parser.add_argument('movie', metavar ='movie', help='The name of the movie, without extensions, will be the name of the images and clusters folder')
    parser.add_argument('path', metavar = 'path', help='path to the movie file')
    parser.add_argument('--output', default="output", metavar = 'output', help='path to the dir in which the faces and clusters will be saved')
    args = parser.parse_args()
    movie = args.movie
    path = args.path
    output = args.output
    if output == "output":
        output = os.path.join(os.getcwd(),movie)
        
    #faces_path = mp_detect_and_extract(path,output) # USING MEDIAPIPE FRAME BY FRAME
    
    ''' 1. Retrieve faces boxes and times from a movie '''
    annotation_result = detect_faces(path)
    print("retrieved faces")
    
    ''' 2. Extract faces pictures to a folder '''
    faces_path = extract_faces(annotation_result,movie,path,output)
    print("extreacted faces images to: ", faces_path)
    
    """
    Used to start from dir containing all the images. mark 1 + 2 as remarks
    faces_path = "...\\movie_name\\faces"
    """
    
    ''' 3. extrect embeddings for each face '''
    embeddings = pymain(os.path.join(faces_path,"all"))
    np.save(os.path.join(faces_path,"embeddings"),embeddings)
    print("Extrected embeddings")
    
    """
    Used to start from file containing images embedding. mark 1 + 2 + 3 as remarks
    embeddings = np.load(os.path.join(faces_path,"embeddings.npy"))
    """
    
    ''' 4. Cluster the  faces by their landmarks '''
    labels = cluster(embeddings)
    print("images clustered: ", labels)
    
    ''' 5. Save the photos to different dirs '''
    split_images(labels,faces_path)
    print("images splitted.")