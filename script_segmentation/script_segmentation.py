"""
Most of the code inspired by AdeboyeML/Film_Script_Analysis
https://github.com/AdeboyeML/Film_Script_Analysis
"""

from collections import Counter

import glob
import os
import shutil
import random
import sys

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import tools
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
import plotly.express as px


from bs4 import  BeautifulSoup, SoupStrainer
import httplib2
import pprint
import re
import string
import argparse
from tqdm import tqdm
import concurrent.futures as cf

# Parent Directory path
parent_dir = "D:\OneDrive\OneDrive - mail.tau.ac.il\python\script segmentation\movies"

"""
#####Extract the film script URL LIst - unused

def extract_url(movie, url):
    http = httplib2.Http()
    #status, response = http.request('http://www.imsdb.com/all%20scripts/')
    status, response = http.request(url)
    url_list = []
    for link in BeautifulSoup(response, parse_only= SoupStrainer('a')):
        if link.has_attr('href'):
            url_in = link['href']
            if '/Movie Scripts/' in url_in:
                if link.has_attr('title'):
                    url_in = link['title']
                    url_in = url_in.replace(' ', '-')
                    url_in = re.sub(r'\:', '', url_in)
                    url_in = re.sub(r'-Script', '', url_in)
                    url_in = 'http://www.imsdb.com/scripts/' + url_in + '.html'
                    url_list.append(url_in)
                    #print(url_in)
    #len(url_list)
    
    ###Write the FILM SCRIPTS INTO TEXT FILES 
    script_list =[]
    http = httplib2.Http()
    film_list= url_list
    for index in range(len(film_list)):
        film_name  = film_list[index].strip('http://www.imsdb.com/scripts/')
        film_name = film_name.replace('.html','')
        film_name = film_name.replace(':', '')
        print(film_name)
        status, response = http.request(film_list[index])
        
        filename ='./imsdbfilmscripts/'+film_name+'.txt'
        for link in BeautifulSoup(response, parse_only= SoupStrainer('pre')):
            script_list = link.text
            with open(filename, "w", encoding='utf-8', errors='ignore') as f:
                for s in script_list:
                    f.write(s)
"""

##########----SEGMENTING THE FILM SCRIPTS INTO DIFFERENT SCENES SEGMENTS------#######
##########################################################################
############ EXTRACT OUT SCENES AND CHARACTERS ##########

def extract_scene_characters(movie, filename):
    
    # read the data into a list (each row is one list element)
    with open(filename, "r", encoding='utf-8', errors='ignore') as f:
        data = [row for row in f]
    
    dat = []
    for x in data:
        x = re.sub(r'\(.*\)', '', x)
        x = re.sub(r'\-|\#\d+', '', x)
        #x = re.sub(r"[^a-zA-Z0-9.,?'\n ]+", '', x)
        x = re.sub(r"POINT OF VIEW", 'Point of view', x)
        x = re.sub(r"TEXT", 'Text', x)
        x = re.sub(r"NEXT", 'Next', x)
        dat.append(x.replace('\t', ' ').lstrip(" "))
    
    scenes = []
    for l in dat:
        match = re.search(r'(((INT\.|EXT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s+[A-Z]+.*)|((INT\.|EXT\.)\s[A-Z]+)|((INT\.|EXT\.)\s[0-9]+.*)|\
        ((INT\./EXT\.|EXT\./INT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s[0-9]+)|((INT\./EXT\.|EXT\./INT\.)\s[0-9]+.*)|(INT\.\s+.*|EXT\.\s+.*)\
        |((INT\.|EXT\.)\s[A-Z]+\W+.+)|((INT|EXT)\s[A-Z]+.*)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s[A-Z]+)|((INT|EXT)\s[0-9]+.*)\
        |((INT/EXT|EXT/INT)\s[A-Z]+.*)|((INT|EXT)\s[0-9]+)|((INT/EXT|EXT/INT)\s[0-9]+.*)|((I/E\.|E/I\.)\s+[A-Z].*)\
        |((INT|EXT)\s[A-Z]+\W+.+)|((I/E\.|E/I\.)\s+.*))\n', l)
        if match:
            curr_scene = match.group(1)
            scenes.append(curr_scene)
    #scenes = [x.strip(" ") for x in scenes]
    
    sc_text = clean_text(filename)
    write_scenes_manager(movie, scenes, sc_text)
    
    """
    #TODO - understand from here
    characters = []
    for x in dat:
        xters = re.findall(r'(^[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\s+\n)|(^[A-Z]+\.\s+[A-Z]+\n)|(^[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s\n)\
        |(^[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+[A-Z]+[A-Z]+\n)\
        |(^[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\n)|(^[A-Z]+[A-Z]+\'S\s+[A-Z]+[A-Z]+\s+\n)|(^MR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\n)\
        |(^[A-Z]+[A-Z]+\s+\&\s+[A-Z]+[A-Z]+\n)|(^MR\s+[A-Z]+[A-Z]+|MRS\s+[A-Z]+[A-Z]+\s+\n)', x)
        characters.append(xters)
        
    characters = [x for x in characters if x != []]
    refined_characters = []
    for c in characters:
        cc = [tuple(filter(None, i)) for i in c]
        refined_characters.append(cc)
    refined_xters = [x[0][0] for x in refined_characters]
    
    best_ = ['BEST DIRECTOR', 'BEST ADAPTED SCREENPLAY', 'BROADCASTING STATUS', 'BEST COSTUME DESIGN', 'TWENTIETH CENTURY FOX', 'BEST ORIGINAL SCORE', 'BEST ACTOR', 'BEST SUPPORTING ACTOR', 'BEST CINEMATOGRAPHY', 'BEST PRODUCTION DESIGN', 'BEST FILM EDITING', 'BEST SOUND MIXING', 'BEST SOUND EDITING', 'BEST VISUAL EFFECTS']
    transitions = ['RAPID CUT TO:', 'TITLE CARD', 'FINAL SHOOTING SCRIPT', 'CUT TO BLACK', 'CUT TO:', 'SUBTITLE:', 'SMASH TO:', 'BACK TO:', 'FADE OUT:', 'END', 'CUT BACK:', 'CUT BACK', 'DISSOLVE TO:', 'CONTINUED', 'RAPID CUT', 'RAPID CUT TO', 'FADE TO:', \
                   'FADE IN:', 'FADE OUT:', 'FADES TO BLACK', 'FADE TO', 'CUT TO', 'FADE TO BLACK', 'FADE UP:', 'BEAT', 'CONTINUED:', 'FADE IN', \
                   'TO:', 'CLOSE-UP','WIDE ANGLE', 'WIDE ON LANDING', 'THE END', 'FADE OUT','CONTINUED:', 'TITLE:', 'FADE IN','DISSOLVE TO','CUT-TO','CUT TO', 'CUT TO BLACK',\
                   'INTERCUT', 'INSERT','CLOSE UP', 'CLOSE', 'ON THE ROOF', 'BLACK', 'BACK IN SILENCE', 'TIMECUT', 'BACK TO SCENE',\
                   'REVISED', 'PERIOD', 'PROLOGUE', 'TITLE', 'SPLITSCREEN.', 'BLACK.',\
                   'FADE OUT', 'CUT HARD TO:', 'OMITTED', 'DISSOLVE', 'WIDE SHOT', 'NEW ANGLE']
    movie_characters = []
    for x in refined_xters:
        x = re.sub(r'INT\..*|EXT\..*', '', x)
        x = re.sub(r'ANGLE.*', '', x)
        trans = re.compile("({})+".format("|".join(re.escape(c) for c in transitions)))
        x = trans.sub(r'', x)
        best = re.compile("({})+".format("|".join(re.escape(c) for c in best_)))
        x = best.sub(r'', x)
        movie_characters.append(x.replace('\n', '').strip())
    movie_characters = [x.strip() for x in movie_characters if x]
    """
    movie_characters = [] #TODO - delete
    return scenes, movie_characters




#######################################################################
######## CLEAN THE FILM SCRIPT NOT THOROUGHLY THOUGH ##############

def clean_text(filename):
    """
    Applies some pre-processing on the given text.
    """
    with open(filename, "r", encoding='utf-8', errors='ignore') as r:
        text = [row for row in r]
        
    #REMOVE TRANSITIONS OR CAMERA ANGLES
    transitions = ['SMASH CUT TO:', 'FINAL SHOOTING SCRIPT', 'CUT TO BLACK', 'SMASH TO:', 'RAPID CUT TO:', 'BACK TO:', 'BLACK SCREEN', 'FADE OUT TO WHITE LIGHT', 'CUT TO:', 'CUT BACK:', 'CUT BACK', 'DISSOLVE TO:', 'CONTINUED', 'RAPID CUT', 'RAPID CUT TO', 'FADE TO:', \
                   'FADE IN:', 'FADES TO BLACK', 'FADE TO', 'CUT TO', 'FADE UP:', 'BEAT', 'AFTERNOON', 'EVENING', 'CONTINUED:', 'FADE IN', \
                   'TO:', 'CLOSE-UP','WIDE ANGLE','CONTINUED:', 'TITLE:', 'FADE IN','DISSOLVE TO','CUT-TO','CUT TO', 'CUT TO BLACK',\
                   'INTERCUT', 'INSERT', 'CLOSE UP', 'TITLE CARD', 'PAUSE', 'SOUND', 'SONG CONTINUES OVER', 'BACK TO SCENE',\
                   'CUT', 'WATCH', 'CU WATCH', 'BLACK', 'BACK IN SILENCE', 'SUBTITLE:', 'CLOSE', 'ON THE ROOF','CUT HARD TO:',\
                   'THE SCREEN', 'TITLE', 'PROLOGUE', 'SPLITSCREEN.', 'OMITTED', 'BLACK.',\
                   'FADE OUT:', 'FADE OUT.', 'FADE OUT', 'DISSOLVE', 'NEW ANGLE', 'WIDE SHOT']
    # remove directors or the film production company
    best_ = ['BEST DIRECTOR', 'BEST ADAPTED SCREENPLAY', 'SENTENCE', 'BROADCASTING STATUS', 'BEST COSTUME DESIGN', 'TWENTIETH CENTURY FOX', 'BEST ORIGINAL SCORE', 'BEST ACTOR', 'BEST SUPPORTING ACTOR', 'BEST CINEMATOGRAPHY', 'BEST PRODUCTION DESIGN', 'BEST FILM EDITING', 'BEST SOUND MIXING', 'BEST SOUND EDITING', 'BEST VISUAL EFFECTS']
    #text = re.sub('\d+', '', text)
    tex = []
    for x in text:
        tx = x.replace('\t', ' ').lstrip(" ")
        tx = re.sub(r'^\d+\n', r'', tx)
        #tx = re.sub(r'\(.*\)', r'', tx) #TODO aborted
        #tx = re.sub(r'\#\d+', r'', tx) #TODO aborted
        #tx = tx.replace('\n', '')
        #tx = re.sub(r'\d+', r'', tx)
        tx = re.sub(r'(((INT\.|EXT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s+[A-Z]+.*)|((INT\.|EXT\.)\s[A-Z]+)|((INT\.|EXT\.)\s[0-9]+.*)|\
        ((INT\./EXT\.|EXT\./INT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s[0-9]+)|((INT\./EXT\.|EXT\./INT\.)\s[0-9]+.*)|(INT\.\s+.*|EXT\.\s+.*)\
        |((INT\.|EXT\.)\s+[A-Z]+\W+.+)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s[A-Z]+)|((INT|EXT)\s[0-9]+.*)|\
        ((INT/EXT|EXT/INT)\s+[A-Z]+.*)|((INT|EXT)\s+[0-9]+)|((INT/EXT|EXT/INT)\s+[0-9]+.*)|((I/E\.|E/I\.)\s+[A-Z].*)\
        |((INT|EXT)\s+[A-Z]+\W+.+)|((I/E\.|E/I\.)\s+.*))', 'SCC', tx)
        tx = re.sub(r'((INT|EXT)\s+)','SCC',tx) #the_shawshank_redemption special script
        tx = re.sub(r'(^\d+\w+\.\s\n)|(^\d+\.\s\n)|(^\d+\.\n)', r'', tx)
        #tx = re.sub(r'^\W+', r'', tx) #TODO aborted, need?
        tx = re.sub(r'^\d+\.', r'', tx)
        tx = re.sub(r'^\d+/\d+/\d+', r'', tx)
        tx = re.sub(r'ANGLE.*', '', tx)
        tx = re.sub(r'(\'m|\’m)', r' am', tx)
        tx = re.sub(r'(\'ll|\’l)', r' will', tx)
        tx = re.sub(r'(\'re|\’re)', r' are', tx)
        tx = re.sub(r'(\'d|\’d)', r' had', tx)
        tx = re.sub(r'(\'ve|\’ve)', r' have', tx)
        tx = re.sub(r'SEQ\.\s+\d+', r'', tx)
        #tx = re.sub(r'Final\s+\d+\.', r'', tx)
        tx = re.sub(r'Goldenrod\s+\-\s+\d+\.\d+\.\d+\s+\d+\.', r'', tx)
        tx = re.sub(r'(^\d+\s+\d+\s+\d+\s+\-\sRev\.\s\d+/\d+/\d+\s+\d+[A-Z])|(^\d+\s+\d+\s+\d+\s+\-\sRev\.\s\d+/\d+/\d+\s+\d+)', '', tx)
        tx = re.sub(r'([A-Z]+[A-Z]+\sREV\s\d+\-\d+\-\d+\s\d+\.)|([A-Z]+[A-Z]+\sREV\s\d+\-\d+\-\d+\s\d+[A-Z]\.)|(DBL\.\s[A-Z]+[A-Z]+\sREV\s\d+\-\d+\-\d+\s\d+\.)', '', tx)
        #tx = re.sub(r'^TITLE:\n', '', tx)
        #end = re.compile(r'THE END.*|FADE OUT.*', re.MULTILINE)
        #tx = end.sub(r'', tx)
        trans = re.compile("({})+".format("|".join(re.escape(c) for c in transitions)))
        tx = trans.sub(r'', tx)
        #tx = re.sub(r'[A-Z]+\'S', '', tx)
        #tx = tx.replace('[^a-zA-Z]', '')
        #tx = tx.replace('', '')
        #tx = tx.strip()
        #tx = re.sub(r'\d+', r'', tx)
        tx = re.sub(r"[^a-zA-Z0-9.,!?'&\n ]+", '', tx)
        #tx = re.sub(r'\W+', ' ', tx)
        tex.append(tx)
    txt = "".join([s for s in tex if s.strip()])
    txt = re.sub(r'\nTHE END\n(.|\n)*', '', txt)
    
    return txt

def write_scenes_manager(movie, scenes, sc_text):
    executor = cf.ThreadPoolExecutor(max_workers=10)
    
    movie_path = os.path.join(parent_dir, movie)
    os.mkdir(movie_path)
    print("Opened dir for the movie ", movie, ": ", movie_path)
    
    scene_data = sc_text.split('SCC')[1:]

    
    for scene_num in tqdm(range(len(scene_data))):
        curr_scene = scenes[scene_num]
        curr_data = scene_data[scene_num]
        future = executor.submit(write_scene, movie_path, scene_num, curr_scene, curr_data)
    executor.shutdown()
       
def write_scene(movie_path, scene_num, curr_scene, curr_data):    
        with open(movie_path + "\\" + str(scene_num+1) + ".txt", 'w') as o:
            o.write(curr_scene + "\n" + curr_data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentize moviescript to scenes, using regexes/')
    parser.add_argument('movie', metavar ='m', help='The name of the movies, without extensions, will be the name of the scenes folder')
    args = parser.parse_args()
    movie = args.movie
    parser.add_argument('--path', default=str(parent_dir+"\\"+movie+".txt"), metavar = 'p', help='path to the movie\'s script, txt file. default=parent_dir/movies_name.txt')
    args = parser.parse_args()
    path = args.path
    #if args.path.startswith("http"):
    #    path = extract_url(movie, path) #TODO - ensure nothing get back.
    #else:
    print("Segmentaizing the movie: ", movie, " from path: ", path)
    scenes, movie_xters = extract_scene_characters(movie, path)
        
    """
    films = []
    dest = 'Films_not_analyzed/'
    for f in glob.glob('./imsdbfilmscripts\*'):  
        film_name = re.sub(r'.txt|\./imsdbfilmscripts\\', '', f)
        films.append(film_name)
        sc, movie_xters = extract_scene_characters(f)
        if len(sc) > 1:
            sc_text = clean_text(f)
            df_sc, df_xtrs, df_dia = characters_dialogue_action(sc_text, sc, movie_xters)
            df_sc.to_pickle('Films/' + film_name + '.pkl')
            df_xtrs.to_pickle('Characters/' + film_name + '.pkl')
            df_dia.to_pickle('Dialogues/' + film_name + '.pkl')
            print(film_name + ' IS GOING TO BE ANALYZED \n')
        else:
            shutil.copy(f, dest)
            print(film_name + ' IS NOT GOING TO BE ANALYZED \n')   
    """