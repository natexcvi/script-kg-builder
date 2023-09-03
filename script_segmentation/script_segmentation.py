"""
Most of the code inspired by AdeboyeML/Film_Script_Analysis
https://github.com/AdeboyeML/Film_Script_Analysis
"""
import os
import re
import argparse
from tqdm import tqdm
import concurrent.futures as cf
import time
import shutil


##########----SEGMENTING THE FILM SCRIPTS INTO DIFFERENT SCENES SEGMENTS------#######
##########################################################################
############ EXTRACT OUT SCENES ##########

def extract_scenes(movie, filename):
    
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
    
    sc_text = clean_text(filename)
    write_scenes_manager(movie, scenes, sc_text)
    
    return scenes

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
        #tx = re.sub(r'\(.*\)', r'', tx)
        #tx = re.sub(r'\#\d+', r'', tx) 
        #tx = tx.replace('\n', '')
        #tx = re.sub(r'\d+', r'', tx)
        tx = re.sub(r'(((INT\.|EXT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s+[A-Z]+.*)|((INT\.|EXT\.)\s[A-Z]+)|((INT\.|EXT\.)\s[0-9]+.*)|\
        ((INT\./EXT\.|EXT\./INT\.)\s[A-Z]+.*)|((INT\.|EXT\.)\s[0-9]+)|((INT\./EXT\.|EXT\./INT\.)\s[0-9]+.*)|(INT\.\s+.*|EXT\.\s+.*)\
        |((INT\.|EXT\.)\s+[A-Z]+\W+.+)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s+[A-Z]+.*)|((INT|EXT)\s[A-Z]+)|((INT|EXT)\s[0-9]+.*)|\
        ((INT/EXT|EXT/INT)\s+[A-Z]+.*)|((INT|EXT)\s+[0-9]+)|((INT/EXT|EXT/INT)\s+[0-9]+.*)|((I/E\.|E/I\.)\s+[A-Z].*)\
        |((INT|EXT)\s+[A-Z]+\W+.+)|((I/E\.|E/I\.)\s+.*))', 'SCC', tx)
        tx = re.sub(r'((INT|EXT)\s+)','SCC',tx) #the_shawshank_redemption special script
        tx = re.sub(r'(^\d+\w+\.\s\n)|(^\d+\.\s\n)|(^\d+\.\n)', r'', tx)
        #tx = re.sub(r'^\W+', r'', tx)
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
    
    movie_path = os.path.join(os.getcwd(), movie)
    if os.path.exists(movie_path):
        shutil.rmtree(movie_path)
        time.sleep(0.5)
    os.mkdir(movie_path)
    
    scene_data = sc_text.split('SCC')[1:]

    
    for scene_num in tqdm(range(len(scene_data))):
        curr_scene = scenes[scene_num]
        curr_data = scene_data[scene_num]
        future = executor.submit(write_scene, movie_path, scene_num, curr_scene, curr_data)
    print("Saved", str(len(scene_data)+1), " segmentized scenes at: ", movie_path)
    executor.shutdown()
       
def write_scene(movie_path, scene_num, curr_scene, curr_data):    
        with open(movie_path + "\\" + str(scene_num+1) + ".txt", 'w') as o:
            o.write(curr_scene + "\n" + curr_data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentize moviescript to scenes, using regexes/')
    parser.add_argument('movie', metavar ='m', help='The name of the movies, without extensions, will be the name of the scenes folder')
    parser.add_argument('--path', default="path", metavar = 'p', help='path to the movie\'s script, txt file. default=terminal_dir/movies_name.txt')
    args = parser.parse_args()
    movie = args.movie
    path = args.path
    if path == "path":
        path = os.path.join(os.getcwd(),str(movie+".txt"))
    print("Segmentaizing the movie: ", movie, " from path: ", path)
    scenes = extract_scenes(movie, path)