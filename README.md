# Modelling the Semantic Component in Visual Perception of Human Faces by Training a DNN on a Visuo-semantic Knowledge Graph

This repository accompanies our project submitted by Yonatan Elsaesser and Nate Liebmann as part of the _Workshop on Computational Methods in Brain Research_ at Tel Aviv University, 2023, under the supervision of Idan Grossbard, Prof. Galit Yovel & Prof. Amir Globerson.

## Abstract
Recognition of human faces is known to rely on more than merely visual characteristics, with semantic information appearing to play an important role in our mental representation of others. Multi-modal DNNs like CLIP mark a promising direction in modelling human perception, but fail to capture its full semantic depth. We present an end-to-end framework for fine-tuning CLIP on a visuo-semantic knowledge graph derived automatically from a motion picture's scirpt and video. We show, anecdotally, that the learned reprentations express the semantics of the film. More work over a larger dataset is needed to generalise the results.

## Repository Structure
- `script_segmentation` - a helper tool that given a text file containing a movie script, segments it into scenes.
- `face_detection` - a pipeline for annotating and extracting face images from the video of a film, embedding them and clustering them by character.
- The knowledge graph builder is a Go application in the top level directory.
- `model` - contains the training code used to fine-tune CLIP on the knowledge graph.

## Script Segmentation
This project is a tool for segmenting a movie script into scenes.

### Installation
1. Clone the repository.
2. Install the requirements:
```bash
pip install -r requirements.txt
```

### Usage
1. Prepare a text file containing the script of a movie.
2. Make sure your terminal is in the `script_segmentation` directory.
3. Run the script segmentation script:
```bash
python script_segmentation.py <movie name> --path <script file>
```

## Face Detection
This project is a pipeline for annotating and extracting face images from the video of a film, embedding them and clustering them by character.

### Installation
1. Clone the repository.
2. Install the requirements:
```bash
pip install -r requirements.txt
```

### Usage
1. Prepare a video file containing the movie.
2. Make sure your terminal is in the `face_detection` directory.
3. Run the face detection script:
```bash
python cluster_pipeline.py <movie name> <video file> --output <output directory>
```

## ScriptKGBuilder
This project is a script knowledge graph builder.

### Installation
1. Download the binary appropriate for your machine from the latest release.
2. Unzip the binary.

### Usage
1. Prepare a script directory containing scene files in format `<index>.txt`, where `<index>` is a number starting from 0.
2. Make sure your terminal is in the same directory as the binary.
3. If needed, make the binary executable:
```bash
chmod +x script-kg-builder_<platform>
```
4. Run the binary:
```bash
./script-kg-builder_<platform> <script directory> <output file>
```

## Model
This project is a fine-tuning pipeline for CLIP on a visuo-semantic knowledge graph.

### Installation
1. Clone the repository.
2. Install the requirements:
```bash
pip install -r requirements.txt
```

### Usage
1. Prepare a knowledge graph file in the format described in the paper.
2. Make sure your terminal is in the `model` directory.
3. Run the training script:
```bash
python training.py
```