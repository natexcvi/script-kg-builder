import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy.linalg import norm
from PIL import Image
import csv
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision import transforms
from common.models import get_vgg_pretrained_vggface2, get_vgg_pretrained_imagenet, get_resnet_model
from common.mytransforms import normalize_imagenet, Rescale

logging.basicConfig(level=logging.INFO)

cos = torch.nn.CosineSimilarity()
composed_transforms = transforms.Compose([transforms.ToTensor(), Rescale((224, 224)), normalize_imagenet])


def load_data(data_dir, num_classes=None, instance_num_per_class=None):
    """
    get list of paths to images in data_dir
    :param data_dir: path to data directory
    :param num_classes: num classes to load. If None, load all classes
    :param instance_num_per_class: num instances per class to load. If None, load all instances
    :return: list of paths to images
    """
    paths_list = []
    names = []
    classes_list = os.listdir(data_dir)
    if num_classes:
        if num_classes == 1:
            imgs = classes_list
            if instance_num_per_class:
                imgs = imgs[:instance_num_per_class]
            for img in imgs:
                im_path = os.path.join(data_dir, img)
                if not os.path.isfile(im_path):
                    continue
                paths_list.append(im_path)
                
            return paths_list, names
        else:
            classes_list = classes_list[:num_classes]
    for c in classes_list:
        class_path = os.path.join(data_dir, c)
        if not os.path.isdir(class_path):
            continue
        imgs = os.listdir(class_path)
        if instance_num_per_class:
            imgs = imgs[:instance_num_per_class]
        for img in imgs:
            im_path = os.path.join(class_path, img)
            paths_list.append(im_path)
            names.append(f'{c}_{img}')

    return paths_list, names


def get_model(model_type, model_weights=None, layer_name=None, layer_size=None):
    if model_type == 'vgg_vggface2':
        if model_weights is None:
            raise TypeError('model weights must be provided for vgg_vggface2')
        if layer_size and layer_name:
            model = get_vgg_pretrained_vggface2(model_weights, return_layer=layer_name)
            embedding_size = layer_size
        else:
            model = get_vgg_pretrained_vggface2(model_weights)
            embedding_size = 4096
    elif model_type == 'vgg_imagenet':
        model = get_vgg_pretrained_imagenet(return_layer='features.2', return_layer_new_name='fc7')
        embedding_size = 64
    elif model_type == 'resnet_vggface2':
        model = get_resnet_model(pretrain='vggface2')
        embedding_size = 512
    elif model_type == 'resnet_casia':
        model = get_resnet_model(pretrain='casia-webface')
        embedding_size = 512
    else:
        raise ValueError('model type not supported')
    return model, embedding_size


def get_mtcnn(model_type):
    if model_type.startswith('resnet'):
        return MTCNN(image_size=160, post_process=False)
    elif model_type.startswith('vgg'):
        return MTCNN(image_size=224, post_process=False)


def get_embeddings(data, model_type, model_path=None, perform_mtcnn=True, layer_name='fc7', layer_size=4096):
    model, embedding_size = get_model(model_type, model_path, layer_name, layer_size)
    embeddings = np.zeros((len(data), embedding_size))
    mtcnn = get_mtcnn(model_type)

    embeddings_dict = {}
    
    for i, im_path in tqdm(enumerate(data), desc = "faces embedding", total=len(data)):
        img = Image.open(im_path)
        if img.mode != 'RGB':  # PNG imgs are RGBA
            img = img.convert('RGB')
        if perform_mtcnn:
            img = mtcnn(img)
            # if mtcnn post_processing = true, image is standardised to -1 to 1, and size 160x160.
            # else, image is not standardised ([0-255]), and size is 160x160
            if img == None:
                continue
            img = img / 255.0
            img = normalize_imagenet(img)
        else:  # in case images are already post mtcnn. In this case need to rescale to 160x160 and normalize
            img = composed_transforms(img)

        if model_type.startswith('vgg'):
            img_embedding = model(img.unsqueeze(0).float())[0]['output']
        else:
            img_embedding = model(img.unsqueeze(0).float())[0]

        embeddings[i] = torch.flatten(img_embedding).detach().numpy()
        """
        id = os.path.splitext(os.path.basename(im_path))[0]
        
        embeddings_dict[id] = (im_path,embeddings[i])
        """
        
    return embeddings


def validate_args(args, save_name):
    data_dir = args.data_dir
    if not args.model_type:
        logging.info('model type not provided, using default resnet pretrained on vggface2')
        model_type = 'resnet_vggface2'
    else:
        model_type = args.model_type
    if model_type == 'vgg_vggface2' and not args.model_path:
        raise TypeError('model weights must be provided for vgg_vggface2')

    if not args.output_path:
        logging.info(f'results path not provided, saving to data_dir {args.data_dir}/rdm.csv')
        output_path = os.path.join(args.data_dir, 'rdm.csv')
    else:
        output_path = args.output_path
        output_path = os.path.join(output_path, f'rdm_{save_name}.csv')

    return data_dir, model_type, args.model_path, output_path

def pymain(data_dir,model_type = None, model_path = None):
    save_name = 'fc7'
    if not model_type:
        logging.info('model type not provided, using default resnet pretrained on vggface2')
        model_type = 'resnet_vggface2'
    else:
        model_type = model_type
    if model_type == 'vgg_vggface2' and not model_path:
        raise TypeError('model weights must be provided for vgg_vggface2')

    data_paths_list, names_list = load_data(data_dir, 1)
    model_embeddings = get_embeddings(data_paths_list, model_type, model_path, perform_mtcnn=True)
    
    return model_embeddings
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")
    parser.add_argument("-model_type", "--model_type", dest="model_type", help="type from models.py", required=False)
    parser.add_argument("-model_path", "--model_path", dest="model_path", help="path to model weights", required=False)
    parser.add_argument("-output_path", "--output_path", dest="output_path", help="csv result dir", required=False)
    args = parser.parse_args()
    save_name = 'fc7'
    data_dir, model_type, model_path, output_path = validate_args(args, save_name)
    data_paths_list, names_list = load_data(args.data_dir)
    print("loaded data")
    model_embeddings = get_embeddings(data_paths_list, model_type, model_path, perform_mtcnn=True)
