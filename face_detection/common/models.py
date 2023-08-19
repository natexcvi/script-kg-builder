import torch
from facenet_pytorch import InceptionResnetV1
import torchvision
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


def get_vgg_pretrained_vggface2(weights_path, return_layer='classifier.4'):
    model = torchvision.models.vgg16().eval()
    model.features = torch.nn.DataParallel(model.features)
    model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=8749)
    weights = torch.load(weights_path, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(weights)
    return_layers = {return_layer: 'output'}
    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
    return mid_getter


def get_vgg_pretrained_imagenet(return_layer='classifier.4', return_layer_new_name='fc7'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights='DEFAULT').eval()
    return_layers = {return_layer: return_layer_new_name}
    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
    return mid_getter


def get_vgg(num_classes=1000, pretrained=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights='DEFAULT').eval()
    model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    return model

def get_resnet_model(pretrain='vggface2'):
    # options are: vggface2, casia-webface https://github.com/timesler/facenet-pytorch
    model = InceptionResnetV1(pretrained=pretrain).eval()
    return model
