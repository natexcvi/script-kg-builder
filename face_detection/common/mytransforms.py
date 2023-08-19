import torch
from torchvision import transforms
from skimage import transform
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(device=device)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(sample, (float(3), new_h, new_w))

        img = torch.from_numpy(img)
        img.type(torch.DoubleTensor)
        return img


class ToTensorWithLabels(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample[0], sample[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}


class Scale(object):
    """ scale the image by a given factor"""

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, tensor):
        return tensor / self.scale_factor.float()


to_tensor = transforms.ToTensor()
normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

