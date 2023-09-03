from tempfile import NamedTemporaryFile

import pytest
from PIL import Image

from preprocessing import process_image


def test_preprocess_image():
    image_path = "testdata/apples2.jpg"
    img = process_image(image_path)
    assert img.shape == (3, 224, 224)
    assert img.max() <= 255
    assert img.min() >= 0
    preprocessed_path = NamedTemporaryFile()
    Image.fromarray(img.numpy().transpose(1, 2, 0)).save(preprocessed_path.name, "JPEG")
    with open(preprocessed_path.name, "rb") as f:
        preprocessed = f.read()
    with open("testdata/apples2_preprocessed.jpg", "rb") as f:
        expected = f.read()
    assert preprocessed == expected
