import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
from keras.utils import img_to_array

from UNET.dataset_common_functions import get_parameters_from_filename
from tests.module_tests import get_model


def test_model_has_good_accuracy():
    path = './tests_files'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    model = get_model()
    for f in files:
        D, V, tb, tp = get_parameters_from_filename(f)
        image = Image.open(os.path.join(path, f))
        image = image.resize((128, 128))
        np_image = img_to_array(image, data_format="channels_first") / 255.0

        prediction = model.predict(np.array([np_image]))[0]

        assert abs(prediction[0] - D) < 0.03
        assert abs(prediction[1] - V) < 0.03
        assert abs(prediction[2] - tb) < 0.04
        assert abs(prediction[3] - tp) < 0.03

