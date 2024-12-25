import numpy as np
import pytest
from PIL import Image
from keras.utils import img_to_array

from UNET.dataset_common_functions import get_parameters_from_filename
from UNET.model_common_functions import unet_full_model


def get_model():
    model = unet_full_model()
    model.load_weights('./UNET/best_model.h5')
    return model


def test_model_should_return_four_parameters():
    model = get_model()
    image = np.random.random((1, 3, 128, 128))

    prediction = model.predict(image)

    assert len(prediction) == 1
    assert len(prediction[0]) == 4
    assert prediction[0].dtype == 'float32'


@pytest.mark.parametrize("filename, expected_D, expected_V, expected_tb, expected_tp", [
    ("L32_D0.0821_V0.3597_tb0.7500_tp0.4242_tn0.0000_tpn0.0000_J0.2500_results.png", 0.0821, 0.3597, 0.7500, 0.4242),
    ("L32_D0.2800_V0.3650_tb0.7700_tp0.4500_tn0.0000_tpn0.0000_J0.2500_results.png", 0.2800, 0.3650, 0.7700, 0.4500),
    ('L32_D0.2879_V0.3669_tb0.7924_tp0.4000_tn0.0000_tpn0.0000_J0.2500_results.png', 0.2879, 0.3669, 0.7924, 0.4000)])
def test_should_correctly_get_parameters_from_filename(filename, expected_D, expected_V, expected_tb, expected_tp):
    D, V, tb, tp = get_parameters_from_filename(filename)

    assert D == expected_D
    assert V == expected_V
    assert tb == expected_tb
    assert tp == expected_tp

def test_predict_parameters_success():
    image = Image.open('./tests/tests_files/L32_D0.0821_V0.3536_tb0.7561_tp0.5453_tn0.0000_tpn0.0000_J0.2500_results.jpg')
    image = image.resize((128, 128))
    np_image = img_to_array(image, data_format = "channels_first") / 255.0
    model = get_model()

    predicted_params = model.predict(np.array([np_image]))

    assert isinstance(predicted_params, np.ndarray)
    assert predicted_params.shape == (1,4)
    assert all(-1.0 <= x <= 1.0 for x in predicted_params[0])