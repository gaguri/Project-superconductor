from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_read_api_is_runninng():
    response = client.get("/")

    assert response.status_code == 200


def test_api_send_422_on_invalid_request():
    response = client.post("/calculate-parameters")

    assert response.status_code == 422

def test_api_accept_invalid_form_data_request():
    files = {"file": ("file", [0,1,2], "multipart/form-data")}

    response = client.post("/calculate-parameters", files=files)

    assert response.status_code == 400


def test_api_predict_parameters():
    with open('../tests/tests_files/L32_D0.0821_V0.3536_tb0.7561_tp0.5453_tn0.0000_tpn0.0000_J0.2500_results.jpg', 'rb') as f:
        binary = f.read()
    files = {"file": ("file", binary, "multipart/form-data")}

    response = client.post("/calculate-parameters", files=files)

    result = response.json()
    assert response.status_code == 200
    assert 'params' in result
    assert len(result['params']) == 4
