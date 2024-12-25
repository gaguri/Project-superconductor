from locust import HttpUser, task


class MyTestUser(HttpUser):
    @task
    def predict_params(self):
        with open('../tests/tests_files/L32_D0.0821_V0.3536_tb0.7561_tp0.5453_tn0.0000_tpn0.0000_J0.2500_results.jpg',
                  'rb') as f:
            binary = f.read()
        files = {"file": ("file", binary, "multipart/form-data")}
        response = self.client.post("/calculate-parameters", files=files)
        print(response)
