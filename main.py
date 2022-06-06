# �������� ���������
import uvicorn
from fastapi import FastAPI
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import pickle as pkl
import numpy as np


# �������� ������ ��� ������������
predict_model = pkl.load(open("saved_model_01.pkl","rb"))


# �������� ���������� ������ FastAPI
app_model_01 = FastAPI()

# �������� ������ ��� ����������� ���� ������� � ��������� ���� ������� ��������
class request_body(BaseModel):
    user_id : int
    mean_score : float



# # �������� �������� ����� Endpoint ��� ��������� ������ ��� ���������������
@app_model_01.post('/predict')
def predict(data: request_body):
    # ���������� ������ � �����, ���������� ��� ���������������
    test_data_a = [[
            data.user_id,
            data.mean_score
    ]]
    
    #user_idx = test_data_a[0]
    test_data_b = np.array(test_data_a).reshape(-1, 1)
    # �������������� ������ (1 - ����� � �����, 0 - �� �����)
    class_idx = predict_model.predict(test_data_b)[1]
    # ������� ���������� ������������
    return {'user_id': data.user_id, 'prediction': class_idx}    


uvicorn.run(app_model_01)
