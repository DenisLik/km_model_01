# загрузка библиотек
import uvicorn
from fastapi import FastAPI
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import pickle as pkl
import numpy as np


# загрузка модели для предсказания
predict_model = pkl.load(open("saved_model_01.pkl","rb"))


# создание экзампляра класса FastAPI
app_model_01 = FastAPI()

# Создание класса для определения тела запроса и подсказок типа каждого атрибута
class request_body(BaseModel):
    user_id : int
    mean_score : float



# # Создание конечной точки Endpoint для получения данных для прогнозирования
@app_model_01.post('/predict')
def predict(data: request_body):
    # Приведение данных в форму, подходящую для прогнозирования
    test_data_a = [[
            data.user_id,
            data.mean_score
    ]]
    
    #user_idx = test_data_a[0]
    test_data_b = np.array(test_data_a).reshape(-1, 1)
    # Предсказывание класса (1 - готов к сдаче, 0 - не готов)
    class_idx = predict_model.predict(test_data_b)[1]
    # Возврат результата предсказания
    return {'user_id': data.user_id, 'prediction': class_idx}    


uvicorn.run(app_model_01)
