import pickle as pkl
import numpy as np


# загрузка модели для предсказания
predict_model = pkl.load(open("saved_model_01.pkl","rb"))


lst_a = [[0.713, 0.696, 0.754, 0.7, 0.79]]
lst_b = np.array(lst_a).reshape(-1, 1)

a = 0.70
b = np.array(a).reshape(1, -1)
#temp = np.array(temp).reshape((len(temp), 1))

#print(type(predict_model))
print(predict_model.predict(b))

#print(predict_model.predict(lst_b)[0])