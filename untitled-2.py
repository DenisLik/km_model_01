import numpy as np
from sklearn.linear_model import LinearRegression


x = np.array([2.0 , 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7])
y = np.array([196, 221, 136, 255, 244, 230, 232, 255, 267])

lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y)

print(lr.predict([[2.4]]))