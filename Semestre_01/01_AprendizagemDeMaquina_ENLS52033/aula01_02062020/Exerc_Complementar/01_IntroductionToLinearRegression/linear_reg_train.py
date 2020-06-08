#!/usr/bin/python3
import pandas as pd
import numpy as np


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

# clf = LinearRegression(normalize=True)

# clf.fit(x_train,y_train) # Linha de código original: Esta linha faz o treinamento, 
						 # efetivamente
# clf.fit(np.nan_to_num(x_train),np.nan_to_num(y_train)) # Usado quando existe um 
														 # NaN (Not a Number ) na basa de
														 # dados
# y_pred = clf.predict(x_test)
# print(r2_score(y_test,y_pred))

n = 700
alpha = 0.0001

a_0 = np.zeros((n,1))
a_1 = np.zeros((n,1))

epochs = 0
while(epochs < 20):
    y = a_0 + a_1 * x_train[:, None]

    print(epochs) #FOR DEBUG

    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    a_0 = a_0 - alpha * 2 * np.sum(error)/n 
    a_1 = a_1 - alpha * 2 * np.sum(error * x_train[:, None])/n
    epochs += 1
    if(epochs%10 == 0):
        print("mean square root: ")
        print(mean_sq_er)

import matplotlib.pyplot as plt 

y_prediction = a_0 + a_1 * x_test[:, None]
print('R2 Score:',r2_score(y_test,y_prediction))

y_plot = []
for i in range(100):
    y_plot.append(a_0 + a_1 * i)
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red',label='GT')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
plt.legend()
plt.show()