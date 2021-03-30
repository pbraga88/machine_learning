#!/usr/bin/python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt 

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


clf = LinearRegression(normalize=True)

clf.fit(x_train,y_train) # Linha de código original: Esta linha faz o treinamento, 
						 # efetivamente
# clf.fit(np.nan_to_num(x_train),np.nan_to_num(y_train)) # Usado quando existe um 
														 # NaN (Not a Number ) na basa de
														 # dados
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred)) # Taxa de erro

# print(y_pred) #FOR DEBUG

y_plot = []
for i in range(100):
    for j in range(100):
        if x_test[j] == i:
            y_plot.append(y_pred[j])
            # print("test") #FOR DEBUG
            # print(y_pred[count]) #FOR DEBUG
    # print(i) #FOR DEBUG
#print(y_plot) #FOR DEBUG

plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red',label='GT')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
plt.legend()
plt.show()