#!/usr/bin/python3
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

def linear_regression():
	columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
	# Carrega um dataset 'default' do sklearn
	diabetes = datasets.load_diabetes()

	df = pd.DataFrame(diabetes.data, columns=columns)
	y = diabetes.target # define a variável 'target' como a variável dependente 'y'


	# Criando as variáveis de treinamento e teste
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	# Fit a model
	lm = linear_model.LinearRegression()

	model = lm.fit(X_train, y_train)
	predictions = lm.predict(X_test) # Retorna uma array

	print(predictions[0:5])

	# Print da predição de acordo a linha idade 
	count = X_test.shape[0]
	for i in range(count):
		print(y_test[i],"\t",predictions[i])

	# Acurácia do modelo
	print("Score:", model.score(X_test, y_test))

	# Para plotar o modelo
	plt.scatter(y_test, predictions) # x,y
	plt.xlabel("True values")
	plt.ylabel("predictions")
	plt.show()
# linear_regression()

def kfold_example():
	from sklearn.model_selection import KFold # import KFold
	X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # Create array
	print(X)
	y = np.array([1, 2, 3, 4]) # Create another array
	print(y)
	kf = KFold(n_splits=2) # Define the split into 2 folds
	print(kf.get_n_splits(X)) # returns the number of splitting iterations in the cross-validator
	print(kf)

	for train_index, test_index in kf.split(X):
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print(X_train, "\t", X_test)
		print(y_train, y_test)

# kfold_example()

def leave_one_out_example():
	from sklearn.model_selection import LeaveOneOut 
	X = np.array([[1, 2], [3, 4]])
	y = np.array([1, 2])
	loo = LeaveOneOut()
	loo.get_n_splits(X)

	for train_index, test_index in loo.split(X):
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print(X_train, X_test, y_train, y_test)
# leave_one_out_example()

''' Em datasets maiores é recomendado utilizar kfold (k=3, é um bom exemplo). 
Já em datasets menores, o mais recomendado é o Leave One Out Cross Validation '''

# Regressão linear com 'Cross-Validation'
def linear_regression_with_cv_kfold():
	# Necessary imports: 
	from sklearn.model_selection import cross_val_score, cross_val_predict
	from sklearn import metrics

	columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
	# Carrega um dataset 'default' do sklearn
	diabetes = datasets.load_diabetes()

	df = pd.DataFrame(diabetes.data, columns=columns)
	y = diabetes.target # define a variável 'target' como a variável dependente 'y'


	# Criando as variáveis de treinamento e teste
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	# Fit a model
	lm = linear_model.LinearRegression()

	model = lm.fit(X_train, y_train)
	predictions = lm.predict(X_test) # Retorna uma array

	# print(predictions[0:5])

	# Print da predição de acordo a linha idade 
	# count = X_test.shape[0]
	# for i in range(count):
	# 	print(y_test[i],"\t",predictions[i])

	# Aplicando '6-fold cross validation'
	scores = cross_val_score(model, df, y, cv = 6)
	print("Cross-validated scores:", scores)

	# Fazendo a previsão com 'cross-validation'
	predictions = cross_val_predict(model, df, y, cv=6)

	# Cálculo da acurácia (R^2)
	accuracy = metrics.r2_score(y, predictions)
	print("Cross-Predicted Accuracy", accuracy)

	# Plot do gráfico	
	plt.scatter(y, predictions) # x,y
	plt.xlabel("True values")
	plt.ylabel("predictions")
	plt.show()

linear_regression_with_cv_kfold()