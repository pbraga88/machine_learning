#!/usr/bin/python3

import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import matplotlib.pyplot as graph
import statsmodels.formula.api as smf
from scipy import stats

dataset = pd.read_csv('Data/winequality-red.csv')
dataset.rename(columns = {'total sulfur dioxide':'total_sulfur_dioxide'}, inplace = True)

def PerformLinearRegression(form):
	# Aqui é feita a regressão linear
	
	lm = smf.ols(form, data = dataset).fit()
	
	featureName = form.split(" ")[-1]

	# Colhe os dados para o parâmetro x (parâmetro sabido)
	train_X = dataset[featureName]

	# Para montar e mostrar o gráfico
	intercept = lm.params[0]
	slope = lm.params[1]
	line = slope * train_X + intercept
	graph.plot(train_X, line, '-', c='red')
	graph.scatter(train_X, dataset.pH)
	graph.ylabel('pH')
	graph.xlabel(featureName)
	graph.show()

PerformLinearRegression('pH ~ total_sulfur_dioxide')

# print(array_ols[0])