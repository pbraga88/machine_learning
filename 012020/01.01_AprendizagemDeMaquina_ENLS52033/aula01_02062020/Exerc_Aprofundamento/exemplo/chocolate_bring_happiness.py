#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import matplotlib.pyplot as graph
import statsmodels.formula.api as smf
from scipy import stats

dataset = pd.read_csv('Data/chocolate_data.txt', index_col=False, sep="\t", header=0)

print(dataset.head())

def PerformLinearRegression(formula):
	# Aqui é feita a regressão linear
	lm = smf.ols(formula = formula, data = dataset).fit()

	featureName = formula.split(" ")[-1]

	# Colhe os dados para o parâmetro x (parâmetro sabido)
	train_X = dataset[featureName]

	# Para montar e mostrar o gráfico
	intercept = lm.params[0]
	slope = lm.params[1]
	line = slope * train_X + intercept
	graph.plot(train_X, line, '-', c='red')
	graph.scatter(train_X, dataset.customer_happiness)
	graph.ylabel('customer_happiness')
	graph.xlabel(featureName)
	graph.show()

PerformLinearRegression('customer_happiness ~ cocoa_percent')