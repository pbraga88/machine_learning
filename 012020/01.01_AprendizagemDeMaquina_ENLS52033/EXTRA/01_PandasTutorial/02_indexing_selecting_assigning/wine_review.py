#!/usr/bin/python3

import pandas as pd

wine_reviews = pd.read_csv("../Data/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

# print(wine_reviews.head())

# # Para acessar uma propriedade do objeto 'wine_reviews' pode-se:
# print(wine_reviews.country)		# Pela propriedade
# print(wine_reviews['country'])	# Usando dicionário

# # Para acessar um campo específico da coluna, pode-se utilizar o operador de 
# # indexação '[]':
# for i in range(100):
# 	print(wine_reviews.country[i])
	# print(wine_reviews['country'][i])

''' ===== iloc -  Seleção baseada em indexadores numéricos) ====='''
# Para acessar a linha utiliza-se o operador iloc, da seguinta maneira:
# print(wine_reviews.iloc[0]) # Acessando a primeira linha 

# # 'iloc' e 'loc' funcionam como '[row][column]', que é o oposto do python nativo 
# # que é [column][row]. Veja esse exemplo para acessar uma coluna utilizando iloc:
# print(wine_reviews.iloc[:,0]) # Printa a primeira coluna "inteira"
# print(wine_reviews.iloc[:,0][2]) # Printa o segundo elemento da primeira coluna

# O operador ':' significa "tudo" e, quando um valor é passado, ele pode ser
# manipulado. Veja os exemplos:
# print(wine_reviews.iloc[:3, 0]) # Printa as 3 primeiras linhas da primeira coluna
# print(wine_reviews.iloc[2:5, 0]) # Printa a primeira coluna da linha 3 até a 5

# Também é possível passar uma uma lista para iloc:
# print(wine_reviews.iloc[[0,4,28], 0]) # Printa as linhas 0, 4 e 28

# É possível ainda imprimir à partir do final da lista, utilizando-se de números 
# negativos:
# print(wine_reviews.iloc[-5:, 0]) # Printa as últimas 5 linhas da primeira coluna

''' ===== loc - Seleção baseada em 'labels' ====='''
# Neste tipo de seleção, é o indexador 'label' que importa. Veja o exemplo:
print(wine_reviews.loc[0, 'country'])


print(wine_reviews.loc[:,['taster_name', 'taster_twitter_handler',\
						  'points'] ]) # Printa todas as linhas das colunas referidas
print(wine_reviews.loc[:3,['taster_name', 'taster_twitter_handler',\
						  'points'] ]) # Printa as 3 primeiras linhas das 
									   # colunas referidas
print(wine_reviews.loc[2:5,['taster_name', 'taster_twitter_handler',\
						  'points'] ]) # Printa da linha 3 até 5 das colunas referidas
