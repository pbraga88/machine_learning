#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('../Data/banking.csv', header=0)
data = data.dropna() #Drop Not Aplicable. Tira linhas com informações faltantes

#=====================================================
# print(data.shape)	# Mostra a quantidade de linhas e colunas, respectivamente
# print(list(data.columns))	# Mostra o nome das colunas

# print(data.head()) #Mostra as 5 primeiras linhas
#=====================================================

#=====================================================
# print(data['education'].unique()) # Mostra os tipos de classificação da coluna 
								  # 'education'
# Agrupando os grupos 'basic.9y', 'basic.6y' e 'basic.4y' em 'Basic'
data['education'] = np.where(data['education']=='basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.4y', 'Basic', data['education'])
# print(data['education'].unique())
#=====================================================

#=====================================================
# print(data['y'].value_counts()) # Conta a quantidade de cada valor da coluna 'y'
#=====================================================

#=====================================================
# Então, podemos plotar no gráfico:
# sns.countplot(x='y', data=data, palette='hls') # Define a base de dados e o 
													   # eixo 'y'
# plt.show()	# Mostra o gráfico na tela
# plt.savefig('count_plot') # opcionalmente, é possível salvar o gráfico
#=====================================================

#=====================================================
# Calculamos então a porcentagem de 'subscription' e 'no subscription':
count_no_sub = len(data[data['y']==0]) # conta os valores '0' da coluna 'y'
count_sub = len(data[data['y']==1]) # conta os valores '1' da coluna 'y'
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub) # Cálculo da porcentagem 
													  # de 'no subscription'
# print("Percentage of no subscription is: ", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_sub+count_no_sub) # Cálculo da porcentagem de 'subscription'
# print("Percentage of subscription is: ", pct_of_sub*100)
#=====================================================

#=====================================================
# Para tirar a média e possíveis conclusões
# print(data.groupby('y').mean()) # Por 'subscription' ou 'no subscription'
# print(data.groupby('job').mean()) # Por emprego
# print(data.groupby('marital').mean()) # Por Estado Civíl
# print(data.groupby('education').mean()) # Por educação
#=====================================================

#=====================================================
# Plotando o gráfico em barra da frequência de subscription em função da variável
# 'job'. 
# Aqui o gráfico mostra, que o tipo de emprego tem relação com a compra 
# ou não do serviço 
# pd.crosstab(data.job,data.y).plot(kind='bar') # Essa linha faz a relação entre a 
									# quantidade de subscriptions 'y' em função da 
									# variável 'job'
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')
# plt.show()
# plt.savefig('purchase_fre_job')

# Já o estado civíl parace não influenciar tanto na decisão de compra:
# table=pd.crosstab(data.marital,data.y)
# # table.sum(1) -> soma de todos os elementos da coluna 1 (y)
# # table.div() -> 'total_de_0/total_y' e 'total_de_1/total_y'
# table = table.div(table.sum(1).astype(float), axis=0)
# table.plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Marital Status vs Purchase')
# plt.xlabel('Marital Status')
# plt.ylabel('Proportion of Customers')
# plt.subplots_adjust(bottom=0.25) # Para ajustar a parte de baixo da imagem
# plt.show()
# plt.savefig('marital_vs_pur_stack')

# Usando o método anterior para observar por nível educacional
# table=pd.crosstab(data.education,data.y)
# # table.sum(1) -> soma de todos os elementos da coluna 1 (y)
# # table.div() -> 'total_de_0/total_y' e 'total_de_1/total_y'
# table = table.div(table.sum(1).astype(float), axis=0)
# table.plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Education vs Purchase')
# plt.xlabel('Education')
# plt.ylabel('Proportion of Customers')
# plt.subplots_adjust(bottom=0.35) # Para ajustar a parte de baixo da imagem
# plt.show()

# Agora com day_of_week
# pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
# plt.title('Stacked Bar Chart of Day Of Week vs Purchase')
# plt.xlabel('Day of Week')
# plt.ylabel('Proportion of Customers')
# plt.subplots_adjust(bottom=0.35) # Para ajustar a parte de baixo da imagem
# plt.show()

# Por mês
# pd.crosstab(data.month, data.y).plot(kind='bar')
# plt.title('Stacked Bar Chart of Month vs Purchase')
# plt.xlabel('Month')
# plt.ylabel('Proportion of Customers')
# plt.subplots_adjust(bottom=0.35) # Para ajustar a parte de baixo da imagem
# plt.show()

# Vendo o histograma (distribuição de frequências) de idades
# data.age.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()
#=====================================================

#=====================================================
#As linhas de código à seguir criam variáveis 'dummy' com valores de '0' e '1'.
# Por exemplo: a variável recém criada 'job_admin' terá valor 1, se verdadeiro para o 
# cliente, e '0' se falsa
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
# print(data_final.columns.values)
# for i in data_final.columns.values:
# 	print(i)
#=====================================================

#=====================================================
# SMOTE (Synthetic Minority Oversampling Technique):
# O SMOTE funciona à partir da criação de amostras da classe minoritária, ou seja, que está em 
# menor número no dataset (em nosso, caso 'subscription'). Este é um tipo de aumento de dados 
# para a classe minoritária que desequilibrada em relação a classe majoritária
X = data_final.loc[:, data_final.columns != 'y'] # X recebe todas as colunas, exceto a coluna 'y'
y = data_final.loc[:, data_final.columns == 'y'] # y recebe a coluna 'y'

# Para DEBUG
print("Number of no subscription before SMOTE", len(y[y['y']==0]))
print("Number of subscription before SMOTE",len(y[y['y']==1]))

from imblearn.over_sampling import SMOTE

# Aqui é criado o objeto 'os' que é do tipo SMOTE 
os = SMOTE(random_state=0)
# A função 'train_test_split()', retorna uma lista de treinamento/teste para x e y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
columns = X_train.columns

# Para DEBUG, apenas: Diferença entre 'train' e 'teste'
# for i in range(10):
# 	print(X_train.age.iloc[i], " ", X_test.age.iloc[i])
# print("========================================")
# for i in range(10):
# 	print(y_train.y.iloc[i], " ", y_test.y.iloc[i])

# Então o método 'fit_sample()' é chamado para gerar a amostragem sintética
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",\
	   len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",\
	   len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",\
	   len(os_data_y[os_data_y['y']==1])/len(os_data_X))
#=====================================================

#=====================================================
# RFE (Recursive Feature Elimination)
# O RFE é baseado na ideia de repetir a construção de um modelo inúmeras vezes
# e escolher entre o recurso que se sai melhor ou pior
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression() # Criado objeto da Regressão Logística
rfe = RFE(logreg, 20)	# Criado o objeto do RFE, propriamente dito. Importante notar que aqui é passado 
						# o objeto 'logreg' criado anteriormente, juntamente com a quantidade final de 
						# features que o algorítmo deve gerar (20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel()) # Por fim, o algorítmo RFE é iníciado com os valores de 
												   # treinamento os_data_X e "os_data_y.values.ravel()" o 
												   # método numpy ravel() é passado para fazer da array 
												   # uma lista (ou achatá-la)

# Rotina para pegar automaticamente os valores 'True' e linkar com os features da 'os_data_X'
count = 0
count_array = 0
rfe_feature_array = [0]*20 # Array para guardar os resultados 'True' da RFE
for i in rfe.support_:
	if i == True:
		rfe_feature_array[count_array] = os_data_X.columns[count]
		count_array+=1
	count+=1
# print(rfe_feature_array)
# print(rfe.support_)
# print(rfe.ranking_)
#=====================================================

#=====================================================
# Implementing the model
# Descomentar a linha abaixo e comentar a posterior quando o algorítmo RFE estiver não-comentado
# cols = rfe_feature_array
# cols=['marital_divorced', 'marital_married', 'marital_single', 'marital_unknown', 'education_Basic', 
# 	  'education_high.school', 'education_professional.course', 'education_university.degree', 
# 	  'education_unknown', 'housing_no', 'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 
# 	  'loan_yes', 'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed']

# Após remover as variáveis que não serão utilizadas
cols=['marital_divorced', 'marital_married', 'marital_single', 'education_Basic', 
	  'education_high.school', 'education_professional.course', 'education_university.degree', 
	  'education_unknown', 'loan_no',  
	  'loan_yes', 'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed']
X=os_data_X[cols]
y=os_data_y['y']
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#=====================================================

#=====================================================
# Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: \
	  {:.2f}'.format(logreg.score(X_test, y_test)))
#=====================================================

#=====================================================
# Aplicando a matriz de confusão
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
verdadeiro_positivo = confusion_matrix[0][0]
falso_positivo = confusion_matrix[0][1]
falso_negativo = confusion_matrix[1][0]
verdadeiro_negativo = confusion_matrix[1][1]
soma = 0
for i in range (2):
	for j in range(2):
		soma += confusion_matrix[i][j]
print(soma)
print("Acurácia = ", (verdadeiro_positivo+verdadeiro_negativo)/soma)
print("Sensibilidade = ", verdadeiro_positivo/(verdadeiro_positivo + falso_negativo))
print("Especificidade = ", verdadeiro_negativo/(verdadeiro_negativo + falso_positivo))
#=====================================================

#=====================================================
# Por fim, aplicando Acurácia, Sensibilidade e Especificidade
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#=====================================================












