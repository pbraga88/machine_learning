#!/usr/bin/python3

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus # Para criar o gráfico da Árvore de Decisão
from IPython.display import Image # Para mostrar o gráfico da 
                                  # Árvore de Decisão
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data/train.csv')

def data_info():
    df.info()
    print("Soma 'null': ",'\n', df.isnull().sum())
def data_analysis(feature = "Survived"):
    sns.countplot(df[feature], color = 'red')
    
def distribution_analysis(feature = 'Age'):
    sns.set(font_scale=1.5)
    f = plt.figure(figsize=(20,4))
    f.add_subplot(1,2,1)
    sns.distplot(df[feature])
    
    f.add_subplot(1,2,2)
    sns.boxplot(df[feature])

def treat_features_values():
    '''ESCREVER FUNÇÃO PARA DEFINIR VALORES INTEIROS NAS FEATURES
       COM VALORES STRING'''
    # Tratando a feature 'Sex'
    df_aux = df.join(pd.get_dummies(df[['Sex']]))
    df_aux = df_aux.drop(columns=['Sex'])
    
    ''' TRATAR FEATURE AGE, EMBARKED, CABIN(Remover)'''
    
    return df_aux
#     df = df_aux
#     df.info()
df = treat_features_values()

def var_correlation():
    return df.corr().style.background_gradient().set_precision(2)

def gaussian_nb(feature = 'Survived'):
    nb = GaussianNB()
    
    # Criando as variáveis x e y, apenas para teste
    # 1. MODIFICAR PARA UTILIZAR DOIS DATAFRAMES (TRAIN E TEST)
    # 2. TRATAR AS VARIÁVEIS EXCLUÍDAS (Name talvez não precise)
    x = df.drop(columns=[feature, 'Name','Sex','Ticket','Cabin',\
                        'Embarked', 'Age'])
    y = df[feature]
    
    # ALTERAR PARA UTILIZAR DOIS DATAFRAMES (TRAIN E TEST)
    X_train, X_test, y_train, y_test = train_test_split(x, y, \
                                    test_size = 0.2, random_state=4)
    
    # Treinando o modelo
    # AQUI DEVERÁ TER SÓ O DATAFRAME DE TREINAMENTO
    nb.fit(X_train, y_train)
    
    # Prevendo no data_set de teste
    # ALTERAR PARA PREVER NO DATAFRAME DE TESTE
    y_pred = nb.predict(X_test)
    
    # Performance do modelo
    print(accuracy_score(y_test, y_pred))
    
# def decision_tree(feature = 'Survived'):
print("teste")
feature = 'Survived'
x = df.drop(columns=[feature, 'Name','Ticket','Cabin',\
                    'Embarked', 'Age'])
y = df[feature]
clf = tree.DecisionTreeClassifier()

clf_train = clf.fit(x, y)

# Export/Print a decision tree in DOT format.
#     print(tree.export_graphviz(clf_train, None))

#Create Dot Data
#Gini decides which attribute/feature should be placed at the root 
# node, which features will act as internal nodes or leaf nodes
dot_data = tree.export_graphviz(clf_train, out_file=None,\
        feature_names=list(x.columns.values),\
        class_names=['Not', 'Yes'], rounded=True, filled=True) 
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)
print("teste2")
# Show graph
Image(graph.create_png())
print("teste3")
    
# decision_tree()