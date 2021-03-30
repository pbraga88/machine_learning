#!/usr/bin/python3

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../Exerc_Aprofundamento/Data/train.csv')

print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
