''' TAREA 02.
INTRODUCCIÓN A LA CIENCIA DE DATOS.
María Alejandra BL, Luz María SM, Jesús Alonso '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns

#============================================
# EXPLORACIÓN Y PREPROCESAMIENTO DE DATOS 
#============================================

#Ajustamos la ruta al archivo de datos .csv:
#file_path = "/Users/aleborrego/Downloads/Tarea 1 Ciencia de Datos/data.csv"
file_path = "/Users/aleborrego/Documents/Ciencia de Datos/Tarea2Equipo4/Data.csv"

#Creamos un DataFrame con las variables por columnas
df = pd.read_csv(file_path, sep = ";")
#Vemos las características de las variables (en especial si hay vacíos)
df.info() 

#Vemos los distintos valores en cada variable
for col in df.columns:
    print(f"\nColumna: {col}")
    print(df[col].unique())
    
#Buscamos las columnas que tengan "unknown"
columnas_unknown = (df == "unknown").any()
columnas_unknown = columnas_unknown[columnas_unknown].index.tolist()

#Calculamos el porcentaje de valores desconocidos "unknown"
porcentajes = (df[columnas_unknown] == "unknown").sum() / len(df) * 100
print(porcentajes)

#Contamos cuántas respuestas hay de cada una (no, yes y unknown) en la variable "default"
print(df["default"].value_counts())

# Por ahora vamos a imputar los unknown de todas las columnas incluyendo default
# Pero probablemente la quitemos en un futuro

#Quitamos  los valores "unknown" en las variables que lo tengan
for col in columnas_unknown:
    #Guardamos los valores que no son "unknown"
    valores = df.loc[df[col] != "unknown", col]

    #Calculamos frecuencias relativas de cada valor y las guardamos como las probabilidades de aparecer
    probas = valores.value_counts(normalize=True)

    #Guardamos las posiciones en las que hay "unknown"
    si_unknown = df[col] == "unknown"
    #Calculamos cuántos "unknown" hay en la columna contando las posiciones en las que aparece
    num_unknown = si_unknown.sum()

    #Generamos reemplazos aleatorios de acuerdo a las probabilidades frecuencistas para reemplazar "unknown"'s
    reemplazos = np.random.choice(probas.index, size=num_unknown, p=probas.values)

    #Cambiamos los "unknow" por los valores obtenidos con imputación aleatoria proporcional
    df.loc[si_unknown, col] = reemplazos

#Revisamos que las columnas ya no tengan "unknown"'s
for col in columnas_unknown:
    print(f"\nColumna: {col}")
    print(df[col].unique())
    
#Codificamos las var. categoricas con one-hot para hacerlas var. dummys
columnas_dummies = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']
df_onehot = pd.get_dummies(df, columns=columnas_dummies, drop_first = True, dtype = int)

#Codificamos las variables de "si y no" como binarias "1 y 0"
columnas_si_no = ['default', 'housing', 'loan', 'y']
df_onehot[columnas_si_no] = df_onehot[columnas_si_no].replace({'yes': 1, 'no': 0})

#Codificamos la variable ordinal "education" para asiganrle orden a los valores del 0 al 6
orden = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4,
         'professional.course': 5, 'university.degree': 6}
df_onehot['education'] = df_onehot['education'].map(orden)



