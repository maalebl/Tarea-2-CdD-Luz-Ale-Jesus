''' TAREA 02.
INTRODUCCIÓN A LA CIENCIA DE DATOS.
MarÃ­a Alejandra BL, Luz MarÃ­a SM, JesÃºs Alonso '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import LinearSegmentedColormap


np.random.seed(11)

#============================================
# EXPLORACIÓN Y PREPROCESAMIENTO DE DATOS 
#============================================

#Ajustamos la ruta al archivo de datos .csv:
#file_path = "/Users/aleborrego/Downloads/Tarea 1 Ciencia de Datos/data.csv"
file_path = "C:/Users/luzma/Downloads/Data_2.csv"

#Creamos un DataFrame con las variables por columnas
df = pd.read_csv(file_path, sep = ";")
#Vemos las caracterÃ­sticas de las variables (en especial si hay vací­os)
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

#Imputación de datos "unknown"

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
    
#Codificamos las var. categoricas con one-hot para hacerlas var. dummies
columnas_dummies = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']
df_onehot = pd.get_dummies(df, columns=columnas_dummies, drop_first = True, dtype = int)

#Codificamos las variables de "si y no" como binarias "1 y 0"
columnas_si_no = ['default', 'housing', 'loan', 'y']
df_onehot[columnas_si_no] = df_onehot[columnas_si_no].replace({'yes': 1, 'no': 0})

#Codificamos la variable ordinal "education" para asiganrle orden a los valores del 0 al 6
orden = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4,
         'professional.course': 5, 'university.degree': 6}
df_onehot['education'] = df_onehot['education'].map(orden)

#Quitamos la variable "duration" para tener el data frame listo para el anÃ¡lisis
df_listo = df_onehot.drop(columns='duration')

#====================
# MODELADO 
#====================

x_train, x_test, y_train, y_test = train_test_split(
    df_listo.drop(columns=['y']), df_listo['y'], test_size=0.3, stratify=df_listo['y'], 
    random_state=42)

### NAIVE BAYES ###

# Entramos Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)

# Calculamos estimadores de desempeño
y_pred_nb = nb.predict(x_test)
cm = confusion_matrix(y_test, y_pred_nb)
print("Matriz de confusión (Naive Bayes):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precisión:", precision_score(y_test, y_pred_nb, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_nb, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_nb, average="weighted"))


# ROC - AUC
y_proba_nb = nb.predict_proba(x_test)
if len(nb.classes_) == 2:
    # caso binario: usar la columna de la clase positiva
    idx_pos = list(nb.classes_).index(1) if 1 in nb.classes_ else 1
    auc_nb = roc_auc_score(y_test, y_proba_nb[:, idx_pos])
else:
    # multiclase: usar esquema one-vs-rest con promedio ponderado
    auc_nb = roc_auc_score(y_test, y_proba_nb, multi_class="ovr", average="weighted")

print("ROC–AUC:", auc_nb)


### LDA (LINEAR DISCRIMINANT ANALYSIS) ###

# Entrenamos LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Calculamos estimadores de desempeño
y_pred_lda = lda.predict(x_test)
cm = confusion_matrix(y_test, y_pred_lda)
print("Matriz de confusión (LDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_lda))
print("Precisión:", precision_score(y_test, y_pred_lda, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_lda, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_lda, average='weighted'))

# ROC_AUC
y_proba_lda = lda.predict_proba(x_test)

if len(lda.classes_) == 2:
    # caso binario: toma la prob. de la clase positiva 
    idx_pos = list(lda.classes_).index(1) if 1 in lda.classes_ else 1
    auc_lda = roc_auc_score(y_test, y_proba_lda[:, idx_pos])
else:
    # multiclase: promedio one-vs-rest ponderado
    auc_lda = roc_auc_score(y_test, y_proba_lda, multi_class="ovr", average="weighted")

print("ROC–AUC:", auc_lda)

### QDA (QUADRATIC DISCRIMINANT ANALYSIS) ###

# Entrenamos QDA
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(x_train, y_train)

# Calculamos estimadores de desempeño
y_pred_qda = qda.predict(x_test)
cm = confusion_matrix(y_test, y_pred_qda)
print("Matriz de confusión (QDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_qda))
print("Precisión:", precision_score(y_test, y_pred_qda, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_qda, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_qda, average='weighted'))

#ROC-AUC
y_proba_qda = qda.predict_proba(x_test)

if len(qda.classes_) == 2:
    # caso binario: toma la prob. de la clase positiva 
    idx_pos = list(qda.classes_).index(1) if 1 in qda.classes_ else 1
    auc_qda = roc_auc_score(y_test, y_proba_qda[:, idx_pos])
else:
    # multiclase: promedio one-vs-rest ponderado
    auc_qda = roc_auc_score(y_test, y_proba_qda, multi_class="ovr", average="weighted")

print("ROC–AUC:", auc_qda)

### k-NN (k-NEAREST NEIGHBORS) ###
# Entrenar (con k=5 vecinos)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Calculamos estimadores de desempeño
y_pred_knn = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred_knn)
print("Matriz de confusión (k-NN):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precisión:", precision_score(y_test, y_pred_knn, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_knn, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_knn, average='weighted'))

#AUC-ROC
y_proba_knn = knn.predict_proba(x_test) 

if len(knn.classes_) == 2:
    # caso binario: usa la prob. de la clase positiva
    idx_pos = list(knn.classes_).index(1) if 1 in knn.classes_ else 1
    auc_knn = roc_auc_score(y_test, y_proba_knn[:, idx_pos])
else:
    # multiclase: promedio one-vs-rest ponderado
    auc_knn = roc_auc_score(y_test, y_proba_knn, multi_class="ovr", average="weighted")

print("ROC–AUC:", auc_knn)

# Guardamos los resultados en un diccionario
results = {
    "Naive Bayes": {
        "Acc": accuracy_score(y_test, y_pred_nb),
        "Precisión": precision_score(y_test, y_pred_nb, average="weighted"),
        "Recall": recall_score(y_test, y_pred_nb, average="weighted"),
        "F1": f1_score(y_test, y_pred_nb, average="weighted")
    },
    "LDA": {
        "Acc": accuracy_score(y_test, y_pred_lda),
        "Precisión": precision_score(y_test, y_pred_lda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_lda, average="weighted"),
        "F1": f1_score(y_test, y_pred_lda, average="weighted")
    },
    "QDA": {
        "Acc": accuracy_score(y_test, y_pred_qda),
        "Precisión": precision_score(y_test, y_pred_qda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_qda, average="weighted"),
        "F1": f1_score(y_test, y_pred_qda, average="weighted")
    },
    "k-NN (k=5)": {
        "Acc": accuracy_score(y_test, y_pred_knn),
        "Precisión": precision_score(y_test, y_pred_knn, average="weighted"),
        "Recall": recall_score(y_test, y_pred_knn, average="weighted"),
        "F1": f1_score(y_test, y_pred_knn, average="weighted")
    }
}

### FISHER ###

def ajustar_fisher(X, y, alpha=1e-3):
    """
    Ajusta el clasificador por Fisher (binario, y ∈ {0,1}).
    alpha: regularización para la matriz de dispersión dentro de clases.
    Devuelve: vector de proyección 'a', umbral, medias m0, m1.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    X0 = X[y == 0]
    X1 = X[y == 1]

    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)

    # Dispersión dentro de clases (pooled)
    S0 = np.cov(X0, rowvar=False)
    S1 = np.cov(X1, rowvar=False)
    Sw = S0 + S1 + alpha * np.eye(X.shape[1])

    # Dirección óptima
    a = np.linalg.solve(Sw, (m1 - m0))

    # Umbral = punto medio de las medias proyectadas (priors iguales)
    umbral = 0.5 * ((m0 @ a) + (m1 @ a))
    return a, umbral, m0, m1

def puntajes_fisher(X, a):
    X = np.asarray(X, dtype=float)
    return X @ a

def predecir_fisher(X, a, umbral):
    return (puntajes_fisher(X, a) > umbral).astype(int)

def umbral_fisher_con_priores(a, m0, m1, X_train, y_train, priors='empirical', c_fp=1.0, c_fn=1.0):
    """
    Calcula el umbral t para Fisher en la proyección z = a^T x,
    incorporando priores y (opcionalmente) costos asimétricos.
    """
    X = np.asarray(X_train, float); y = np.asarray(y_train, int)
    X0, X1 = X[y==0], X[y==1]
    n0, n1 = X0.shape[0], X1.shape[0]

    # covarianza pooled para sigma^2 proyectada
    S0 = np.cov(X0, rowvar=False); S1 = np.cov(X1, rowvar=False)
    Sigma_hat = ((n0-1)*S0 + (n1-1)*S1) / (n0 + n1 - 2)
    sigma2 = float(a @ Sigma_hat @ a)

    # priores
    if priors == 'empirical':
        pi0, pi1 = n0/(n0+n1), n1/(n0+n1)
    elif priors == 'equal':
        pi0, pi1 = 0.5, 0.5
    else:
        pi0, pi1 = priors  # tupla (pi0, pi1)

    z0 = float(a @ m0); z1 = float(a @ m1)
    return 0.5*(z0 + z1) + (sigma2/(z1 - z0)) * np.log((pi0*c_fp)/(pi1*c_fn))

# ====== Entrenamiento y evaluación en TEST (Fisher desde cero) ======
# Ajuste Fisher (dirección 'a' y medias)
a, umbral_mid, m0, m1 = ajustar_fisher(x_train, y_train, alpha=1e-3)

# Nuevo umbral con priores empíricas (recomendado en desbalance)
umbral_emp = umbral_fisher_con_priores(a, m0, m1, x_train, y_train, priors='empirical')

# Predicciones
y_pred_fisher_emp = predecir_fisher(x_test, a, umbral_emp)  # priores empíricas

y_pred_fisher = y_pred_fisher_emp
scores_test   = puntajes_fisher(x_test, a)  # para ROC-AUC 

# Métricas 
cm = confusion_matrix(y_test, y_pred_fisher)
print("Matriz de confusión (Fisher con priores empíricas):\n", cm)
print("Accuracy:",     accuracy_score(y_test, y_pred_fisher))
print("Precisión:",    precision_score(y_test, y_pred_fisher, average="weighted", zero_division=0))
print("Sensibilidad:", recall_score(y_test, y_pred_fisher, average="weighted", zero_division=0))
print("F1-score:",     f1_score(y_test, y_pred_fisher, average="weighted", zero_division=0))
print("ROC-AUC:",      roc_auc_score(y_test, scores_test))

# Mapa de calor 
try:
    pink_cmap
except NameError:
    pink_cmap = LinearSegmentedColormap.from_list("custom_pink", ["#ffffff", "#FFA8C2", "#FF4F95"])

cm = confusion_matrix(y_test, y_pred_fisher, labels=[0, 1])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap=pink_cmap, cbar=False,
            xticklabels=["0 ", "1 "], yticklabels=["0 ", "1"])
plt.title("Fisher")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

######
######
## VALIDACI�N CRUZADA CON FISHER ###

#aqui









# VALIDACIÓN CRUZADA COMPARATIVA

#models = {
 # "Naive Bayes": GaussianNB(),
  #  "LDA": LinearDiscriminantAnalysis(),
   # "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    #"k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
#}

#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#print("\n=== Validación Cruzada (5-fold, accuracy) ===")
#for name, model in models.items():
 #   scores = cross_val_score(model, df_onehot.drop(columns='y'), df_onehot['y'], cv=cv, scoring="accuracy")
  #  print(f"{name:15s}: Acc = {scores.mean():.3f} Â± {scores.std():.3f}")

# MATRICES DE CONFUSIÓN



# Lista de modelos ya entrenados
trained_models = {
    "Naive Bayes": nb,
    "LDA": lda,
    "QDA": qda,
    "k-NN": knn
}



# Graficar matrices de confusión

pink_cmap = LinearSegmentedColormap.from_list("custom_pink", ["#ffffff", "#FFA8C2", "#FF4F95"])

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=pink_cmap, cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()
    

for name, model in trained_models.items():
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))  # cada matriz tendrá su propia figura
    sns.heatmap(cm, annot=True, fmt="d", cmap=pink_cmap, cbar=False)
    plt.title(name)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

