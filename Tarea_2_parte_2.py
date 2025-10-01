import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from scipy.stats import norm, multivariate_normal
import seaborn as sns

# Función para generar el vector "y" de datos binarios
def vector_y(n,pi_0) :
    Y = np.random.choice([0, 1], size=n, p=[pi_0, 1-pi_0])
    return(Y)

# Función para generar a X codicionado a y
def vector_x(n, p, y, mu0, mu1, Sigma0, Sigma1):
    X = np.zeros((n, p))
    for i in range(n):
        if y[i] == 0:
            X[i, :] = np.random.multivariate_normal(mu0, Sigma0)
        else:
            X[i, :] = np.random.multivariate_normal(mu1, Sigma1)
    return(X)

def plot_de_decision_boundary(X, y, clf, title=''):
    h = .02  
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


def configurar_escenario(dificultad='facil', covarianzas_iguales=True):
    p = 2
    if dificultad == 'facil':
        mu0 = np.zeros(p)
        mu1 = np.ones(p) * 3
    elif dificultad == 'medio':
        mu0 = np.zeros(p)
        mu1 = np.ones(p) * 1.5
    else:  # difícil
        mu0 = np.zeros(p)
        mu1 = np.ones(p) * 0.5

    sigma0 = np.array([[1, 0.5], [0.5, 1]])
    sigma1 = sigma0 if covarianzas_iguales else np.array([[1, -0.5], [-0.5, 1.5]])

    return mu0, mu1, sigma0, sigma1


class BayesOptimo:
    def __init__(self, mu0, mu1, sigma0, sigma1, pi0=0.5):
        self.pi0 = pi0
        self.pi1 = 1 - pi0
        self.rv0 = multivariate_normal(mean=mu0, cov=sigma0)
        self.rv1 = multivariate_normal(mean=mu1, cov=sigma1)

    def predict(self, X):
        post0 = self.pi0 * self.rv0.pdf(X)
        post1 = self.pi1 * self.rv1.pdf(X)
        return (post1 > post0).astype(int)
    

class FisherClassifier:
    def __init__(self, mu0, mu1, sigma0, sigma1):
        Sw = sigma0 + sigma1
        self.w = np.linalg.inv(Sw).dot(mu1 - mu0)
        m0 = mu0.dot(self.w)
        m1 = mu1.dot(self.w)
        self.threshold = 0.5 * (m0 + m1)

    def predict(self, X):
        z = X.dot(self.w)   
        return (z > self.threshold).astype(int)

# =====================
# Experimento
# =====================



#-----------Parametros------------
dificultad = 'medio'
cov_iguales = True
n = 200
pi_0 = 0.5
k = 5
p=2
#----------Almacenamiento de riesgo--------
Muestras=5
LDA_datos_k_nn=np.zeros(Muestras)
QDA_datos_k_nn=np.zeros(Muestras)
Bayes_datos=np.zeros(Muestras)
#---------------------------------------------



mu0, mu1, sigma0, sigma1 = configurar_escenario(dificultad, cov_iguales)
y = vector_y(n, pi_0)
X = vector_x(n, p, y, mu0, mu1, sigma0, sigma1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Clasificador k-NN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
riesgo_knn = 1-accuracy_score(y_test, y_pred_knn)
print(f"k-NN (k={k}): Riesgo = {riesgo_knn:.3f}")

plot_de_decision_boundary(X_test, y_test, knn, title=f"k-NN (k={k}) - Riesgo={riesgo_knn:.2f}")

# --- Bayes Óptimo ---
bayes = BayesOptimo(mu0, mu1, sigma0, sigma1, pi_0)
y_pred_bayes = bayes.predict(X_test)
riesgo_bayes = 1-accuracy_score(y_test, y_pred_bayes)

print(f"Bayes Óptimo (densidades verdaderas): Riesgo = {riesgo_bayes:.3f}")
plot_de_decision_boundary(X_test, y_test, bayes,
                          title=f"Bayes Óptimo - Riesgo={riesgo_bayes:.2f}")

# --- Fisher ---
fisher = FisherClassifier(mu0, mu1, sigma0, sigma1)
y_pred_fisher = fisher.predict(X_test)
riesgo_fisher = 1-accuracy_score(y_test, y_pred_fisher)

print(f"Fisher: Riesgo = {riesgo_fisher:.3f}")
plot_de_decision_boundary(X_test, y_test, fisher,
                          title=f"Fisher - Riesgo={riesgo_fisher:.2f}")






################## AQUÍ LA PARTE NUEVA CON LOS DEMAS RESULTADOS ######################








# Función para guardar los valores de riesgo del clasificador k-NN
def Riesgos_knn(n, p, pi_0, mu0, mu1, Sigma0, Sigma1):
    resultados = []
    k_val = [1,3,5,11,21]
    for k in k_val:
        riesgos = []
        for i in range(20):  # Realizamos 20 réplicas
            y = vector_y(n, pi_0)
            X = vector_x(n, p, y, mu0, mu1, Sigma0, Sigma1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y
            )
            modelo = KNeighborsClassifier(n_neighbors=k)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            riesgos.append(1 - accuracy_score(y_test, y_pred))
        
        resultados.append({
            "Modelo": f"k-NN (k={k})",
            "n": n,
            "mean": np.mean(riesgos),
            "std": np.std(riesgos),
            "k": k
        })
    
    return pd.DataFrame(resultados)

# Función para guardar los riesgos de los clasificadores NB, LDA, QDA
def Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1):
    modelos = {
        "Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        riesgos = []
        for i in range(20): # Realizamos 20 replicas 
            y = vector_y(n, pi_0)
            X = vector_x(n, p, y, mu0, mu1, Sigma0, Sigma1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y
            )
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            riesgos.append(1 - accuracy_score(y_test, y_pred))
        
        resultados[nombre] = {
            "mean": np.mean(riesgos),
            "std": np.std(riesgos),
            "n": n   
        }
    
    return pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "Modelo"})

# Función para calcular el riesgo de Bayes con matrices var-cov iguales
def bayes_error_equal_cov(mu0, mu1, Sigma, pi0):
    delta = mu1 - mu0
    Sigma_inv = np.linalg.inv(Sigma)
    w = Sigma_inv @ delta
    wSigmaw = float(delta.T @ Sigma_inv @ delta)
    
    c = -0.5 * (mu1.T @ Sigma_inv @ mu1 - mu0.T @ Sigma_inv @ mu0) + np.log((1-pi0)/pi0)
    # Numeradores para las normales
    z0 = ( -c - w.T @ mu0 ) / np.sqrt(wSigmaw)
    z1 = ( -c - w.T @ mu1 ) / np.sqrt(wSigmaw)
    p0_err = 1 - norm.cdf(z0)   # P(decide 1 | Y=0)
    p1_err = norm.cdf(z1)       # P(decide 0 | Y=1)
    Rstar = pi0 * p0_err + (1-pi0) * p1_err
    return Rstar

# Función para calcular el riesgo de Bayes con matrices var-cov distintas
def bayes_error_montecarlo(mu0, mu1, Sigma0, Sigma1, pi0, n_sim=200000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n0 = int(n_sim * pi0)
    n1 = n_sim - n0

    X0 = rng.multivariate_normal(mu0, Sigma0, size=n0)
    X1 = rng.multivariate_normal(mu1, Sigma1, size=n1)

    # Densidades
    f0_X0 = multivariate_normal.pdf(X0, mean=mu0, cov=Sigma0)
    f1_X0 = multivariate_normal.pdf(X0, mean=mu1, cov=Sigma1)
    f0_X1 = multivariate_normal.pdf(X1, mean=mu0, cov=Sigma0)
    f1_X1 = multivariate_normal.pdf(X1, mean=mu1, cov=Sigma1)

    # Regla de Bayes: decidir 1 si (1-pi0)f1 > pi0 f0
    err0 = np.mean((1-pi0) * f1_X0 > pi0 * f0_X0)
    err1 = np.mean(pi0 * f0_X1 >= (1-pi0) * f1_X1) 
    Rstar_est = pi0 * err0 + (1-pi0) * err1
    return Rstar_est

# Función para obtener todo lo que necesitamos de los clasificadores NB, LDA, QDA
def Resultados(p, pi_0, mu0, mu1, Sigma0, Sigma1) :
    lista_resultados = []
    
    for n in [50, 100, 200, 500]:
        df_temp = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma0)
        df_k = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
        lista_resultados.append(df_temp)
        lista_resultados.append(df_k)
        

    df_final = pd.concat(lista_resultados, ignore_index=True)

    df_final = df_final.sort_values(by=["Modelo", "n"]).reset_index(drop=True)

    error_bayes = bayes_error_equal_cov(mu0, mu1, Sigma0, pi_0)
    
    y = vector_y(n, pi_0)
    X = vector_x(n, 2, y, mu0, mu1, Sigma0, Sigma1)
    
    # Graficar con colores distintos según y
    plt.figure(figsize=(6,6))
    plt.scatter(X[y==0,0], X[y==0,1], c="pink", label="Clase 0", alpha=0.7)
    plt.scatter(X[y==1,0], X[y==1,1], c="hotpink", label="Clase 1", alpha=0.7)
    
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Datos simulados según su clase verdadera")
    plt.legend()
    plt.show()
    
       
        
    # Gráfica para riesgos NB, LDA, QDA

    plt.figure(figsize=(8,6))
    ax = sns.lineplot(
        data=df_final, x="n", y="mean", hue="Modelo", marker="o", err_style="bars"
    )
    for modelo, line in zip(df_final["Modelo"].unique(), ax.lines):
        subset = df_final[df_final["Modelo"] == modelo]
        color = line.get_color()  
        
        plt.errorbar(
            subset["n"], subset["mean"], yerr=subset["std"],
            fmt="none", capsize=5, color=color, alpha=0.8
        )
  
    plt.axhline(
        y=error_bayes, color="red", linestyle="--", linewidth=2,
        label="Error de Bayes"
    ) 
    plt.xlabel("Tamaño de muestra (n)")
    plt.ylabel("Riesgo promedio")
    plt.title("Riesgos vs tamaño de muestra")
    plt.legend(title="Modelo")
    plt.show()
    
    # Gráfica de brechas

    df_plot = df_final.copy()
    df_plot['gap'] = df_plot['mean'] - error_bayes
    
    plt.figure(figsize=(8,6))
    ax = sns.lineplot(
        data=df_plot, x="n", y="gap", hue="Modelo", marker="o", legend='full', errorbar = None
    )
    
    legend = ax.legend_
    labels = [t.get_text() for t in legend.texts]
    
    lines = ax.lines[:len(labels)]
    colors = [line.get_color() for line in lines]
    
    plt.axhline(0.0, color="red", linestyle="--", linewidth=2, label="Error de Bayes (R*)")
    
    plt.xlabel("Tamaño de muestra (n)")
    plt.ylabel("Brecha respecto a Error de Bayes (mean - R*)")
    plt.title("Brecha del riesgo promedio respecto al error de Bayes")
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.show()
    
    return df_final, error_bayes

def Resultados_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1) :
    lista_resultados = []
    
    for n in [50, 100, 200, 500]:
        df_k = Riesgos_knn(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
        lista_resultados.append(df_k)

    df_final = pd.concat(lista_resultados, ignore_index=True)

    df_final = df_final.sort_values(by=["k", "n"]).reset_index(drop=True)

    error_bayes = bayes_error_equal_cov(mu0, mu1, Sigma0, pi_0)
            
    # Gráfica para riesgos k-NN 
    plt.figure(figsize=(8,6))

    ax = sns.lineplot(
        data=df_final, x="k", y="mean", hue="n", marker="o"
    )

    for n_val, line in zip(df_final["n"].unique(), ax.lines):
        subset = df_final[df_final["n"] == n_val]
        color = line.get_color()
        plt.errorbar(
            subset["k"], subset["mean"], yerr=subset["std"],
            fmt="none", capsize=5, color=color, alpha=0.8
        )

    plt.axhline(
        y=error_bayes, color="red", linestyle="--", linewidth=2,
        label="Error de Bayes"
    )

    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("Riesgo promedio")
    plt.title("Riesgo de k-NN vs número de vecinos")
    plt.legend(title="Tamaño de muestra (n)")
    plt.show()

    # Grafica para brechas 
    
    lista_resultados = []
    
    for n in [50, 100, 200, 500]:
        df_k = Riesgos_knn(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
        lista_resultados.append(df_k)
    
    df_final = pd.concat(lista_resultados, ignore_index=True)
    
    df_final = df_final.sort_values(by=["k", "n"]).reset_index(drop=True)
    
    error_bayes = bayes_error_equal_cov(mu0, mu1, Sigma0, pi_0)
    
    # Gráfica para riesgos k-NN 
    plt.figure(figsize=(8,6))
    
    ax = sns.lineplot(
        data=df_final, x="k", y="mean", hue="n", marker="o"
    )
    
    for n_val, line in zip(df_final["n"].unique(), ax.lines):
        subset = df_final[df_final["n"] == n_val]
        color = line.get_color()
        plt.errorbar(
            subset["k"], subset["mean"], yerr=subset["std"],
            fmt="none", capsize=5, color=color, alpha=0.8
        )
    
    plt.axhline(
        y=error_bayes, color="red", linestyle="--", linewidth=2,
        label="Error de Bayes"
    )
    
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("Riesgo promedio")
    plt.title("Riesgo de k-NN vs número de vecinos")
    plt.legend(title="Tamaño de muestra (n)")
    plt.show()
    
    # Grafica para brechas 
    
    df_plot = df_final.copy()
    df_plot['gap'] = df_plot['mean'] - error_bayes
    
    plt.figure(figsize=(8,6))
    ax = sns.lineplot(
        data=df_plot, x="k", y="gap", hue="n", marker="o", legend='full', errorbar = None
    )

    legend = ax.legend_
    labels = [t.get_text() for t in legend.texts]      
   
    lines = ax.lines[:len(labels)]
    colors = [line.get_color() for line in lines]
    
    plt.axhline(0.0, color="red", linestyle="--", linewidth=2, label="Error de Bayes (R*)")
    
    plt.xlabel("Tamaño de muestra (n)")
    plt.ylabel("Brecha respecto a Error de Bayes (mean - R*)")
    plt.title("Brecha del riesgo promedio respecto al error de Bayes")
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.show()
    
    
    df_final = df_final.drop(columns="k")
    
    return df_final, error_bayes

    lines = ax.lines[:len(labels)]
    colors = [line.get_color() for line in lines]
    
    plt.axhline(0.0, color="red", linestyle="--", linewidth=2, label="Error de Bayes (R*)")
    
    plt.xlabel("Tamaño de muestra (n)")
    plt.ylabel("Brecha respecto a Error de Bayes (mean - R*)")
    plt.title("Brecha del riesgo promedio respecto al error de Bayes")
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.show()
    

    df_final = df_final.drop(columns="k")

    return df_final, error_bayes


# ===============================================================================
# Simulaciones para calcular el riesgo de cada modelo bajo distintos escenarios
# ===============================================================================

# =========================== ESCENARIOS ===============================

p = 2 # valor p fijo

# Caso balanceado 
pi_0 = 0.5

# Covarianzas iguales

# Medias y covarianzas de X|Y
mu0 = np.array([1.5, 1.5])
mu1 = np.array([2, 2])

Sigma0 = np.eye(p)              
Sigma1 = Sigma0

#Comparación de los riesgos de cada clasificador

# Casos con distintos grados de separación
casos = [
    (np.array([2, 2]), np.array([5, 5])), # Fácil
    (np.array([2, 2]), np.array([3.5, 3.5])), # Medio
    (np.array([2, 2]), np.array([1.5, 1.5])) #Difícil
]

# Iterar y aplicar
for i, (mu0, mu1) in enumerate(casos, start=1):
    
    Resultados(p, pi_0, mu0, mu1, Sigma0, Sigma1)
    Resultados_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)
    

# Covarianzas distintas

# Medias y covarianzas de X|Y
mu0 = np.array([0, 0])
mu1 = np.array([2, 2])

Sigma0 = np.eye(p)              
Sigma1 = np.array([[1, 0.5], 
                   [0.5, 1]])  

Resultados(p, pi_0, mu0, mu1, Sigma0, Sigma1)
Resultados_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)

# * - * - * - * - * - * - * - * - * - * - * - * - * 

# Caso desbalanceado 
pi_0 = 0.8

# Covarianzas iguales

# Medias y covarianzas de X|Y
mu0 = np.array([1.5, 1.5])
mu1 = np.array([2, 2])

Sigma0 = np.eye(p)              
Sigma1 = Sigma0

#Comparación de los riesgos de cada clasificador
Resultados(p, pi_0, mu0, mu1, Sigma0, Sigma1)
Resultados_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)


# Covarianzas distintas


# Medias y covarianzas de X|Y
mu0 = np.array([0, 0])
mu1 = np.array([2, 2])

Sigma0 = np.eye(p)             
Sigma1 = np.array([[1, 0.5], 
                   [0.5, 1]])  

Resultados(p, pi_0, mu0, mu1, Sigma0, Sigma1)
Resultados_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)



