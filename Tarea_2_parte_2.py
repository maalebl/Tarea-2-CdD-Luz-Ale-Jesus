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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score


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

#Para el clasificador de Fisher
class Fisher1D(BaseEstimator, ClassifierMixin):
    """
    Criterio de Fisher: proyección 1D + umbral escogido para minimizar
    el error en entrenamiento.
    """
    def __init__(self):
        self.w_ = None
        self.t_ = None
        self.flip_ = 1  # +1 si clase 1 es z grande; -1 si al revés

    def fit(self, X, y):
        X0 = X[y==0]; X1 = X[y==1]
        mu0 = X0.mean(axis=0); mu1 = X1.mean(axis=0)

        # Covarianzas muestrales 
        S0 = np.cov(X0, rowvar=False, bias=False)
        S1 = np.cov(X1, rowvar=False, bias=False)
        Sw = S0 + S1

        # Regularización leve si Sw es singular (por si acaso)
        eps = 1e-8
        Sw = Sw + eps*np.eye(Sw.shape[0])

        # Dirección de Fisher
        self.w_ = np.linalg.solve(Sw, (mu1 - mu0))

        # Proyección y orientación
        z = X @ self.w_
        z0 = z[y==0]; z1 = z[y==1]
        m0, m1 = z0.mean(), z1.mean()
        self.flip_ = 1 if (m1 >= m0) else -1
        z_flip = self.flip_ * z
        y_flip = y  # no cambiamos etiquetas; sólo volteamos el eje

        # Umbral que minimiza error en entrenamiento
        ord_idx = np.argsort(z_flip)
        z_sorted = z_flip[ord_idx]
        y_sorted = y_flip[ord_idx]

        # Candidatos: puntos medios entre observaciones adyacentes de clases distintas
        cand = []
        for i in range(len(z_sorted)-1):
            if y_sorted[i] != y_sorted[i+1]:
                cand.append(0.5*(z_sorted[i] + z_sorted[i+1]))

        if not cand:
            # Si por alguna razón no hay mezcla, usa punto medio de medias
            self.t_ = 0.5*(self.flip_*m0 + self.flip_*m1)
        else:
            cand = np.array(cand)
            # Evalúa error 0-1 en train para cada candidato
            errs = []
            for t in cand:
                y_hat = (z_flip >= t).astype(int)
                errs.append(np.mean(y_hat != y))
            self.t_ = cand[int(np.argmin(errs))]

        return self

    def predict(self, X):
        z = self.flip_ * (X @ self.w_)
        return (z >= self.t_).astype(int)


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
        "Fisher (1D + umbral)": Fisher1D(),
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

# Función para obtener todo lo que necesitamos de los clasificadores NB, LDA, QDA (Caso Cov iguales)
def ResultadosCI(p, pi_0, mu0, mu1, Sigma0, Sigma1) :
    lista_resultados = []
    
    for n in [50, 100, 200, 500]:
        df_k = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
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
    
    print(df_final, error_bayes)
    return df_final, error_bayes

# Función para obtener todo lo que necesitamos de los clasificadores NB, LDA, QDA (Caso Cov distintas)
def ResultadosCD(p, pi_0, mu0, mu1, Sigma0, Sigma1) :
    lista_resultados = []
    
    for n in [50, 100, 200, 500]:
        df_k = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
        lista_resultados.append(df_k)
        

    df_final = pd.concat(lista_resultados, ignore_index=True)

    df_final = df_final.sort_values(by=["Modelo", "n"]).reset_index(drop=True)

    error_bayes = bayes_error_montecarlo(mu0, mu1, Sigma0, Sigma1, pi_0)
    
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
    
    print(df_final, error_bayes)
    return df_final, error_bayes

# Función para obtener todo lo que necesitamos del clasificador k-NN (Caso Cov iguales)
def ResultadosCI_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1) :
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
    
    print(df_final, error_bayes)
    return df_final, error_bayes

# Función para obtener todo lo que necesitamos del clasificador k-NN (Caso Cov disntintas)
def ResultadosCD_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1) :
    lista_resultados = []
    
    for n in [50, 100, 200, 500]:
        df_k = Riesgos_knn(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
        lista_resultados.append(df_k)

    df_final = pd.concat(lista_resultados, ignore_index=True)

    df_final = df_final.sort_values(by=["k", "n"]).reset_index(drop=True)

    error_bayes = bayes_error_montecarlo(mu0, mu1, Sigma0, Sigma1, pi_0)
            
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

    print(df_final, error_bayes)
    return df_final, error_bayes


# ===============================================================================
# Simulaciones para calcular el riesgo de cada modelo bajo distintos escenarios
# ===============================================================================

# =========================== ESCENARIOS ===============================

np.random.seed(546)

p = 2 # valor p fijo

# Caso balanceado 
pi_0 = 0.5

# Covarianzas iguales

# ovarianzas de X|Y

Sigma0 = np.eye(p)              
Sigma1 = Sigma0

# Casos con distintos grados de separación (medias)
casos = [
    (np.array([2, 2]), np.array([5, 5])), # Fácil
    (np.array([2, 2]), np.array([3.5, 3.5])), # Medio
    (np.array([2, 2]), np.array([1.5, 1.5])) #Difícil
]

# Comparación de riesgos
for i, (mu0, mu1) in enumerate(casos, start=1):
    
    ResultadosCI(p, pi_0, mu0, mu1, Sigma0, Sigma1)
    ResultadosCI_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)
    

# Covarianzas distintas
np.random.seed(546)

# Medias y covarianzas de X|Y
mu0 = np.array([1.5, 1.5])
mu1 = np.array([3, 3])

Sigma0 = np.eye(p)              
Sigma1 = np.array([[1, 0.5], 
                   [0.5, 1]])  

ResultadosCD(p, pi_0, mu0, mu1, Sigma0, Sigma1)
ResultadosCD_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)

# * - * - * - * - * - * - * - * - * - * - * - * - * 

# Caso desbalanceado 

pi_0 = 0.8
mu0 = np.array([1.5, 1.5])
mu1 = np.array([3, 3])

# Covarianzas iguales

# Medias y covarianzas de X|Y

Sigma0 = np.eye(p)              
Sigma1 = Sigma0

#Comparación de los riesgos de cada clasificador
ResultadosCI(p, pi_0, mu0, mu1, Sigma0, Sigma1)
ResultadosCI_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)

# Covarianzas distintas

Sigma0 = np.eye(p)             
Sigma1 = np.array([[1, 0.5], 
                   [0.5, 1]])  

ResultadosCD(p, pi_0, mu0, mu1, Sigma0, Sigma1)
ResultadosCD_kNN(p, pi_0, mu0, mu1, Sigma0, Sigma1)


#######RIESGOS ESTIMADOS ######
def riesgos_cv_modelos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1,
                       k_folds=5, n_repeats=1, R=20, random_state=123):
    """
    Estima L_cv(g) con K-fold estratificado (con repetición opcional).
    Genera R datasets nuevos; en cada uno hace CV y promedia riesgos.
    """
    modelos = {
        "Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Fisher (1D + umbral)": Fisher1D(),   # ¡asegúrate que esté definido a nivel módulo!
    }
    rng = np.random.default_rng(random_state)
    riesgos_por_modelo = {m: [] for m in modelos}

    for r in range(R):
        y = vector_y(n, pi_0)
        X = vector_x(n, p, y, mu0, mu1, Sigma0, Sigma1)

        cv = RepeatedStratifiedKFold(
            n_splits=k_folds, n_repeats=n_repeats,
            random_state=int(rng.integers(1e9))
        )
        for nombre, modelo in modelos.items():
            # ¡No paralelizar con estimadores propios! y captura errores como NaN
            acc = cross_val_score(
                modelo, X, y, scoring="accuracy", cv=cv,
                n_jobs=None,            # <--- CAMBIO CLAVE
                error_score=np.nan      # <--- CAMBIO CLAVE
            )
            riesgos_por_modelo[nombre].append(1.0 - np.nanmean(acc))  # <--- nan-robusto

    filas = []
    for nombre, vals in riesgos_por_modelo.items():
        m = float(np.nanmean(vals)) if len(vals) else np.nan
        s = float(np.nanstd(vals))  if len(vals) else np.nan
        filas.append({"Modelo": nombre, "n": n, "mean": m, "std": s})
    return pd.DataFrame(filas)




def Resultados_cv(p, pi_0, mu0, mu1, Sigma0, Sigma1,
                  n_grid=[50,100,200,500], k_folds=5, n_repeats=1, R=20):
    frames = []
    for n in n_grid:
        frames.append(riesgos_cv_modelos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1,
                                         k_folds=k_folds, n_repeats=n_repeats, R=R))
    df_cv = pd.concat(frames, ignore_index=True)
    return df_cv

def Resultados_MC_both(p, pi_0, mu0, mu1, Sigma0, Sigma1, n_grid=[50,100,200,500]):
    frames = []
    for n in n_grid:
        df_eq  = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma0)
        df_eq["Escenario"] = "Σ0=Σ1"
        df_neq = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
        df_neq["Escenario"] = "Σ0≠Σ1"
        frames += [df_eq, df_neq]
    df_mc = pd.concat(frames, ignore_index=True)
    return df_mc.sort_values(["Escenario","Modelo","n"]).reset_index(drop=True)

def Resultados_CV_both(p, pi_0, mu0, mu1, Sigma0, Sigma1,
                       n_grid=[50,100,200,500], k_folds=5, n_repeats=1, R=20):
    frames = []
    for n in n_grid:
        dfeq  = riesgos_cv_modelos(n, p, pi_0, mu0, mu1, Sigma0, Sigma0,
                                   k_folds=k_folds, n_repeats=n_repeats, R=R)
        dfeq["Escenario"] = "Σ0=Σ1"
        dfne = riesgos_cv_modelos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1,
                                   k_folds=k_folds, n_repeats=n_repeats, R=R)
        dfne["Escenario"] = "Σ0≠Σ1"
        frames += [dfeq, dfne]
    df_cv = pd.concat(frames, ignore_index=True)
    return df_cv.sort_values(["Escenario","Modelo","n"]).reset_index(drop=True)

def errores_bayes_por_escenario(mu0, mu1, Sigma0, Sigma1, pi_0, n_sim=250000):
    err_eq  = bayes_error_equal_cov(mu0, mu1, Sigma0, pi_0)
    err_neq = bayes_error_montecarlo(mu0, mu1, Sigma0, Sigma1, pi_0, n_sim=n_sim)
    return {"Σ0=Σ1": err_eq, "Σ0≠Σ1": err_neq}

def plot_por_modelo_y_escenario(df_mc, df_cv, err_bayes_dict, modelo):
    for esc in ["Σ0=Σ1","Σ0≠Σ1"]:
        d1 = df_mc[(df_mc["Modelo"]==modelo) & (df_mc["Escenario"]==esc)].sort_values("n")
        d2 = df_cv[(df_cv["Modelo"]==modelo) & (df_cv["Escenario"]==esc)].sort_values("n")
        if d1.empty or d2.empty: 
            continue
        plt.figure(figsize=(7,5))
        plt.plot(d1["n"], d1["mean"], "-o", label="Monte Carlo")
        plt.errorbar(d1["n"], d1["mean"], yerr=d1["std"], fmt="none", capsize=4)
        plt.plot(d2["n"], d2["mean"], "--x", label="CV estratificada")
        plt.errorbar(d2["n"], d2["mean"], yerr=d2["std"], fmt="none", capsize=4)
        plt.axhline(err_bayes_dict[esc], ls="--", lw=2, color="tab:red", label="Error de Bayes")
        plt.title(f"{modelo} — {esc}")
        plt.xlabel("Tamaño de muestra (n)")
        plt.ylabel("Riesgo estimado")
        plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.show()
        
 # genera resultados para ambos escenarios

DIFICULTADES = {
    "Fácil":   (np.array([2, 2]), np.array([5.0, 5.0])),
    "Medio":   (np.array([2, 2]), np.array([3.5, 3.5])),
    "Difícil": (np.array([2, 2]), np.array([1.5, 1.5])),
}

def Resultados_MC_CV_por_dificultad(
    p, pi_0, Sigma0, Sigma1, dificultades=DIFICULTADES,
    n_grid=(50,100,200,500), k_folds=5, n_repeats=1, R=20, n_sim_bayes=250000
):
    frames_mc, frames_cv = [], []
    err_bayes = {}  # clave: (escenario, dificultad) -> valor

    for dif_lbl, (mu0, mu1) in dificultades.items():
        # --- Escenario 1: Σ0 = Σ1 ---
        for n in n_grid:
            d_mc = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma0)
            d_mc["Escenario"] = "Σ0=Σ1"; d_mc["Dificultad"] = dif_lbl
            frames_mc.append(d_mc)

            d_cv = riesgos_cv_modelos(n, p, pi_0, mu0, mu1, Sigma0, Sigma0,
                                      k_folds=k_folds, n_repeats=n_repeats, R=R)
            d_cv["Escenario"] = "Σ0=Σ1"; d_cv["Dificultad"] = dif_lbl
            frames_cv.append(d_cv)

        err_bayes[("Σ0=Σ1", dif_lbl)] = bayes_error_equal_cov(mu0, mu1, Sigma0, pi_0)

        # --- Escenario 2: Σ0 ≠ Σ1 ---
        for n in n_grid:
            d_mc = Riesgos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1)
            d_mc["Escenario"] = "Σ0≠Σ1"; d_mc["Dificultad"] = dif_lbl
            frames_mc.append(d_mc)

            d_cv = riesgos_cv_modelos(n, p, pi_0, mu0, mu1, Sigma0, Sigma1,
                                      k_folds=k_folds, n_repeats=n_repeats, R=R)
            d_cv["Escenario"] = "Σ0≠Σ1"; d_cv["Dificultad"] = dif_lbl
            frames_cv.append(d_cv)

        err_bayes[("Σ0≠Σ1", dif_lbl)] = bayes_error_montecarlo(
            mu0, mu1, Sigma0, Sigma1, pi_0, n_sim=n_sim_bayes
        )

    df_mc = (pd.concat(frames_mc, ignore_index=True)
               .sort_values(["Escenario","Dificultad","Modelo","n"])
               .reset_index(drop=True))
    df_cv = (pd.concat(frames_cv, ignore_index=True)
               .sort_values(["Escenario","Dificultad","Modelo","n"])
               .reset_index(drop=True))
    return df_mc, df_cv, err_bayes


def plot_modelo_por_dificultad(df_mc, df_cv, err_bayes_dict, 
                               modelo, escenario, orden=("Fácil","Medio","Difícil")):
    # Reserva margen superior desde el inicio
    fig, axes = plt.subplots(
        1, len(orden), figsize=(6*len(orden), 4), sharey=True, constrained_layout=False
    )  # <-- agregado constrained_layout=False
    if len(orden) == 1:
        axes = [axes]

    for ax, dif in zip(axes, orden):
        d1 = df_mc[(df_mc["Modelo"]==modelo) &
                   (df_mc["Escenario"]==escenario) &
                   (df_mc["Dificultad"]==dif)].sort_values("n")
        d2 = df_cv[(df_cv["Modelo"]==modelo) &
                   (df_cv["Escenario"]==escenario) &
                   (df_cv["Dificultad"]==dif)].sort_values("n")

        if d1.empty and d2.empty:
            ax.set_visible(False)
            continue

        # Hold-out (MC)
        ax.plot(d1["n"], d1["mean"], "-o", label="Monte Carlo")
        ax.errorbar(d1["n"], d1["mean"], yerr=d1["std"], fmt="none", capsize=4)

        # CV estratificada
        ax.plot(d2["n"], d2["mean"], "--x", label="CV estratificada")
        ax.errorbar(d2["n"], d2["mean"], yerr=d2["std"], fmt="none", capsize=4)

        # Error de Bayes específico (escenario, dificultad)
        ax.axhline(err_bayes_dict[(escenario, dif)], ls="--", lw=2,
                   color="tab:red", label="Error de Bayes")

        ax.set_title(f"{dif}")
        ax.set_xlabel("Tamaño de muestra (n)")
        ax.set_ylabel("Riesgo estimado")
        ax.grid(alpha=0.25)

    # Leyenda y título dentro del canvas
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))  # <--
    fig.suptitle(f"{modelo} — {escenario}", y=0.94)  # <-- baja un poco el suptitle

    # Reserva espacio arriba para suptitle + leyenda
    fig.tight_layout(rect=[0, 0, 1, 0.88])  

    plt.show()

np.random.seed(546)  
# Parámetros del experimento
p = 2
pi_0 = 0.5 #balance
Sigma0 = np.eye(p)
Sigma1 = np.array([[1, 0.5],[0.5, 1]])

# Corre todo (ambos escenarios y tres dificultades)
df_mc, df_cv, err_bayes_dict = Resultados_MC_CV_por_dificultad(
    p, pi_0, Sigma0, Sigma1, DIFICULTADES,
    n_grid=(50,100,200,500), k_folds=5, R=20
)

# Una figura por clasificador y escenario, con 3 subplots (Fácil/Medio/Difícil)
for modelo in ["LDA", "QDA", "Naive Bayes", "Fisher (1D + umbral)"]:
    plot_modelo_por_dificultad(df_mc, df_cv, err_bayes_dict, modelo, escenario="Σ0=Σ1")
    plot_modelo_por_dificultad(df_mc, df_cv, err_bayes_dict, modelo, escenario="Σ0≠Σ1")    

# Parámetros del experimento
p = 2
pi_0 = 0.8 #desbalance
Sigma0 = np.eye(p)
Sigma1 = np.array([[1, 0.5],[0.5, 1]])

# Corre todo (ambos escenarios y tres dificultades)
df_mc, df_cv, err_bayes_dict = Resultados_MC_CV_por_dificultad(
    p, pi_0, Sigma0, Sigma1, DIFICULTADES,
    n_grid=(50,100,200,500), k_folds=5, R=20
)

# Una figura por clasificador y escenario, con 3 subplots (Fácil/Medio/Difícil)
for modelo in ["LDA", "QDA", "Naive Bayes", "Fisher (1D + umbral)"]:
    plot_modelo_por_dificultad(df_mc, df_cv, err_bayes_dict, modelo, escenario="Σ0=Σ1")
    plot_modelo_por_dificultad(df_mc, df_cv, err_bayes_dict, modelo, escenario="Σ0≠Σ1")    



