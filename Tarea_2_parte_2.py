import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

def generar_datos_aleatorios(mu0, mu1, sigma0, sigma1, n=100, pi0=0.5):
    n0 = int(n * pi0)
    n1 = n - n0
    x0 = np.random.multivariate_normal(mu0, sigma0, n0)
    x1 = np.random.multivariate_normal(mu1, sigma1, n1)
    X = np.vstack((x0, x1))
    y = np.array([0]*n0 + [1]*n1)
    return X, y

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
    else:  
        mu0 = np.zeros(p)
        mu1 = np.ones(p) * 0.5

    # Covarianzas
    sigma0 = np.array([[1, 0.5], [0.5, 1]])
    
    if covarianzas_iguales:
        sigma1 = sigma0
    else:
        sigma1 = np.array([[1, -0.5], [-0.5, 1.5]])

    return mu0, mu1, sigma0, sigma1

# Parámetros
dificultad = 'medio'  # Eleccion de Dificultad
cov_iguales = False   # True para LDA, False para QDA
n = 200               # Numero de muestra
pi0 = 0.5              
k = 4                 # Numero de vecinos

mu0, mu1, sigma0, sigma1 = configurar_escenario(dificultad, cov_iguales)
X, y = generar_datos_aleatorios(mu0, mu1, sigma0, sigma1, n=n, pi0=pi0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Precisión (k={k}, dificultad={dificultad}, cov_iguales={cov_iguales}): {acc:.3f}")

plot_de_decision_boundary(X_test, y_test, knn, title=f"k-NN (k={k}) - Precisión={acc:.2f}")