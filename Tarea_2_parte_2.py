import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal

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
pi0 = 0.5
k = 4

#----------Almacenamiento de precisión--------
Muestras=5
LDA_datos_k_nn=np.zeros(Muestras)
QDA_datos_k_nn=np.zeros(Muestras)
Bayes_datos=np.zeros(Muestras)
#---------------------------------------------



mu0, mu1, sigma0, sigma1 = configurar_escenario(dificultad, cov_iguales)
X, y = generar_datos_aleatorios(mu0, mu1, sigma0, sigma1, n=n, pi0=pi0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Clasificador k-NN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"k-NN (k={k}): Precisión = {acc_knn:.3f}")

plot_de_decision_boundary(X_test, y_test, knn, title=f"k-NN (k={k}) - Precisión={acc_knn:.2f}")

# --- Bayes Óptimo ---
bayes = BayesOptimo(mu0, mu1, sigma0, sigma1, pi0=pi0)
y_pred_bayes = bayes.predict(X_test)
acc_bayes = accuracy_score(y_test, y_pred_bayes)

print(f"Bayes Óptimo (densidades verdaderas): Precisión = {acc_bayes:.3f}")
plot_de_decision_boundary(X_test, y_test, bayes,
                          title=f"Bayes Óptimo - Precisión={acc_bayes:.2f}")

# --- Fisher ---
fisher = FisherClassifier(mu0, mu1, sigma0, sigma1)
y_pred_fisher = fisher.predict(X_test)
acc_fisher = accuracy_score(y_test, y_pred_fisher)

print(f"Fisher: Precisión = {acc_fisher:.3f}")
plot_de_decision_boundary(X_test, y_test, fisher,
                          title=f"Fisher - Precisión={acc_fisher:.2f}")

