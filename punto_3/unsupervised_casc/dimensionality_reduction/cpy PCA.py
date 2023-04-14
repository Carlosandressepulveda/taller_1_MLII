Parámetros:
-----------
num_componentes : int
    Número de componentes principales a calcular.

Atributos:
-----------
mis_componentes_ : similar a un array, forma (num_componentes, num_caracteristicas)
    Componentes principales.

mi_varianza_explicada_ : similar a un array, forma (num_componentes,)
    Cantidad de varianza explicada por cada uno de los componentes seleccionados.

mi_proporcion_varianza_explicada_ : similar a un array, forma (num_componentes,)
    Porcentaje de varianza explicada por cada uno de los componentes seleccionados.

mi_media_ : similar a un array, forma (num_caracteristicas,)
    Media de cada característica en los datos originales.

Métodos:
--------
ajustar_transformar(X): Calcula el PCA de una matriz dada X y devuelve sus valores singulares, vectores singulares izquierdos y derechos.
ajustar_transformar_truncado(X): Calcula el PCA truncado de una matriz dada X y devuelve sus valores singulares, vectores singulares izquierdos y derechos.
"""

def __init__(self, num_componentes=None):
    self.num_componentes = num_componentes
    self.mis_componentes_ = None
    self.mi_varianza_explicada_ = None
    self.mi_proporcion_varianza_explicada_ = None
    self.mi_media_ = None

def ajustar(self, X):
    """
    Ajusta el modelo de PCA a los datos dados.

    Parámetros:
    -----------
    X : similar a un array, forma (n_muestras, n_caracteristicas)
        Datos de entrenamiento.
    """
    # Calcula la media de cada característica
    self.mi_media_ = np.mean(X, axis=0)

    # Centra los datos restando la media
    X_centralizado = X - self.mi_media_

    # Calcula la matriz de covarianza
    cov = np.cov(X_centralizado.T)

    # Calcula los vectores y valores propios de la matriz de covarianza
    eigvals, eigvecs = np.linalg.eig(cov)

    # Ordena los vectores y valores propios en orden descendente de valor propio
    eigvecs = eigvecs.T
    idxs = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[idxs]
    eigvals = eigvals[idxs]

    # Normaliza los vectores propios
    norma = np.sqrt(np.sum(np.power(eigvecs, 2), axis=1))
    eigvecs /= norma.reshape(-1, 1)

    # Almacena los primeros num_componentes vectores propios
    if self.num_componentes is not None:
        self.mis_componentes_ = eigvecs[:self.num_componentes]
        self.mi_varianza_explicada_ = eigvals[:self.num_componentes]
    else:
        self.mis_componentes_ = eigvecs
        self.mi_varianza_explicada_ = eigvals

    # Calcula el porcentaje de varianza explicada por cada componente
    self.mi_proporcion_varianza_explicada_ = self.mi_varianza_explicada_ / np.sum(self.mi_varianza_explicada_)

def transformar(self, X):
    """

