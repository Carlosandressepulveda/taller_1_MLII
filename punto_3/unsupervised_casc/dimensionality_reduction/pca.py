import numpy as np

class PCA:
    """
    Análisis de Componentes Principales (PCA)
    
    Parámetros:
    -----------
    n_components : int
        Número de componentes principales a calcular.
        
    Atributos:
    -----------
    components : array-like, shape (n_features, n_components)
        Componentes principales.
        
    mean : array-like, shape (n_features,)
        Media de cada característica en los datos originales.
        
    Métodos:
    --------
    fit(X): Ajusta el modelo de PCA a los datos dados.
    transform(X): Transforma los datos dados al espacio de componentes principales.
    """
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Ajusta el modelo de PCA a los datos dados.
        
        Parámetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrenamiento.
        """
        # Centrar los datos
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calcular la matriz de covarianza
        cov = np.cov(X, rowvar=False)

        # Calcular los autovalores y autovectores de la matriz de covarianza
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Ordenar los autovalores y autovectores en orden descendente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Almacenar los primeros n_components autovectores como componentes principales
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X):
        """
        Transforma los datos dados al espacio de componentes principales.
        
        Parámetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Datos a transformar.
        
        Retorna:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Datos transformados.
        """
        # Centrar los datos
        X = X - self.mean

        # Proyectar los datos sobre los componentes principales
        X_transformed = np.dot(X, self.components)

        return X_transformed
