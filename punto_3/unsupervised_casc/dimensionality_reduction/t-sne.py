import numpy as np

class t_sne:
    """
    Esta clase proporciona una implementación del algoritmo t-SNE (t-Distributed Stochastic Neighbor Embedding) utilizando NumPy.
    Permite calcular el t-SNE de una matriz dada, así como el t-SNE truncado.
    
    Atributos:
    -----------
        n_components (int): El número de componentes a mantener en el t-SNE truncado.
    
    Métodos:
    --------
        fit_transform(X): Calcula el t-SNE de una matriz dada X y devuelve los puntos incorporados en un espacio de menor dimensión.
        fit_transform_truncated(X): Calcula el t-SNE truncado de una matriz dada X y devuelve los puntos incorporados truncados en un espacio de menor dimensión.
    
    """

    def __init__(self, n_components=None):
        """
        Constructor de la clase t_sne.
        
        Args:
            n_components (int): El número de componentes a mantener en el t-SNE truncado. Por defecto, None.
        """
        self.n_components = n_components

    def fit_transform(self, X):
        """
        Calcula el t-SNE de una matriz dada X y devuelve los puntos incorporados en un espacio de menor dimensión.
        
        Args:
            X (numpy.ndarray): La matriz de entrada para calcular el t-SNE.
            
        Returns:
            numpy.ndarray: Un array con los puntos incorporados en un espacio de menor dimensión.
        """
        # Código para calcular el t-SNE
        
        return embedded_points

    def fit_transform_truncated(self, X):
        """
        Calcula el t-SNE truncado de una matriz dada X y devuelve los puntos incorporados truncados en un espacio de menor dimensión.
        
        Args:
            X (numpy.ndarray): La matriz de entrada para calcular el t-SNE truncado.
            
        Returns:
            numpy.ndarray: Un array con los puntos incorporados truncados en un espacio de menor dimensión.
        """
        # Código para calcular el t-SNE truncado
        
        return embedded_points_truncated
