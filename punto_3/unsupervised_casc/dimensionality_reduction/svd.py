Atributos:
-----------
    n_components (int): El número de componentes a mantener en el SVD truncado.

Métodos:
--------
    fit_transform(X): Calcula el SVD de una matriz dada X y devuelve sus valores singulares, vectores singulares izquierdos y vectores singulares derechos.
    fit_transform_truncated(X): Calcula el SVD truncado de una matriz dada X y devuelve sus valores singulares truncados, vectores singulares izquierdos truncados y vectores singulares derechos truncados.

"""

def __init__(self, n_components=None):
    """
    Constructor de la clase SVD.
    
    Args:
        n_components (int): El número de componentes a mantener en el SVD truncado. Por defecto, None.
    """
    self.n_components = n_components

def fit_transform(self, X):
    """
    Calcula el SVD de una matriz dada X y devuelve sus valores singulares, vectores singulares izquierdos y vectores singulares derechos.
    
    Args:
        X (numpy.ndarray): La matriz de entrada para calcular el SVD.
        
    Returns:
        tuple: Una tupla con los valores singulares, vectores singulares izquierdos y vectores singulares derechos.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S, U, Vt.T

def fit_transform_truncated(self, X):
    """
    Calcula el SVD truncado de una matriz dada X y devuelve sus valores singulares truncados, vectores singulares izquierdos truncados y vectores singulares derechos truncados.
    
    Args:
        X (numpy.ndarray): La matriz de entrada para calcular el SVD truncado.
        
    Returns:
        tuple: Una tupla con los valores singulares truncados, vectores singulares izquierdos truncados y vectores singulares derechos truncados.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if self.n_components is None:
        return S, U, Vt.T
    else:
        return S[:self.n_components], U[:, :self.n_components], Vt[:self.n_components, :]
