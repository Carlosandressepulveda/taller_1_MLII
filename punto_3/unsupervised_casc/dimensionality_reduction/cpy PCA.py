import numpy as np

class MyPCA:
    """
    Principal Component Analysis (PCA)
    
    Parameters:
    -----------
    num_components : int
        Number of principal components to compute.
        
    Attributes:
    -----------
    my_components_ : array-like, shape (num_components, num_features)
        Principal components.
        
    my_explained_variance_ : array-like, shape (num_components,)
        Amount of variance explained by each of the selected components.
        
    my_explained_variance_ratio_ : array-like, shape (num_components,)
        Percentage of variance explained by each of the selected components.
        
    my_mean_ : array-like, shape (num_features,)
        Mean of each feature in the original data.
    Methods:
    --------
        fit_transform(X): Computes the PCA of a given matrix X and returns its singular values, left and right singular vectors.
        fit_transform_truncated(X): Computes the truncated PCA of a given matrix X and returns its singular values, left and right singular vectors.
    """
    
    def __init__(self, num_components=None):
        self.num_components = num_components
        self.my_components_ = None
        self.my_explained_variance_ = None
        self.my_explained_variance_ratio_ = None
        self.my_mean_ = None
        
    def fit(self, X):
        """
        Fit the PCA model to the given data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        # Compute the mean of each feature
        self.my_mean_ = np.mean(X, axis=0)
        
        # Center the data by subtracting the mean
        X_centered = X - self.my_mean_
        
        # Compute the covariance matrix
        cov = np.cov(X_centered.T)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov)
        
        # Sort the eigenvectors and eigenvalues in descending order of eigenvalue
        eigvecs = eigvecs.T
        idxs = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[idxs]
        eigvals = eigvals[idxs]
        
        # Normalize the eigenvectors
        norm = np.sqrt(np.sum(np.power(eigvecs, 2), axis=1))
        eigvecs /= norm.reshape(-1, 1)
        
        # Store the first num_components eigenvectors
        if self.num_components is not None:
            self.my_components_ = eigvecs[:self.num_components]
            self.my_explained_variance_ = eigvals[:self.num_components]
        else:
            self.my_components_ = eigvecs
            self.my_explained_variance_ = eigvals
        
        # Compute the percentage of variance explained by each component
        self.my_explained_variance_ratio_ = self.my_explained_variance_ / np.sum(self.my_explained_variance_)
        
    def transform(self, X):
        """
        Transform the given data into the principal component space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, num_components)
            Transformed data.
        """
        # Center the data by subtracting the mean
        X_centered = X - self.my_mean_
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.my_components_.T)
        
        return X
