#FastICA algorithm 
#Sklearn implementation seem to be more complete (handling several decompositions, degenerate case)
#to improve? 
import numpy as np
def g(x):
    y = np.tanh(x)
    return y

def g_prime(x):
    y =  np.ones(x.shape) - np.square(np.tanh(x))
    return y 

def center_matrix(X):
    mean = X.mean(axis=1).reshape(X.shape[0],1) @ np.ones((1, X.shape[1]))
    centered = X - mean
    return centered, mean 

def whiten_matrix_eigenvalues(X, eps=1e-5):
    # Calculate the covariance matrix
    var = np.cov(X)
    # Perform eigenvalues decomposition 
    eigenvalues, eigenvectors = np.linalg.eig(var)
    # We need real values 
    real_eigenvalues, real_eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    # Inverse the eigenvalues to get the whitening matrix, correct the small one to get finite values
    inverse_eigenvalues = np.diag(1/np.sqrt(real_eigenvalues+eps))
    whitening_matrix = inverse_eigenvalues @ real_eigenvectors.T
    whitened = whitening_matrix @ X 
    return whitened, whitening_matrix



def whiten_matrix_svd(X, eps=1e-5):
    # Calculate the covariance matrix
    var = np.cov(X)
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(var)

    # Get inverse of singular values and use correction
    inverse_s_values = np.diag(1/np.sqrt(S + eps))
    whitening_matrix = U @ inverse_s_values @ Vt
    whitened = whitening_matrix @ X
    return whitened, whitening_matrix

def one_unit_step(X,w ):
    y = np.dot(w,X)
    g_y = g(y)
    g_prime_y = g_prime(y)
    newton_update = np.mean(X @ g_y.T, axis=1)- np.mean(g_prime_y,axis=1)@w
    return(newton_update)

def gram_schmidt_orthogonalization(W,w, c):
    # Decorrelate
    w_decorr = w - (w @ W[:c,:].T) @ W[:c,:]
    # Normalize
    w_ortho = w_decorr / np.sqrt((w_decorr ** 2).sum())
    return(w_ortho)

def fastICA(X, n_components=None, whitening= 'svd',tol=1e-10, max_iter=5000):
    T,N = X.shape
    assert whitening in ["svd", "eigenvalues"], "Invalid value for 'whitenning'. Must be 'svd' or 'eigenvalues'."
    assert N >= n_components , "Invalid number of components, must be under number of features"
    if n_components is None:
        n_components = N
    X, meanX = center_matrix(X, N)
    if whitening == "svd":
        X, whiteningX = whiten_matrix_svd(X)
    else : 
        X, whiteningX = whiten_matrix_eigenvalues(X)
    W = np.zeros((n_components, N))
    for i in range(n_components):
        wp = np.random.rand(1, N)
        for _ in range(max_iter):
            wp = one_unit_step(X, wp)
            wp = gram_schmidt_orthogonalization(W, wp, i)
            W[i,:] = wp
    return W @ X , whiteningX @ ( W @ X + meanX )
