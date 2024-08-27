import numpy as np

def torsion(orig_factors, model='minimum-torsion', 
		method='exact', max_niter=10000):
    """
    This function computes the Principal Components torsion and the Minimum Torsion for diversification analysis.
    See A. Meucci, A. Santangelo, R. Deguest - "Measuring Portfolio Diversification Based on Optimized Uncorrelated Factors" to appear (2013).
    """

    Sigma = orig_factors.cov()

    if method is None:
        method = 'exact'
    
    if model == 'pca':
        # PCA decomposition
        e, lambda_ = np.linalg.eig(Sigma)
        flip = e[0, :] < 0
        e[:, flip] = -e[:, flip]  # fix the sign of the eigenvector based on the sign of its first entry
        index = np.argsort(np.diag(lambda_))[::-1]
        
        # PCA torsion
        t = e[:, index].T
        
    elif model == 'minimum-torsion':
        # Correlation matrix
        sigma = np.sqrt(np.diag(Sigma))
        C = np.diag(1. / sigma) @ Sigma @ np.diag(1. / sigma)
        c = np.linalg.sqrtm(C)  # Riccati root of C
        
        if method == 'approximate':
            t = (np.diag(sigma) @ np.linalg.inv(c)) @ np.diag(1. / sigma)
            
        elif method == 'exact':
            n_ = Sigma.shape[0]
            
            # initialize
            d = np.ones(n_)
            f = np.zeros(max_niter)
            for i in range(max_niter):
                U = np.diag(d) @ c @ c @ np.diag(d)
                u = np.linalg.sqrtm(U)
                q = np.linalg.solve(u, np.diag(d) @ c)
                d = np.diag(q @ c)
                pi_ = np.diag(d) @ q  # perturbation
                f[i] = np.linalg.norm(c - pi_, 'fro')
                
                if i > 0 and abs(f[i] - f[i-1]) / f[i] / n_ <= 1e-8:
                    f = f[:i+1]
                    break
                elif i == max_niter - 1 and abs(f[max_niter - 1] - f[max_niter - 2]) / f[max_niter - 1] / n_ > 1e-8:
                    print(f'number of max iterations reached: n_iter = {max_niter}')
                    
            x = pi_ @ np.linalg.inv(c)
            t = np.diag(sigma) @ x @ np.diag(1. / sigma)
    
    return t