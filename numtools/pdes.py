# Description: Partial differential equation solvers.
# Author/date: André Palóczy, February/2017
# E-mail:      paloczy@gmail.com

__all__ = ['pjacobi_poissoneq',
           'gaussseidel_poissoneq']

import numpy as np

def pjacobi_poissoneq(RHS, phi0, tol=1e-2, max_iter=1e3):
    """
    Solve the 2D Poisson equation on a uniform grid with
    isotropic spacing (dx=dy) using the Point Jacobi method.

    REFERENCE
    ---------
    P. Moin (2010), Section 5.10.2
    """
    M, N = phi0.shape

    phi = phi0.copy()
    phip = phi + np.random.random(phi.shape)
    k=0
    while np.abs(phip-phi).max()>tol and k<=max_iter:
        print((np.abs(phip-phi).max(),tol))
        phi = phip
        for j in range(1, M-1):
            for i in range(1, N-1):
                phip[j,i] = 0.25*(phi[j-1,i] + phi[j+1,i] + phi[j,i-1] + phi[j,i+1] - RHS[j,i]) # Moin (2010) eq. 5.70.
        k+=1

    return phip

def gaussseidel_poissoneq(A, x0):
    """
    Solve an elliptic PDE with the Gauss-Seidel method.
    """
    return 1
