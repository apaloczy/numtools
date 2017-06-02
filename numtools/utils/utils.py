# -*- coding: utf-8 -*-
#
# Description: Miscellaneous utility functions.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['newton']

import numpy as np

def newton(F, dFdx, guess=None, tol=1e-12, max_iter=1e4, verbose=True):
    """
    Solves a nonlinear algebraic equation of the form

    F(x) = nonlinear_function(x)

    using the Newton-Raphson iterative method.
    """
    assert callable(F), "'F' is not a function."
    assert callable(dFdx), "'dFdx' is not a function."

    if not guess:
        guess = 10.*tol
    xn = guess

    niter=0
    while F(xn)>tol and niter<max_iter:
        xn = xn - Fn(xn)/dFfx(xn)
        niter+=1

    if verbose:
        if niter<max_iter:
            print("Converged to x = %.f after %d iterations."%(xn, niter))
        else:
            print("Did not converge after %d iterations"%niter)

    return xn
