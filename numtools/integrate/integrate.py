# -*- coding: utf-8 -*-
#
# Description: Numerical tools for calculating integrals.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['rectangle',
           'trapezoid']
        #    'simpson']

import numpy as np

def rectangle(x, f, fpoint='middle', cumulative=False):
    """
    FIXME: Mid-point option seems to be not totally right (using only every other data point). Perhaps the rectangle
    rule is supposed to work only for evenly spaced meshes (https://en.wikipedia.org/wiki/Rectangle_method, http://www.wolframalpha.com/input/?i=rectangle+rule)?

    The numerical approximation of the definite integral of some tabular function
    (i.e., a function whose analytical form is unknown but whose values are
    specified in an array) by Simpson's Rule is

    R(f) = ...

    REFERENCE
    ---------
    P. Moin..., p. x.
    """
    if fpoint=='middle':
        if x.size%2==0:
            raise ValueError("Number of points must be odd if using the mid-points.")

    x, f = map(np.asarray, (x, f))
    assert fpoint in ['left', 'middle', 'right'], "fpoint must be either 'left', 'middle' or 'right'."

    if fpoint=='middle':
        dx = x[2::2] - x[::2][:-1]
    else:
        dx = x[1:] - x[:-1]

    if fpoint=='left':
        frect = f[:-1]     # f evaluated at the left-side points of the rectangles.
    elif fpoint=='right':
        frect = f[1:]      # f evaluated at the right-side points of the rectangles.
    elif fpoint=='middle':
        frect = f[1:][::2] # f evaluated at the mid-points of the rectangles.

    if cumulative:
        I = np.cumsum(frect*dx)
        if fpoint=='middle': # Coordinates of the cumulative integral.
            xI = x[1:][::2]
        elif fpoint=='left':
            xI = x[:-1]
        elif fpoint=='right':
            xI = x[1:]
        return xI, I
    else:
        I = np.sum(frect*dx)
        return I

def trapezoid(x, f, cumulative=False):
    """
    The numerical approximation of the definite integral of some tabular function
    (i.e., a function whose analytical form is unknown but whose values are
    specified in an array) by the Trapezoid Rule is

    T(f) = ...

    Geometrically, this means approximating the area under the graph of f(x)
    by a trapezoid with width x(i+1) - x(i), left height f(i) and right
    height f(i+1).

    REFERENCE
    ---------
    P. Moin..., p. x.
    """
    x, f = map(np.asarray, (x, f))

    dx = x[1:] - x[:-1]
    fm = (f[1:] + f[:-1])/2. # f evaluated at the mid-points.

    if cumulative:
        I = np.cumsum(fm*dx)
        xI = (x[1:] + x[:-1])/2. # Coordinates of the cumulative integral.
        return xI, I
    else:
        I = np.sum(fm*dx)
        return I

# def simpson(x, f, cumulative=False):
#     """
#     FIXME: https://en.wikipedia.org/wiki/Simpson%27s_rule
#
#     The numerical approximation of the definite integral of some tabular function
#     (i.e., a function whose analytical form is unknown but whose values are
#     specified in an array) by Simpson's Rule is
#
#     S[f(i)] = [ f(i-1) + 4*f(i) + f(i+1) ]*dx/3
#
#     REFERENCE
#     ---------
#     P. Moin..., p. x.
#     """
#     if x.size%2==0:
#         raise ValueError("Number of points must be odd.")
#     x, f = map(np.asarray, (x, f))
#
#     dx = x[1:] - x[:-1]
#     S = (f[:-2] + 4*f[1:-1] + f[2:])*dx/3. # Area of parabola centered at point i.
#
#     if cumulative:
#         I = np.cumsum(S)
#         xI = x[1:-1] # Coordinates of the cumulative integral.
#         return xI, I
#     else:
#         I = np.sum(S)
#         return I
