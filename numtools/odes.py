# Description: Ordinary differential equation solvers.
# Author/date: André Palóczy, February/2017
# E-mail:      paloczy@gmail.com

__all__ = ['eeuler',
           'ieuler',
           'crank_nicolson',
           'runge_kutta2',
           #'runge_kutta3',
           'runge_kutta4',
           #'runge_kutta45',
           'adams_bashforth2']

import numpy as np

def eeuler(f, y0, dt=0.1, ti=0., tf=100., verbose=False):
    """
    USAGE
    -----
    ynum, tnum = eeuler(f, y0, dt=0.1, ti=0., tf=100., verbose=False)

    Solves the first-order Initial Value Problem

    y'(t) = f(y(t), t);         y(t0) = y0

    using the Explicit Euler method, i.e.,
    y_(n+1) = y_n + dt*f(y_n, t_n).

    'f' is a function object based on the form of the right-hand side
    of the ODE that returns f(y,t). 'y0' is the initial condition, 'dt'
    is the time step, 'ti' is the start time and 'tf' is the end time. The
    numerical solution 'ynum' and the numerical grid 'tnum' are returned.
    """
    assert callable(f), "'f' is not a function."
    y0, dt, ti, tf = map(np.float, (y0, dt, ti, tf))
    t = np.arange(ti, tf+dt, dt) # Create time axis.

    N = t.size
    Y = [y0]
    yn = y0
    tn = ti
    n = 0
    while n<N:
        fn = f(yn, tn)   # Evaluate the RHS at time t = tn.
        ynp = yn + dt*fn # Time-step the solution to time t = tn + dt.
        Y.append(ynp)    # Store the solution at  t = tn + dt.
        yn=ynp           # Update the value of the solution at t = tn.
        tn+=dt           # Update time.
        n+=1             # Update counter.
        if verbose:
            print("Time-stepping: Step %d of %d."%(n,N))

    Y = np.array(Y[:-1]) # Eliminate last timestep (overshoots tf in the while loop).
    return Y, t

def ieuler():
    return 1

def crank_nicolson(f, dfdy, y0, dt=0.1, ti=0., tf=100., linearize=True, tol=1e-10, max_iter=1e3, verbose=False):
    """
    USAGE
    -----
    ynum, tnum = crank_nicolson(f, dfdy, y0, dt=0.1, ti=0., tf=100., linearize=True, tol=1e-10, max_iter=1e3, verbose=False)

    Solves the first-order Initial Value Problem

    y'(t) = f(y(t), t);         y(t0) = y0

    using the Crank-Nicolson (trapezoid) method, i.e.,
    y_(n+1) = y_n + (dt/2)*[f(y_(n+1), t_(n+1)) + f(y_n, t_n)].

    'f' is a function object based on the form of the right-hand side
    of the ODE that returns f(y,t), and 'dfdy' is its derivative (determined
    analytically). 'y0' is the initial condition, 'dt' is the time step,
    'ti' is the start time and 'tf' is the end time. The numerical solution
    'ynum' and the numerical grid 'tnum' are returned.

    If 'linearize' is True (default) the time-stepping is done
    with the linearized form of f(y_(n+1), t_(n+1)), see e.g.,
    Moin (2010), page 62, equation (4.18).
    """
    assert callable(f), "'f' is not a function."
    assert callable(dfdy), "'dfdy' is not a function."
    y0, dt, ti, tf = map(np.float, (y0, dt, ti, tf))
    t = np.arange(ti, tf+dt, dt) # Create time axis.

    coeff = dt/2.
    N = t.size
    Y = [y0]
    yn = y0
    tn = ti
    n = 0
    while n<N:
        fn = f(yn, tn)   # Evaluate the RHS at time t = tn.
        # First time-step using the linearized form of f(y, t).
        tnp = tn + dt
        num = f(yn, tnp) + fn
        den = 1. - coeff*dfdy(yn, tnp)
        ynp = yn + coeff*num/den
        # Time-step the solution to time t = tn + dt
        # using the Newton-Raphson method if linearize==False.
        # e.g., Hobs ... (page x, equation x.x)
        # First guess is the linearized solution for y_(n+1).
        if not linearize:
            F = ynp - coeff*f(ynp, tnp) - yn - coeff*fn #tol*10.
            niter=0
            while np.abs(F)>tol and niter<max_iter:
                dFdy = 1. - coeff*dfdy(yn, tnp)
                ynp = ynp - F/dFdy
                F = ynp - coeff*f(ynp, tnp) - yn - coeff*fn
                niter+=1

        Y.append(ynp)    # Store the solution at  t = tn + dt.
        yn=ynp           # Update the value of the solution at t = tn.
        tn+=dt           # Update time.
        n+=1             # Update counter.
        if verbose:
            print("Time-stepping: Step %d of %d."%(n,N))

    Y = np.array(Y[:-1]) # Eliminate last timestep (overshoots tf in the while loop).
    return Y, t

def runge_kutta2(f, y0, dt=0.1, ti=0., tf=100., verbose=False):
    """
    Second-order Runge-Kutta (RK2) scheme.
    """
    assert callable(f), "'f' is not a function."
    y0, dt, ti, tf = map(np.float, (y0, dt, ti, tf))
    t = np.arange(ti, tf+dt, dt)  # Create time axis.

    N = t.size
    Y = [y0]
    yn = y0
    tn = ti
    n = 0
    while n<N:
        k1 = dt*f(yn, tn)
        k2 = dt*f(yn + k1/2., tn + dt/2.)
        ynp = yn + k2
        Y.append(ynp)    # Store the solution at  t = tn + dt.
        yn=ynp           # Update the value of the solution at t = tn.
        tn+=dt           # Update time.
        n+=1             # Update counter.
        if verbose:
            print("Time-stepping: Step %d of %d."%(n,N))

    Y = np.array(Y[:-1]) # Eliminate last timestep (overshoots tf in the while loop).
    return Y, t

def runge_kutta4(f, y0, dt=0.1, ti=0., tf=100., verbose=False):
    """
    Fourth-order Runge-Kutta (RK4) scheme.
    """
    assert callable(f), "'f' is not a function."
    y0, dt, ti, tf = map(np.float, (y0, dt, ti, tf))
    t = np.arange(ti, tf+dt, dt)  # Create time axis.
    c1, c2, c3 = 1/6., 1/3., 1/6. # Set RK4 coefficients.

    N = t.size
    Y = [y0]
    yn = y0
    tn = ti
    n = 0
    while n<N:
        tnm = tn + dt/2.
        k1 = dt*f(yn, tn)
        k2 = dt*f(yn + k1/2., tnm)
        k3 = dt*f(yn + k2/2., tnm)
        k4 = dt*f(yn + k3, tn + dt)
        ynp = yn + c1*k1 + c2*(k2 + k3) + c3*k4
        Y.append(ynp)    # Store the solution at  t = tn + dt.
        yn=ynp           # Update the value of the solution at t = tn.
        tn+=dt           # Update time.
        n+=1             # Update counter.
        if verbose:
            print("Time-stepping: Step %d of %d."%(n,N))

    Y = np.array(Y[:-1]) # Eliminate last timestep (overshoots tf in the while loop).
    return Y, t

def adams_bashforth2(f, y0, dt=0.1, ti=0., tf=100., verbose=False):
    """
    Adams-Bashforth 2 (AB2) scheme.
    """
    assert callable(f), "'f' is not a function."
    y0, dt, ti, tf = map(np.float, (y0, dt, ti, tf))
    t = np.arange(ti, tf+dt, dt)  # Create time axis.
    c1, c2 = 3/2., -1/2. # Set AB2 coefficients.

    # Initialize first time step with RK4.
    Y, tstart = runge_kutta4(f, y0, dt=dt, ti=t[0], tf=t[1], verbose=False)
    Y = Y.tolist()

    N = t.size
    yn, tn = Y[-1], tstart[-1]
    fnm = f(Y[-2], tstart[-2])
    n = 1
    while n<N:
        fn = f(yn, tn)   # Evaluate the RHS at time t = tn.
        ynp = yn + c1*dt*fn + c2*dt*fnm
        Y.append(ynp)    # Store the solution at  t = tn + dt.
        yn=ynp           # Update the value of the solution at t = tn.
        fnm = fn         # Keep the RHS at t = tn for next time step.
        tn+=dt           # Update time.
        n+=1             # Update counter.
        if verbose:
            print("Time-stepping: Step %d of %d."%(n,N))

    Y = np.array(Y[:-1]) # Eliminate last timestep (overshoots tf in the while loop).
    return Y, t
