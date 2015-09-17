#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the Euler equations

This module contains Riemann solvers for the Euler equations which have the
form (in 1d):

.. math:: 
    q_t + f(q)_x = 0
  
where 

.. math:: 
    q(x,t) = \left [ \begin{array}{c} \rho \\ \rho u \\ E \end{array} \right ],
  
the flux function is 

.. math:: 
    f(q) = \left [ \begin{array}{c} \rho u \\ \rho u^2 + p \\ u(E+p) \end{array}\right ].

and :math:`\rho` is the density, :math:`u` the velocity, :math:`E` is the 
energy and :math:`p` is the pressure.

Unless otherwise noted, the ideal gas equation of state is used:

.. math::
    E = (\gamma - 1) \left (E - \frac{1}{2}\rho u^2 \right)

:Authors:
    Kyle T. Mandli (2009-06-26): Initial version
    Kyle T. Mandli (2011-03-28): Interleaved version
"""
# ============================================================================
#      Copyright (C) 2009 Kyle T. Mandli <mandli@amath.washington.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD) 
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import numpy as np
import cython
cimport numpy as np
from libc.math cimport sqrt
import time

cdef unsigned int num_eqn = 3
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def euler_roe_1D(np.ndarray[np.float64_t, ndim = 2] q_l,np.ndarray[np.float64_t, ndim = 2] q_r,aux_l,aux_r, double gamma1, int efix):
    r"""
    Roe Euler solver in 1d
    
    *aug_global* should contain -
     - *gamma* - (float) Ratio of the heat capacities
     - *gamma1* - (float) :math:`1 - \gamma`
     - *efix* - (bool) Whether to use an entropy fix or not
    
    See :ref:`pyclaw_rp` for more details.
    
    :Version: 1.0 (2009-6-26)
    """
    
    # Problem dimensions
    cdef unsigned int num_rp = q_l.shape[1]
    cdef unsigned int num_waves = 3

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) , dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] s = np.empty( (num_waves, num_rp), dtype=np.float64 )
    cdef np.ndarray[np.float64_t, ndim=2] amdq = np.zeros( (num_eqn, num_rp), dtype=np.float64 )
    cdef np.ndarray[np.float64_t, ndim=2] apdq = np.zeros( (num_eqn, num_rp), dtype=np.float64)

    # Calculate Roe averages
    cdef double t1 = time.time()
    cdef np.ndarray[np.float64_t, ndim=1] u, a, enthalpy
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.empty([3, num_rp], dtype=np.float64 )

    u, a, enthalpy = roe_averages(q_l,q_r,gamma1)[0:3]
    cdef double roe_time = time.time() - t1

    # Find eigenvector coefficients
    cdef unsigned int x = 0
    #delta = q_r - q_l
    for x in xrange(num_rp):
        delta[0, x] = q_r[0, x] - q_l[0, x]
        delta[1, x] = q_r[1, x] - q_l[1, x]
        delta[2, x] = q_r[2, x] - q_l[2, x]

    #a2 = gamma1 / a**2 * ((enthalpy -u**2)*delta[0,...] + u*delta[1,...] - delta[2,...])
    cdef np.ndarray[np.float64_t, ndim=1] a2 = np.empty(num_rp, dtype=np.float64)
    for x in xrange(num_rp):
        a2[x] = gamma1 / a[x]**2 * ((enthalpy[x] -u[x]**2)*delta[0,x] + u[x]*delta[1,x] - delta[2,x])

    #a3 = (delta[1,...] + (a-u) * delta[0,...] - a*a2) / (2.0*a)
    cdef np.ndarray[np.float64_t, ndim=1] a3 = np.empty(num_rp, dtype=np.float64)
    x = 0
    for x in xrange(num_rp):
        a3[x] = (delta[1,x] + (a[x]-u[x]) * delta[0,x] - a[x]*a2[x]) / (2.0*a[x])

    #a1 = delta[0,...] - a2 - a3
    cdef np.ndarray[np.float64_t, ndim=1] a1 = np.empty(num_rp, dtype=np.float64)
    for x in xrange(num_rp):
        a1[x] = delta[0,x] - a2[x] - a3[x]

    # Compute the waves
    wave[0,0,...] = a1
    #for x in range(num_rp):
    #    wave[0, 0, x] = a1[x]

    wave[1,0,...] = a1 * (u-a)
    #for x in xrange(num_rp):
    #    wave[1,0,x] = a1[x] * (u[x] - a[x])

    wave[2,0,...] = a1 * (enthalpy - u*a)
    #for x in xrange(num_rp):
    #    wave[2,0,x] = a1[x] * (enthalpy[x] - u[x] * a[x])

    s[0,...] = u - a
    #for x in xrange(num_rp):
    #    s[0, x] = u[x] - a[x]

    wave[0,1,...] = a2
    #for x in xrange(num_rp):
    #    wave[0,1,x] = a2[x]

    wave[1,1,...] = a2 * u
    #for x in xrange(num_rp):
    #    wave[1,1,x] = a2[x] * u[x]

    wave[2,1,...] = a2 * 0.5 * u**2
    #for x in xrange(num_rp):
    #    wave[2,1,x] = a2[x] * 0.5 * u[x]**2

    s[1,...] = u
    #for x in xrange(num_rp):
    #   s[1,x] = u[x]

    wave[0,2,...] = a3
    #for x in xrange(num_rp):
    #    wave[0,2,x] = a3[x]

    wave[1,2,...] = a3 * (u+a)
    #for x in xrange(num_rp):
    #    wave[1,2,x] = a3[x] * (u[x]+a[x])

    wave[2,2,...] = a3 * (enthalpy + u*a)
    #for x in xrange(num_rp):
    #    wave[2,2,x] = a3[x] * (enthalpy[x] + u[x] * a[x])

    s[2,...] = u + a
    #for x in xrange(num_rp):
    #    s[2,x] = u[x] + a[x]

    # Entropy fix
    if efix:
        raise NotImplementedError("Entropy fix has not been implemented!")
    else:
        # Godunov update
        godunov_update2(num_rp, num_eqn, num_waves, amdq, apdq, wave, s)

    return wave,s,amdq,apdq,roe_time

def godunov_update1(int num_rp, int num_eqn, int num_waves, np.ndarray[np.float64_t, ndim=2] amdq, np.ndarray[np.float64_t, ndim=2] apdq, np.ndarray[np.float64_t, ndim=3] wave, np.ndarray[np.float64_t, ndim=2] s):
    # Godunov update
    cdef np.ndarray s_index = np.zeros((2,num_rp), dtype=np.float64)
    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]

@cython.boundscheck(False) # turn of bounds-checking for entire function
def godunov_update2(int num_rp, int num_eqn, int num_waves, np.ndarray[np.float64_t, ndim = 2] amdq, np.ndarray[np.float64_t, ndim = 2] apdq, np.ndarray[np.float64_t, ndim = 3] wave, np.ndarray[np.float64_t, ndim = 2] s):
    # Godunov update
    cdef unsigned int m, mw, i, j, k
    cdef float min_val, max_val
    cdef unsigned int wave2 = wave.shape[2]

    cdef np.ndarray[np.float64_t, ndim = 2] s_index = np.zeros([2,num_rp], dtype=np.float64)
    cdef unsigned int s_index0 = s_index.shape[0]
    cdef unsigned int s_index1 = s_index.shape[1]

    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            # s_index[0,:] = s[mw,:]
            for i in range(s_index1):
                s_index[0, i] = s[mw,i]

            # amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            # apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            for j in range(wave2):
                min_val = max_val = s_index[0, j]
                for k in range(1, s_index0):
                     min_val = min(min_val, s_index[k,j])
                     max_val = max(max_val, s_index[k,j])

                amdq[m,j] += min_val * wave[m,mw,j]
                apdq[m,j] += max_val * wave[m,mw,j]

def euler_hll_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLL euler solver ::
    
         
        W_1 = Q_hat - Q_l    s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = Q_r - Q_hat    s_2 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)
    
        Q_hat = ( f(q_r) - f(q_l) - s_2 * q_r + s_1 * q_l ) / (s_1 - s_2)
    
    *problem_data* should contain:
     - *gamma* - (float) Ratio of the heat capacities
     - *gamma1* - (float) :math:`1 - \gamma`

    :Version: 1.0 (2014-03-04)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 2

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    
    # Solver parameters
    gamma1 = problem_data['gamma1']
    
    # Calculate Roe averages, right and left speeds
    u, a, _, pl, pr = roe_averages(q_l,q_r,problem_data)
    H_r = (q_r[2,:] + pr) / q_r[0,:]
    H_l = (q_l[2,:] + pl) / q_l[0,:]
    u_r = q_r[1,:] / q_r[0,:]
    u_l = q_l[1,:] / q_l[0,:]
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[1,:] = np.max(s_index,axis=0)

    # Compute middle state
    q_hat = np.empty((num_eqn,num_rp))
    q_hat[0,:] = (q_r[1,:] - q_l[1,:] 
                    - s[1,:] * q_r[0,:] + s[0,:] * q_l[0,:]) / (s[0,:] - s[1,:])
    q_hat[1,:] = (q_r[1,:]**2/q_r[0,:] + pr - (q_l[1,:]**2/q_l[0,:] + pl)
                    - s[1,:] * q_r[1,:] + s[0,:] * q_l[1,:]) / (s[0,:] - s[1,:])
    q_hat[2,:] = ((q_r[2,:] + pr)*q_r[1,:]/q_r[0,:] - (q_l[2,:] + pl)*q_l[1,:]/q_l[0,:] 
                    - s[1,:] * q_r[2,:] + s[0,:] * q_l[2,:]) / (s[0,:] - s[1,:])

    # Compute each family of waves
    wave[:,0,:] = q_hat - q_l
    wave[:,1,:] = q_r - q_hat
    
    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            
    return wave, s, amdq, apdq

def euler_exact_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    Exact euler Riemann solver
    
    .. warning::
        This solver has not been implemented.
    
    """
    raise NotImplementedError("The exact Riemann solver has not been implemented.")

def roe_averages(q_l,q_r,gamma1):
    # Calculate Roe averages
    rhsqrtl = np.sqrt(q_l[0,...])
    rhsqrtr = np.sqrt(q_r[0,...])
    pl = gamma1 * (q_l[2,...] - 0.5 * (q_l[1,...]**2) / q_l[0,...])
    pr = gamma1 * (q_r[2,...] - 0.5 * (q_r[1,...]**2) / q_r[0,...])
    rhsq2 = rhsqrtl + rhsqrtr
    u = (q_l[1,...] / rhsqrtl + q_r[1,...] / rhsqrtr) / rhsq2
    enthalpy = ((q_l[2,...] + pl) / rhsqrtl + (q_r[2,...] + pr) / rhsqrtr) / rhsq2
    a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))

    return u, a, enthalpy, pl, pr

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def roe_averages2(np.ndarray[np.float64_t, ndim = 2] q_l,np.ndarray[np.float64_t, ndim = 2] q_r, double gamma1):
    # Calculate Roe averages
    cdef unsigned int x, q_l1, q_r1, rhsqrtl0
    q_l1 = q_l.shape[1]
    q_r1 = q_r.shape[1]

    #rhsqrtl = np.sqrt(q_l[0,...])
    cdef np.ndarray[np.float64_t, ndim = 1] rhsqrtl = np.empty(q_l1, dtype=np.float64)
    for x in range(q_l1):
        rhsqrtl[x] = sqrt(q_l[0, x])

    #rhsqrtr = np.sqrt(q_r[0,...])
    cdef np.ndarray[np.float64_t, ndim = 1] rhsqrtr = np.empty(q_r1, dtype=np.float64)
    for x in range(q_r1):
        rhsqrtr[x] = sqrt(q_r[0, x])

    #pl = gamma1 * (q_l[2,...] - 0.5 * (q_l[1,...]**2) / q_l[0,...])
    cdef np.ndarray[np.float64_t, ndim = 1] pl = np.empty(q_l1, dtype=np.float64)
    for x in range(q_l1):
        pl[x] =  gamma1 * (q_l[2, x] - 0.5 * (q_l[1, x]**2) / q_l[0, x])

    #pr = gamma1 * (q_r[2,...] - 0.5 * (q_r[1,...]**2) / q_r[0,...])
    cdef np.ndarray[np.float64_t, ndim = 1] pr = np.empty(q_r1, dtype=np.float64)
    for x in range(q_l1):
        pr[x] =  gamma1 * (q_r[2, x] - 0.5 * (q_r[1, x]**2) / q_r[0, x])

    #rhsq2 = rhsqrtl + rhsqrtr
    rhsqrtl0 = rhsqrtl.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 1] rhsq2 = np.empty(rhsqrtl0, dtype=np.float64)
    for x in range(rhsqrtl0):
        rhsq2[x] = rhsqrtl[x] + rhsqrtr[x]

    #u = (q_l[1,...] / rhsqrtl + q_r[1,...] / rhsqrtr) / rhsq2
    cdef np.ndarray[np.float64_t, ndim = 1] u = np.empty(q_l1, dtype=np.float64)
    for x in range(q_l1):
        u[x] =  (q_l[1,x] / rhsqrtl[x] + q_r[1,x] / rhsqrtr[x]) / rhsq2[x]

    #enthalpy = ((q_l[2,...] + pl) / rhsqrtl + (q_r[2,...] + pr) / rhsqrtr) / rhsq2
    cdef np.ndarray[np.float64_t, ndim = 1] enthalpy = np.empty(q_l1, dtype=np.float64)
    for x in range(q_l1):
        enthalpy[x] =  ((q_l[2,x] + pl[x]) / rhsqrtl[x] + (q_r[2,x] + pr[x]) / rhsqrtr[x]) / rhsq2[x]

    #a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))
    cdef np.ndarray[np.float64_t, ndim = 1] a = np.empty(q_l1, dtype=np.float64)
    for x in range(q_l1):
        a[x] =  sqrt(gamma1 * (enthalpy[x] - 0.5 * u[x]**2))

    return u, a, enthalpy, pl, pr

